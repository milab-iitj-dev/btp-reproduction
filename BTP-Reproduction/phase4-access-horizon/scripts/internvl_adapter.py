#!/usr/bin/env python3
"""
InternVL2 adapter for the visual-token deletion hook (measure_visual_depth.VisualTokenDeleter).

Supplies the same three things as load_qwen: the decoder layer list, a way to build inputs,
and a way to generate. Two InternVL-specific problems have to be handled.

1. Image positions. InternVL builds inputs_embeds itself and hands the language model no
   input_ids, so the positions of the image tokens cannot be read from inside the decoder.
   InternVLChatModel.generate(pixel_values, input_ids, ...) does see input_ids, so it is
   wrapped: the wrapper finds the IMG_CONTEXT tokens and passes their positions to the
   deleter before any decoder layer runs.

2. Rotary cache. InternLM2 builds cos/sin from the LIVE sequence length
       cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
   and then gathers them with the ORIGINAL position ids. After deletion the live length is
   shorter than the largest position id, the gather runs off the end, and CUDA reports a
   device-side assert at an unrelated line. This is the bug found in E10 bring-up. Here it
   is fixed without editing source: every layer's rotary_emb.forward is wrapped to build at
   least (prompt length + generation budget) positions. Returning a longer table is inert
   for layers that were not shortened, because the gather only reads the ids it is given.

Image preprocessing is InternVL's own dynamic tiling at 448 px (thumbnail on), max_num
tiles set by IVL_MAX_NUM, default 6.

Requires the PRISTINE remote code. The E10 jobs patch the Hub snapshot in place; the job
script restores .BASELINE and clears the transformers module cache before loading.
"""

import os
import types

import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
#  Preprocessing, as in the InternVL2 model card
# ---------------------------------------------------------------------------

def _transform(size):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    return T.Compose([
        T.Lambda(lambda im: im.convert("RGB")),
        T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])


def _closest_ratio(ar, ratios, w, h, size):
    best, best_diff = (1, 1), float("inf")
    area = w * h
    for r in ratios:
        tar = r[0] / r[1]
        d = abs(ar - tar)
        if d < best_diff:
            best_diff, best = d, r
        elif d == best_diff and area > 0.5 * size * size * r[0] * r[1]:
            best = r
    return best


def _dynamic_tiles(img, min_num=1, max_num=6, size=448, thumbnail=True):
    w, h = img.size
    ar = w / h
    ratios = sorted({(i, j) for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1) for j in range(1, n + 1)
                     if min_num <= i * j <= max_num}, key=lambda x: x[0] * x[1])
    r = _closest_ratio(ar, ratios, w, h, size)
    tw, th = size * r[0], size * r[1]
    blocks = r[0] * r[1]
    im = img.resize((tw, th))
    out = []
    for i in range(blocks):
        box = ((i % (tw // size)) * size, (i // (tw // size)) * size,
               ((i % (tw // size)) + 1) * size, ((i // (tw // size)) + 1) * size)
        out.append(im.crop(box))
    if thumbnail and len(out) != 1:
        out.append(img.resize((size, size)))
    return out


# ---------------------------------------------------------------------------
#  Loader
# ---------------------------------------------------------------------------

def load_internvl(model_id):
    from transformers import AutoModel, AutoTokenizer
    max_num = int(os.environ.get("IVL_MAX_NUM", "6"))
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True).eval().cuda()
    layers = model.language_model.model.layers
    tf = _transform(448)
    print("internvl   : max_num tiles =", max_num, "  layers =", len(layers))

    state = {"deleter": None, "min_len": 0, "budget": 64, "img_id": None}

    # ---- 2. rotary cache must span the original positions
    n_wrapped = 0
    for layer in layers:
        rot = layer.attention.rotary_emb
        if getattr(rot, "_p4_wrapped", False):
            continue
        orig = rot.forward

        def fwd(self_, x, *args, _orig=orig, **kw):
            if "seq_len" in kw and kw["seq_len"] is not None:
                kw["seq_len"] = max(int(kw["seq_len"]), state["min_len"])
            elif args and args[0] is not None:
                args = (max(int(args[0]), state["min_len"]),) + tuple(args[1:])
            return _orig(x, *args, **kw)

        rot.forward = types.MethodType(fwd, rot)
        rot._p4_wrapped = True
        n_wrapped += 1
    if n_wrapped != len(layers):
        raise SystemExit("FATAL: wrapped {} rotary modules for {} layers".format(n_wrapped, len(layers)))

    # ---- 1. positions of the image tokens, captured where input_ids is still visible
    orig_generate = model.generate

    def generate_wrapped(*args, **kw):
        input_ids = kw.get("input_ids")
        if input_ids is None and len(args) >= 2:
            input_ids = args[1]
        if state["img_id"] is None:
            state["img_id"] = model.img_context_token_id
        pos = (input_ids[0] == state["img_id"]).nonzero(as_tuple=True)[0]
        state["min_len"] = int(input_ids.shape[1]) + state["budget"] + 8
        if state["deleter"] is not None:
            state["deleter"].begin_sample(pos)
        state["last_n_img"] = int(pos.numel())
        return orig_generate(*args, **kw)

    model.generate = generate_wrapped

    def run(img, question, max_new_tokens=16, suffix="", max_pixels=None):
        # max_pixels is Qwen-specific; InternVL's resolution is set by the tile count.
        tiles = _dynamic_tiles(img, max_num=max_num)
        pv = torch.stack([tf(t) for t in tiles]).to(torch.bfloat16).cuda()
        return {"pixel_values": pv, "question": "<image>\n" + question + suffix}, None, tok

    def generate(inp, max_new_tokens=16):
        state["budget"] = max_new_tokens
        cfg = dict(max_new_tokens=max_new_tokens, do_sample=False)
        with torch.inference_mode():
            out = model.chat(tok, inp["pixel_values"], inp["question"], cfg)
        return str(out).strip()

    def bind(deleter):
        state["deleter"] = deleter

    return model, layers, run, generate, bind, state
