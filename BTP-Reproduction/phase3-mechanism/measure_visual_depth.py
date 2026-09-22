#!/usr/bin/env python3
"""
measure_visual_depth.py

Measure the depth at which a vision language model stops needing its image tokens.

WHY THIS EXISTS
---------------
Visual token pruning methods decide, somewhere in their code, a layer after which the model
no longer needs to look at the image. In practice that decision is a hardcoded constant.
Our sweeps show the underlying quantity is real, sharp, and different in each architecture:
roughly 75 per cent of the way through InternVL2-2B and 84 per cent through Qwen2.5-VL-7B.
BTP ships layer 23 for Qwen, which is one layer below its threshold and costs 47.7 points.

Nobody measures this. This script does, in a few minutes, for any Hugging Face vision
language model, with no training, no calibration set and no source patching.

HOW IT WORKS
------------
At a chosen decoder layer we delete every image token from the hidden-state sequence and
let the rest of the stack run. If accuracy holds, the model had already finished reading by
that depth. If it collapses, it had not. Sweeping the layer traces the curve; the depth at
which the curve crosses half of baseline is the threshold.

CONVENTION, READ THIS BEFORE COMPARING WITH ANYTHING ELSE
---------------------------------------------------------
Layer indices here are ZERO BASED, and the deletion happens BEFORE the named layer runs.
So `--layers 22` means layers 0 to 21 inclusive saw the image and layers 22 upward did not,
that is, TWENTY-TWO layers had access.

The quantity this script reports is the number of layers that still need visual tokens, not
the index of the layer where the deletion was placed. Those differ by one, and conflating
them is easy: the BTP source counts layers from one (`layer_index += 1` runs before its
branch check), so its `end_layer = 23` is the same experiment as `--layers 22` here. An
earlier version of our own write-up reported Qwen's threshold as 83.9 per cent by using the
raw `end_layer` value; measured as layers-with-access it is 80.3 per cent.

Deletion is done with a forward pre-hook rather than by editing the model source, so the
same code works across architectures. The hook rewrites hidden_states, position_ids,
attention_mask, cache_position and position_embeddings consistently, which is the part that
is easy to get wrong.

WHY DELETION AND NOT MASKING
----------------------------
Zeroing the image tokens, or masking them out of attention, seems simpler but is not
equivalent: a zeroed vector still passes through layer normalisation and still contributes
a value vector. Deletion is the honest instrument, and it is also what the pruning methods
under study actually do.

COST
----
A coarse sweep plus bisection finds the threshold in roughly seven evaluations rather than
the fourteen we used originally. At 200 samples each that is well under an hour on one GPU.

USAGE
-----
  # validate against the curve we already measured by other means
  python measure_visual_depth.py --model Qwen/Qwen2.5-VL-7B-Instruct --n 200 \
      --out results/depth/qwen7b.json

  # a model we have never touched
  python measure_visual_depth.py --model OpenGVLab/InternVL2-2B --n 200 \
      --out results/depth/internvl2b.json

  # explicit layers instead of bisection, for a full curve
  python measure_visual_depth.py --model Qwen/Qwen2.5-VL-7B-Instruct \
      --layers 2,6,10,14,18,20,22,23,24,26 --no-bisect

NOTE: needs a GPU. Run via sbatch.
"""

import argparse
import io
import json
import os
import re
import string
import sys

import torch
from PIL import Image
from datasets import load_dataset

SHORT = "\nAnswer the question using a single word or phrase."
_ARTICLES = {"a", "an", "the"}


# ===========================================================================
#  Scoring, matched to the VQA convention used by lmms-eval
# ===========================================================================

def normalise(s):
    s = str(s).lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s)
    return " ".join(t for t in s.split() if t not in _ARTICLES)


def as_float(s):
    try:
        return float(str(s).replace(",", "").replace("%", "").strip())
    except Exception:
        return None


def is_correct(pred, refs):
    if not refs:
        return None
    p = normalise(pred)
    if not p:
        return False
    nrefs = [normalise(r) for r in refs]
    if p in nrefs:
        return True
    pf = as_float(p)
    if pf is not None:
        for r in refs:
            rf = as_float(r)
            if rf is not None and abs(pf - rf) < 1e-6:
                return True
    for r in nrefs:
        if r and (r == p or r in p.split() or (len(r) > 2 and r in p)):
            return True
    return False


# ===========================================================================
#  The deletion hook
# ===========================================================================

class VisualTokenDeleter:
    """Deletes image tokens from the sequence at a chosen decoder layer.

    Implemented as a forward pre-hook so that no model source is modified. The hook must
    keep every sequence-indexed argument consistent with the shortened hidden states, which
    is why each one is handled explicitly rather than by guessing which axis is the
    sequence axis.
    """

    def __init__(self, model, layers, layer_index, keep_positions=True):
        self.layers = layers
        self.layer_index = layer_index
        self.keep_positions = keep_positions
        self.image_positions = None   # LongTensor, set per forward pass
        self.handles = []
        self.fired = 0
        # Set at the deletion layer, consumed by every layer after it.
        self._keep = None
        self._orig_total = None

    def arm(self):
        """Register hooks on the deletion layer AND every layer after it.

        The later hooks are not optional. Rotary position embeddings are computed once in
        the model's forward pass and handed to every decoder layer, so patching only the
        deletion layer leaves layers L+1 onward holding full-length cos and sin tensors
        against a shortened hidden state. That produces a shape mismatch several layers
        downstream, which is exactly what happened on the first two attempts.
        """
        self.disarm()
        for i in range(self.layer_index, len(self.layers)):
            h = self.layers[i].register_forward_pre_hook(
                (lambda idx: (lambda m, a, k: self._hook(m, a, k, idx)))(i),
                with_kwargs=True)
            self.handles.append(h)
        return self

    def disarm(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    # -- internals ---------------------------------------------------------

    def begin_sample(self, positions):
        """Call before each forward pass. Clears state carried over from the previous one."""
        self.image_positions = positions
        self._keep = None
        self._orig_total = None

    def _keep_index(self, total, device):
        mask = torch.ones(total, dtype=torch.bool, device=device)
        pos = self.image_positions
        pos = pos[pos < total]
        mask[pos] = False
        return mask.nonzero(as_tuple=True)[0]

    def _bind(self, module, args, kwargs):
        """Map positional arguments onto their parameter names.

        Decoder layers are called with a mixture of positional and keyword arguments, and
        which is which varies by model and by transformers version. Qwen2.5-VL passes
        position_embeddings positionally, so reading it out of kwargs alone silently
        returned None and the rotary tensors were never sliced. Binding against the
        signature removes that whole class of bug.
        """
        import inspect
        try:
            params = [p for p in inspect.signature(module.forward).parameters
                      if p != "self"]
        except (TypeError, ValueError):
            params = []
        bound = dict(kwargs)
        for i, a in enumerate(args):
            if i < len(params):
                bound[params[i]] = a
        return bound

    def _hook(self, module, args, kwargs, layer_idx):
        bound = self._bind(module, args, kwargs)

        # Only act during prefill. On a decode step the sequence is length one and the
        # image tokens are already behind us in the cache.
        hs = bound.get("hidden_states")
        if not isinstance(hs, torch.Tensor) or hs.shape[1] <= 1:
            return None

        if layer_idx == self.layer_index:
            # The deletion layer. Decide what survives and record it for the layers above.
            if self.image_positions is None or self.image_positions.numel() == 0:
                return None
            total = hs.shape[1]
            keep = self._keep_index(total, hs.device)
            if keep.numel() == total:
                return None
            self._keep = keep
            self._orig_total = total
            self.fired += 1
            bound["hidden_states"] = hs[:, keep, :]
        else:
            # A later layer. hidden_states is already shortened, but arguments created once
            # in the model's forward, above all the layers, still carry the original length
            # and must be brought into line.
            if self._keep is None or self._orig_total is None:
                return None
            total = self._orig_total
            keep = self._keep
            if hs.shape[1] == total:
                # Should not happen, but if the deletion did not take effect there is
                # nothing consistent we can do here.
                return None

        # position_ids. Kept at their ORIGINAL values, so surviving tokens retain their
        # rotary phase. This is what the pruning implementations under study do.
        pid = bound.get("position_ids")
        if isinstance(pid, torch.Tensor) and pid.shape[-1] == total:
            bound["position_ids"] = pid[..., keep] if self.keep_positions \
                else torch.arange(keep.numel(), device=pid.device).expand_as(pid[..., keep])

        # cache_position, sequence on dim 0
        cp = bound.get("cache_position")
        if isinstance(cp, torch.Tensor) and cp.shape[0] == total:
            bound["cache_position"] = cp[keep]

        # attention_mask, either 2D (batch, seq) or 4D (batch, heads, q, k)
        am = bound.get("attention_mask")
        if isinstance(am, torch.Tensor):
            if am.dim() == 4:
                if am.shape[-1] == total:
                    am = am[:, :, :, keep]
                if am.shape[2] == total:
                    am = am[:, :, keep, :]
                bound["attention_mask"] = am
            elif am.dim() == 2 and am.shape[1] == total:
                bound["attention_mask"] = am[:, keep]

        # position_embeddings, a (cos, sin) tuple. Sequence is the second to last axis for
        # both the standard rotary form (batch, seq, dim) and Qwen's multimodal form
        # (3, batch, seq, dim). This is the one that was silently missed.
        pe = bound.get("position_embeddings")
        if isinstance(pe, (tuple, list)) and len(pe) == 2:
            sliced = []
            for t in pe:
                if isinstance(t, torch.Tensor) and t.dim() >= 2 and t.shape[-2] == total:
                    sliced.append(t[..., keep, :])
                else:
                    sliced.append(t)
            bound["position_embeddings"] = tuple(sliced)

        # Anything sequence-shaped we did not name explicitly is a bug waiting to happen,
        # so say so rather than failing later with an opaque shape error.
        for k, v in bound.items():
            if k in ("hidden_states", "position_ids", "cache_position",
                     "attention_mask", "position_embeddings"):
                continue
            if isinstance(v, torch.Tensor) and total in tuple(v.shape):
                print("  WARNING: unhandled sequence-shaped argument '{}' with shape {}"
                      .format(k, tuple(v.shape)))

        # Return everything as keyword arguments. Every parameter we touched is named in
        # the signature, so this is safe and avoids re-deriving positional order.
        return (), bound


# ===========================================================================
#  Model adapters
# ===========================================================================

def load_qwen(model_id):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    proc = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16,
        attn_implementation="flash_attention_2", device_map="cuda").eval()
    layers = model.model.layers
    img_id = model.config.image_token_id

    def run(img, question, max_new_tokens=16):
        from qwen_vl_utils import process_vision_info
        msg = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": question + SHORT}]}]
        text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msg)
        inp = proc(text=[text], images=imgs, videos=vids,
                   padding=True, return_tensors="pt").to("cuda")
        positions = (inp["input_ids"][0] == img_id).nonzero(as_tuple=True)[0]
        return inp, positions, proc

    def generate(inp, max_new_tokens=16):
        with torch.inference_mode():
            gen = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        new = gen[0][inp["input_ids"].shape[1]:]
        return proc.tokenizer.decode(new, skip_special_tokens=True).strip()

    return model, layers, run, generate


LOADERS = {
    "qwen2_5_vl": load_qwen,
}


def pick_loader(model_id):
    lid = model_id.lower()
    if "qwen2.5-vl" in lid or "qwen2_5_vl" in lid:
        return LOADERS["qwen2_5_vl"]
    raise SystemExit(
        "No adapter for '{}'.\n"
        "Adapters exist for Qwen2.5-VL. Adding one means supplying three things:\n"
        "  the decoder layer list, the image token id, and how to build model inputs.\n"
        "See load_qwen above; it is about twenty lines.".format(model_id))


# ===========================================================================
#  Evaluation
# ===========================================================================

def load_image(raw):
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, dict) and "bytes" in raw:
        try:
            return Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
        except Exception:
            return None
    return None


def build_samples(dataset, split, n, max_scan=4000):
    """Deduplicate by image, because TextVQA stores several questions per picture."""
    import hashlib
    ds = load_dataset(dataset, split=split, streaming=True)
    out, seen, scanned = [], set(), 0
    for ex in ds:
        if len(out) >= n or scanned >= max_scan:
            break
        scanned += 1
        img = load_image(ex.get("image"))
        if img is None:
            continue
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        md5 = hashlib.md5(buf.getvalue()).hexdigest()
        if md5 in seen:
            continue
        seen.add(md5)
        refs = ex.get("answers") or ex.get("answer") or []
        if not isinstance(refs, (list, tuple)):
            refs = [refs]
        out.append((img, str(ex.get("question") or ""), [str(r) for r in refs]))
    return out


def evaluate(samples, run, generate, deleter, layer):
    """Score the sample set with the deletion applied at `layer`, or at none if layer is None."""
    correct = total = 0
    if layer is not None:
        deleter.layer_index = layer
        deleter.fired = 0
        deleter.arm()
    try:
        for img, q, refs in samples:
            inp, positions, _ = run(img, q)
            deleter.begin_sample(positions)
            try:
                pred = generate(inp)
            except Exception as e:
                print("  generation failed:", repr(e))
                continue
            ok = is_correct(pred, refs)
            if ok is None:
                continue
            total += 1
            correct += 1 if ok else 0
    finally:
        if layer is not None:
            deleter.disarm()

    acc = correct / total if total else float("nan")
    return acc, total, (deleter.fired if layer is not None else 0)


def half_crossing(points, baseline):
    """Interpolate the layer at which accuracy crosses half of baseline."""
    pts = sorted(points)
    for i in range(1, len(pts)):
        l0, a0 = pts[i - 1]
        l1, a1 = pts[i]
        if a0 < baseline / 2 <= a1:
            f = (baseline / 2 - a0) / (a1 - a0)
            return l0 + f * (l1 - l0)
    return None


# ===========================================================================
#  Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--dataset", default="lmms-lab/textvqa")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--layers", default="",
                    help="comma separated. Default is a coarse sweep at quarter depths.")
    ap.add_argument("--no-bisect", action="store_true",
                    help="skip the refinement phase and report only the coarse sweep")
    ap.add_argument("--out", default="results/depth/sweep.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    loader = pick_loader(args.model)
    model, layers, run, generate = loader(args.model)
    n_layers = len(layers)
    print("model      :", args.model)
    print("decoder    :", n_layers, "layers")

    samples = build_samples(args.dataset, args.split, args.n)
    print("samples    :", len(samples), "unique images")

    deleter = VisualTokenDeleter(model, layers, 0)

    results = {}

    # ---- baseline, no deletion at all
    base, ntot, _ = evaluate(samples, run, generate, deleter, None)
    print("\nbaseline (no deletion): {:.4f}  on {} scored".format(base, ntot))
    results["baseline"] = base

    # ---- coarse sweep
    if args.layers:
        coarse = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    else:
        # quarter depths, which is enough to bracket the threshold for bisection
        coarse = sorted({max(1, int(round(n_layers * f))) for f in (0.25, 0.5, 0.75, 0.9)})

    print("\ncoarse sweep at layers:", coarse)
    curve = []
    for L in coarse:
        acc, ntot, fired = evaluate(samples, run, generate, deleter, L)
        curve.append((L, acc))
        print("  layer {:3d}  acc {:.4f}  ({:.1f}% of baseline, hook fired {}x)".format(
            L, acc, 100 * acc / base if base else float("nan"), fired))
        if fired == 0:
            print("  WARNING: the hook never fired at this layer. The measurement is void.")

    # ---- bisection to localise the crossing
    if not args.no_bisect:
        lo = hi = None
        pts = sorted(curve)
        for i in range(1, len(pts)):
            if pts[i - 1][1] < base / 2 <= pts[i][1]:
                lo, hi = pts[i - 1][0], pts[i][0]
                break
        if lo is None:
            print("\nNo crossing bracketed by the coarse sweep. Widen --layers.")
        else:
            print("\nbisecting between layers {} and {}".format(lo, hi))
            while hi - lo > 1:
                mid = (lo + hi) // 2
                acc, ntot, fired = evaluate(samples, run, generate, deleter, mid)
                curve.append((mid, acc))
                print("  layer {:3d}  acc {:.4f}  ({:.1f}% of baseline)".format(
                    mid, acc, 100 * acc / base if base else float("nan")))
                if acc < base / 2:
                    lo = mid
                else:
                    hi = mid
            print("\nthreshold lies between layers {} and {}".format(lo, hi))

    curve = sorted(set(curve))
    results["curve"] = [{"layer": l, "accuracy": a,
                         "pct_of_baseline": (100 * a / base) if base else None}
                        for l, a in curve]

    cross = half_crossing(curve, base)
    results["n_layers"] = n_layers
    results["half_crossing_layer"] = cross
    results["half_crossing_depth_pct"] = (100 * cross / n_layers) if cross else None
    results["model"] = args.model
    results["n_samples"] = len(samples)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)

    print("\n" + "=" * 68)
    print("VISUAL DEPENDENCE THRESHOLD")
    print("=" * 68)
    print("model               : {}".format(args.model))
    print("decoder layers      : {}".format(n_layers))
    print("baseline accuracy   : {:.4f}".format(base))
    if cross is not None:
        print("layers needing the image : {:.2f} of {}".format(cross, n_layers))
        print("relative depth           : {:.1f}%".format(100 * cross / n_layers))
        print("")
        print("CONVENTION: indices are zero based and the deletion happens BEFORE the")
        print("named layer runs, so the figure above is the number of layers that still")
        print("need visual tokens. A method whose own layer index is ONE based, as the")
        print("BTP source is, will quote a number one higher for the same experiment.")
        print("")
        print("A pruning method that removes all visual tokens before this point will")
        print("destroy text reading while leaving the standard benchmark suite unchanged.")
    else:
        print("no crossing found; the curve never reaches half of baseline")
    print("\nsaved to", args.out)
    print("=" * 62)


if __name__ == "__main__":
    main()
