#!/usr/bin/env python3
"""
Build BTP-style pruning variants of InternVL2-2B's remote modeling code.

WHY
---
The BTP authors released patches for LLaVA and Qwen2.5-VL only. To test whether our two
Qwen findings generalise to a different VLM family, we have to implement the pruning
ourselves. This script does that by textual injection into the two remote-code files that
InternVL2 ships on the Hub, exactly mirroring how the authors patch transformers.

WHAT WE ARE TESTING (see phase3-mechanism/RESULTS_LOG.md sections 5I and 5J)
---------------------------------------------------------------------------
Finding A: a final-layer total discard of image tokens is invisible on the standard VLM
           benchmark suite (GQA, MME, MMBench, POPE, SQA) but destroys text-reading tasks.
Finding B: graded pruning to 12.5% has a residual cost that scales with text density.

InternVL2-2B is the decisive cross-family test because it differs from Qwen2.5-VL on every
axis that could otherwise explain the result:
  - backbone        InternLM2-1.8B, 24 layers (Qwen2.5-VL-7B has 28)
  - positions       standard 1D RoPE, NOT Qwen's M-RoPE
  - visual encoding dynamic 448px tiling, 256 tokens per tile after pixel shuffle,
                    NOT Qwen's variable merged patch grid
If the pattern reproduces here, position handling and patch layout are ruled out.

MODES (env var IVL_MODE)
------------------------
  off         no pruning. Sanity check that the patched file still matches baseline.
  prune       graded pruning at IVL_PRUNE_LAYERS down to IVL_RETAIN. No final discard.
              This is the spec-compliant arm.
  wipe        no graded pruning, but delete ALL image tokens at IVL_WIPE_LAYER.
              This isolates Finding A with nothing else changing.
  prune_wipe  both. The analogue of BTP as shipped for Qwen.

LAYER CHOICE
------------
Qwen2.5-VL-7B (28 layers) prunes at 4, 7, 16 and wipes at 23. Scaled to 24 layers that is
3.4, 6.0, 13.7 and 19.7, so we use 3, 6, 14 and 20. This is a proportional transfer, NOT
the paper's calibration procedure (Section 4.3 uses 64 calibration samples to find layers
where image-token representations shift). Running that calibration is out of scope here and
the choice is reported as a limitation.

SELECTOR
--------
We implement the diversity component only (farthest-point sampling on cosine distance),
which is BTP's dominant selector at stage 1 where roughly 90% of the budget is filled by
div_prune. The attention component is omitted because InternLM2's FlashAttention-2 path does
not expose attention scores without further surgery, and because our E1 selector ablation
cannot currently justify the extra complexity (E1 was run under the layer-23 confound and is
invalidated, see RESULTS_LOG).

This is stated plainly in the write-up: the InternVL arm tests the PRUNING REGIME, not a
faithful reimplementation of BTP's selector.

USAGE
-----
  python make_internvl_btp.py --snapshot /path/to/snapshots/<hash>

Writes, next to the originals:
  modeling_internlm2.py.BASELINE      copy of the original
  modeling_internlm2.py.BTP           patched
  modeling_internvl_chat.py.BASELINE  copy of the original
  modeling_internvl_chat.py.BTP       patched

Install an arm by copying the .BTP (or .BASELINE) files over the originals, then CLEARING
the remote-code module cache, which otherwise serves a stale copy:
  rm -rf $HF_HOME/modules/transformers_modules
"""

import argparse
import os
import shutil
import sys


# ---------------------------------------------------------------------------
# Block 1: module-level helpers, injected into modeling_internlm2.py
# ---------------------------------------------------------------------------
LM_HEADER = '''
# ======================= BTP-STYLE PRUNING (INJECTED) =======================
# See phase3-mechanism/make_internvl_btp.py for rationale. Configuration is read from the
# environment at import time, mirroring how the BTP authors' BTP_PARAM variant behaves.
import os as _btp_os

_IVL_MODE = _btp_os.environ.get("IVL_MODE", "off")
_IVL_PRUNE_LAYERS = tuple(
    int(x) for x in _btp_os.environ.get("IVL_PRUNE_LAYERS", "3,6,14").split(",") if x.strip() != ""
)
_IVL_WIPE_LAYER = int(_btp_os.environ.get("IVL_WIPE_LAYER", "20"))
_IVL_RETAIN = float(_btp_os.environ.get("IVL_RETAIN", "0.125"))
_IVL_DEBUG = _btp_os.environ.get("IVL_DEBUG", "0") == "1"

# Largest position id the current forward pass will use, i.e. the sequence length BEFORE
# pruning. InternLM2 sizes its rotary cache from the live sequence length and then gathers
# cos[position_ids]. Because pruning shortens the sequence while deliberately keeping the
# ORIGINAL position ids, that gather runs off the end of the cache and trips a device-side
# assert. We therefore grow the cache to cover the original range. Set to 0 when no pruning
# has happened, which makes the max() below a no-op and keeps the 'off' arm inert.
_BTP_MAX_POS = 0

# Per-stage keep fraction, so that the product over all stages equals IVL_RETAIN.
_IVL_STAGE_F = (
    _IVL_RETAIN ** (1.0 / len(_IVL_PRUNE_LAYERS)) if len(_IVL_PRUNE_LAYERS) > 0 else 1.0
)

print(
    "[BTP-IVL] mode=%s prune_layers=%s wipe_layer=%s retain=%s stage_f=%.4f"
    % (_IVL_MODE, _IVL_PRUNE_LAYERS, _IVL_WIPE_LAYER, _IVL_RETAIN, _IVL_STAGE_F)
)


def _btp_div_prune(x, k):
    """Farthest-point sampling on cosine distance.

    Greedily picks the token whose minimum distance to the already-selected set is largest,
    which is the Max-Min Diversity heuristic BTP uses. x is (n, d), returns k indices.
    """
    import torch as _t

    n = x.shape[0]
    if k >= n:
        return _t.arange(n, device=x.device)
    if k <= 0:
        return _t.empty(0, dtype=_t.long, device=x.device)

    xn = _t.nn.functional.normalize(x.float(), dim=-1)
    dist = 1.0 - (xn @ xn.t())

    sel = _t.zeros(k, dtype=_t.long, device=x.device)
    # Seed with the token that is on average most distant from everything else.
    sel[0] = dist.sum(dim=1).argmax()
    mind = dist[sel[0]].clone()
    mind[sel[0]] = -1.0
    for i in range(1, k):
        nxt = mind.argmax()
        sel[i] = nxt
        mind = _t.minimum(mind, dist[nxt])
        mind[sel[: i + 1]] = -1.0
    return sel


def _btp_plan(self, idx, hidden_states):
    """Return indices to KEEP at layer `idx`, or None if this layer does not prune.

    State lives on the InternLM2Model instance:
      self._btp_img_pos  LongTensor, positions of surviving image tokens in the CURRENT
                         (already possibly pruned) sequence.
    """
    import torch as _t

    pos = getattr(self, "_btp_img_pos", None)
    if pos is None or pos.numel() == 0:
        return None

    total = hidden_states.shape[1]
    is_prune = (_IVL_MODE in ("prune", "prune_wipe")) and (idx in _IVL_PRUNE_LAYERS)
    is_wipe = (_IVL_MODE in ("wipe", "prune_wipe")) and (idx == _IVL_WIPE_LAYER)

    if not (is_prune or is_wipe):
        return None

    all_idx = _t.arange(total, device=hidden_states.device)
    img_mask = _t.zeros(total, dtype=_t.bool, device=hidden_states.device)
    img_mask[pos] = True
    non_img = all_idx[~img_mask]

    if is_wipe:
        # Total discard: every remaining image token goes. This is the InternVL analogue of
        # the released Qwen code's `elif layer_index == self.end_layer` branch.
        self._btp_img_pos = _t.empty(0, dtype=_t.long, device=hidden_states.device)
        keep = non_img
        if _IVL_DEBUG:
            print("[BTP-IVL] layer %2d WIPE   seq %4d -> %4d   img %4d -> 0"
                  % (idx, total, keep.numel(), pos.numel()))
        # Recompute surviving image positions in the new indexing: none.
        return keep

    # Graded pruning: keep a fraction of the surviving image tokens, chosen for diversity.
    k = int(round(pos.numel() * _IVL_STAGE_F))
    k = max(1, min(k, pos.numel()))
    local = _btp_div_prune(hidden_states[0, pos, :], k)
    kept_img = pos[local]

    keep = _t.cat([non_img, kept_img]).sort().values

    # Remap surviving image positions into the new, shorter indexing.
    new_mask = _t.zeros(total, dtype=_t.bool, device=hidden_states.device)
    new_mask[kept_img] = True
    self._btp_img_pos = new_mask[keep].nonzero(as_tuple=True)[0]
    if _IVL_DEBUG:
        print("[BTP-IVL] layer %2d PRUNE  seq %4d -> %4d   img %4d -> %4d"
              % (idx, total, keep.numel(), pos.numel(), kept_img.numel()))
    return keep
# ===================== END BTP-STYLE PRUNING (INJECTED) =====================

'''


# ---------------------------------------------------------------------------
# Block 2: the in-loop hook, injected into InternLM2Model.forward
# ---------------------------------------------------------------------------
LM_LOOP_ANCHOR = (
    "            past_key_value = past_key_values[idx] if past_key_values is not None else None\n"
)

LM_LOOP_INJECT = '''
            # ---------------- BTP-STYLE PRUNING (INJECTED) ----------------
            # Only during prefill. seq_length == 1 means we are decoding, and the image
            # tokens are already gone from the live sequence at that point.
            if _IVL_MODE != "off" and seq_length > 1:
                _keep = _btp_plan(self, idx, hidden_states)
                if _keep is not None:
                    hidden_states = hidden_states[:, _keep, :]
                    # Keep ORIGINAL position ids for the surviving tokens, so RoPE phases
                    # are unchanged. This matches what the released Qwen BTP code does.
                    position_ids = position_ids[:, _keep]
                    if attention_mask is not None:
                        if attention_mask.dim() == 4:
                            attention_mask = attention_mask[:, :, _keep, :][:, :, :, _keep]
                        elif attention_mask.dim() == 2:
                            attention_mask = attention_mask[:, _keep]
            # -------------- END BTP-STYLE PRUNING (INJECTED) --------------
'''


# ---------------------------------------------------------------------------
# Block 2b: rotary cache fix, injected into BOTH attention classes
# ---------------------------------------------------------------------------
# InternLM2Attention and InternLM2FlashAttention2 both do:
#     cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
# and rotary_emb returns cos_cached[:seq_len]. apply_rotary_pos_emb then indexes that with
# position_ids. Pruning makes kv_seq_len smaller than max(position_ids)+1, so the gather
# reads past the end of the cache. On CUDA that surfaces as an async device-side assert,
# usually attributed to whatever op syncs next, which is why the first traceback pointed at
# an unrelated line in _btp_plan.
ATTN_ANCHOR = "        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)\n"

ATTN_REPLACE = (
    "        # BTP-INJECTED: pruning keeps ORIGINAL position ids on a shortened sequence,\n"
    "        # so the rotary cache must span the original range, not the live length.\n"
    "        cos, sin = self.rotary_emb(value_states, seq_len=max(kv_seq_len, _BTP_MAX_POS))\n"
)


# ---------------------------------------------------------------------------
# Block 2c: record the original position range, injected into InternLM2Model.forward
# ---------------------------------------------------------------------------
MAXPOS_ANCHOR = (
    "        if inputs_embeds is None:\n"
    "            inputs_embeds = self.tok_embeddings(input_ids)\n"
)

MAXPOS_INJECT = '''
        # ---------------- BTP-STYLE PRUNING (INJECTED) ----------------
        # position_ids run up to seq_length_with_past - 1, so the rotary cache needs at
        # least seq_length_with_past entries even after pruning shortens the sequence.
        global _BTP_MAX_POS
        _BTP_MAX_POS = int(seq_length_with_past)
        # -------------- END BTP-STYLE PRUNING (INJECTED) --------------
'''


# ---------------------------------------------------------------------------
# Block 3: stash the image-token mask, injected into modeling_internvl_chat.py
# ---------------------------------------------------------------------------
# InternVL calls language_model.generate(inputs_embeds=...), so input_ids never reaches
# InternLM2Model and the image tokens cannot be located there. We record their positions
# here, where the IMG_CONTEXT mask is already computed, and hang them on the InternLM2Model
# instance.
CHAT_ANCHOR = (
    "            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)\n"
    "\n"
    "            input_embeds = input_embeds.reshape(B, N, C)\n"
)

CHAT_INJECT = '''
            # ---------------- BTP-STYLE PRUNING (INJECTED) ----------------
            try:
                import torch as _t
                _sel2d = selected.reshape(B, N)
                self.language_model.model._btp_img_pos = _sel2d[0].nonzero(as_tuple=True)[0]
            except Exception as _e:
                print("[BTP-IVL] failed to stash image positions:", _e)
                self.language_model.model._btp_img_pos = None
            # -------------- END BTP-STYLE PRUNING (INJECTED) --------------
'''


def inject(text, anchor, block, label):
    n = text.count(anchor)
    if n != 1:
        raise SystemExit(
            "FATAL: anchor for {} matched {} times, expected exactly 1.\n"
            "The upstream file has changed. Update the anchor before running.\n"
            "Anchor was:\n{}".format(label, n, anchor)
        )
    return text.replace(anchor, anchor + block, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True,
                    help="HF snapshot dir containing modeling_internlm2.py")
    args = ap.parse_args()

    lm = os.path.join(args.snapshot, "modeling_internlm2.py")
    chat = os.path.join(args.snapshot, "modeling_internvl_chat.py")
    for p in (lm, chat):
        if not os.path.isfile(p):
            raise SystemExit("FATAL: not found: {}".format(p))

    # Keep pristine copies exactly once. If BASELINE already exists we must not overwrite
    # it with a possibly-already-patched current file.
    for p in (lm, chat):
        b = p + ".BASELINE"
        if not os.path.exists(b):
            shutil.copy2(p, b)
            print("saved baseline:", b)
        else:
            print("baseline already present, left alone:", b)

    # Always build the patched variant FROM the baseline, never from the live file.
    lm_src = open(lm + ".BASELINE", encoding="utf-8").read()
    chat_src = open(chat + ".BASELINE", encoding="utf-8").read()

    if "BTP-STYLE PRUNING (INJECTED)" in lm_src or "BTP-STYLE PRUNING (INJECTED)" in chat_src:
        raise SystemExit("FATAL: the .BASELINE files are already patched. Re-download the "
                         "snapshot, the pristine copy has been lost.")

    # Header goes BEFORE the class. Note the class carries an @add_start_docstrings
    # decorator, so the insertion point must be above the DECORATOR, not above the class
    # statement, or the decorator ends up dangling and the file will not parse.
    CLASS_ANCHOR = "# Modified from transformers.model.llama.modeling_llama.LlamaModel\n@add_start_docstrings("
    if lm_src.count(CLASS_ANCHOR) != 1:
        raise SystemExit("FATAL: InternLM2Model class anchor matched {} times, expected 1."
                         .format(lm_src.count(CLASS_ANCHOR)))
    lm_out = lm_src.replace(CLASS_ANCHOR, LM_HEADER + CLASS_ANCHOR, 1)
    if lm_out.count("_btp_div_prune") < 1:
        raise SystemExit("FATAL: header injection failed.")

    lm_out = inject(lm_out, LM_LOOP_ANCHOR, LM_LOOP_INJECT, "decoder loop hook")
    lm_out = inject(lm_out, MAXPOS_ANCHOR, MAXPOS_INJECT, "max-position record")

    # The rotary fix applies to BOTH attention implementations, so this anchor must match
    # exactly twice. A count of 1 would mean only one attention path is protected and the
    # other would still trip the device-side assert.
    n_attn = lm_out.count(ATTN_ANCHOR)
    if n_attn != 2:
        raise SystemExit("FATAL: rotary anchor matched {} times, expected exactly 2 "
                         "(eager and flash_attention_2 paths).".format(n_attn))
    lm_out = lm_out.replace(ATTN_ANCHOR, ATTN_REPLACE)

    chat_out = inject(chat_src, CHAT_ANCHOR, CHAT_INJECT, "image-mask stash")

    open(lm + ".BTP", "w", encoding="utf-8").write(lm_out)
    open(chat + ".BTP", "w", encoding="utf-8").write(chat_out)
    print("wrote:", lm + ".BTP")
    print("wrote:", chat + ".BTP")

    # Compile check. A syntax error here would otherwise surface as an opaque
    # trust_remote_code failure hours later inside a queued job.
    import py_compile
    import tempfile
    for src, name in ((lm_out, "modeling_internlm2.py"), (chat_out, "modeling_internvl_chat.py")):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(src)
            tmp = f.name
        try:
            py_compile.compile(tmp, doraise=True)
            print("syntax OK:", name)
        except py_compile.PyCompileError as e:
            raise SystemExit("FATAL: patched {} does not compile:\n{}".format(name, e))
        finally:
            os.unlink(tmp)

    print("\nDone. To install an arm:")
    print("  cp {}.BTP {}".format(lm, lm))
    print("  cp {}.BTP {}".format(chat, chat))
    print("  rm -rf $HF_HOME/modules/transformers_modules   # stale remote-code cache")


if __name__ == "__main__":
    main()
