#!/usr/bin/env python3
"""
E1/E2 - Selector ablation for BTP on Qwen2.5-VL.

Generates variants of the BTP modeling file in which ONLY the token-selection rule
changes. The pruning layers and the token budget stay exactly the same, so any
difference in accuracy is attributable to the selection objective.

Modes
  btp          unmodified (attention seed + diversity fill)   -> control
  div          100% diversity (attention seed set to zero)
  attn         100% attention (diversity fill replaced by attention)
  random       uniform random selection
  oracle_text  force-keep OCR text patches, then fill with diversity (diagnostic)

Usage
  python patch_selector_variants.py \
      --src /path/to/modeling_qwen2_5_vl.py.BTP \
      --out-dir /path/to/variants \
      --modes btp,div,attn,random

Then swap one in exactly like the normal BTP file:
  cp variants/modeling_qwen2_5_vl.py.SEL_attn $MODFILE
  # verify: grep -c "BTP_SELECT_MODE" $MODFILE  -> 1

oracle_text reads the text patch indices for the current image from the environment
variable BTP_TEXT_IDX (comma separated, indices into the image-token block). It is
meant for the small probe set driven by run_oracle_probe.py, not a full benchmark.

NOTE: untested against a GPU run. Smoke-test on one image before a full sweep.
"""

import argparse
import os
import re
import sys

# -----------------------------------------------------------------------------
# The replacement block. It keeps the original farthest-point routine as
# _div_prune_orig and dispatches on the hardcoded mode.
# -----------------------------------------------------------------------------
NEW_METHODS = '''
    BTP_SELECT_MODE = "{mode}"

    def _div_prune_orig(self, hidden_states, k, select_indices=None):
        """Original BTP diversity selection: greedy farthest-point on cosine distance."""
        hidden_states_normalized = F.normalize(hidden_states, p=2, dim=1)
        distance_metric = torch.matmul(hidden_states_normalized, hidden_states_normalized.transpose(0, 1))
        distance_metric = -distance_metric + 1
        distance_metric.fill_diagonal_(float('inf'))
        if select_indices is None:
            select_indices = []
        else:
            distance_metric[:, select_indices] = float('-inf')
        num = 0
        while num < k:
            if len(select_indices) == 0:
                row_min_values, row_min_indices = distance_metric.min(dim=1)
                max_value, max_index = row_min_values.max(dim=0)
                select_indices.append(max_index)
                distance_metric[:, max_index] = float('-inf')
            else:
                selected_dis = distance_metric[select_indices, :]
                min_values, _ = torch.min(selected_dis, dim=0)
                _, max_index = min_values.max(dim=0)
                select_indices.append(max_index)
                distance_metric[:, max_index] = float('-inf')
            num += 1
        return torch.tensor(select_indices)

    def _as_int_list(self, select_indices):
        if select_indices is None:
            return []
        out = []
        for v in select_indices:
            out.append(int(v))
        return out

    def _mean_attn_vector(self, n_tokens, device):
        """Mean attention over heads for the current image block, or None."""
        attn = getattr(self, "_cur_image_attn", None)
        if attn is None:
            return None
        vec = torch.mean(attn, dim=1).squeeze()
        if vec.dim() == 0 or vec.shape[-1] != n_tokens:
            return None
        return vec.to(device)

    def div_prune(self, hidden_states, k, select_indices=None):
        """Budget-preserving selector. Adds k new indices to select_indices."""
        mode = self.BTP_SELECT_MODE
        n = hidden_states.shape[0]
        chosen = self._as_int_list(select_indices)

        if mode in ("btp", "div"):
            return self._div_prune_orig(hidden_states, k, select_indices)

        if mode == "random":
            pool = [i for i in range(n) if i not in set(chosen)]
            g = torch.Generator(device="cpu")
            g.manual_seed(1234 + n)
            perm = torch.randperm(len(pool), generator=g).tolist()
            chosen = chosen + [pool[i] for i in perm[:k]]
            return torch.tensor(chosen)

        if mode == "attn":
            vec = self._mean_attn_vector(n, hidden_states.device)
            if vec is None:
                # attention unavailable at this stage, fall back so the run does not crash
                return self._div_prune_orig(hidden_states, k, select_indices)
            masked = vec.clone().float()
            if len(chosen) > 0:
                masked[torch.tensor(chosen, device=masked.device)] = float('-inf')
            take = min(k, int((masked > float('-inf')).sum().item()))
            _, idx = torch.topk(masked, take, dim=-1)
            chosen = chosen + idx.tolist()
            return torch.tensor(chosen)

        if mode == "oracle_text":
            raw = os.environ.get("BTP_TEXT_IDX", "")
            text_idx = []
            if raw.strip():
                for tok in raw.split(","):
                    tok = tok.strip()
                    if tok.isdigit():
                        v = int(tok)
                        if 0 <= v < n:
                            text_idx.append(v)
            already = set(chosen)
            add = [i for i in text_idx if i not in already][:k]
            chosen = chosen + add
            remaining = k - len(add)
            if remaining > 0:
                return self._div_prune_orig(hidden_states, remaining, chosen)
            return torch.tensor(chosen)

        raise ValueError("unknown BTP_SELECT_MODE: " + str(mode))

    def attn_prune(self, image_attn, select_num):
        mean_attn = torch.mean(image_attn, dim=1)
        if self.BTP_SELECT_MODE == "random":
            n = mean_attn.shape[-1]
            g = torch.Generator(device="cpu")
            g.manual_seed(4321 + n)
            perm = torch.randperm(n, generator=g)[:select_num]
            return perm.to(mean_attn.device)
        top_attn_values, attn_indices = torch.topk(mean_attn, select_num, dim=-1)
        return attn_indices.squeeze()
'''

DIV_PRUNE_START = "    def div_prune(self,hidden_states,k,select_indices=None):"
ATTN_PRUNE_START = "    def attn_prune(self,image_attn,select_num):"
GET_INPUT_EMB = "    def get_input_embeddings(self):"

ATTN_ASSIGN = re.compile(r"^(\s*)image_attention = last_token_attn\[.*\]\s*$", re.M)


def build_variant(src_text, mode):
    text = src_text

    # 1. replace div_prune + attn_prune with the mode-aware versions
    i = text.find(DIV_PRUNE_START)
    j = text.find(GET_INPUT_EMB)
    if i == -1 or j == -1 or j <= i:
        raise RuntimeError("could not locate div_prune/attn_prune block in source file")
    text = text[:i] + NEW_METHODS.format(mode=mode) + "\n" + text[j:]

    # 2. stash the current image attention so the attn mode can reach it
    def stash(m):
        indent = m.group(1)
        return m.group(0) + "\n" + indent + "self._cur_image_attn = image_attention"

    text, n_stash = ATTN_ASSIGN.subn(stash, text)
    if n_stash == 0:
        raise RuntimeError("could not find any 'image_attention = last_token_attn[...]' line")

    # 3. for pure-diversity mode, remove the attention seed so 100% of the budget
    #    is spent by the diversity selector
    if mode == "div":
        text = text.replace("atten_select_num = int(k * 0.1)", "atten_select_num = int(k * 0.0)")
        text = text.replace("atten_select_num = int(k * 0.4)", "atten_select_num = int(k * 0.0)")

    # 4. make sure os is importable inside the model file (oracle mode uses it)
    if "\nimport os" not in text.split("class ")[0]:
        text = text.replace("import torch\n", "import os\nimport torch\n", 1)

    return text, n_stash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to modeling_qwen2_5_vl.py.BTP")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--modes", default="btp,div,attn,random,oracle_text")
    args = ap.parse_args()

    src_text = open(args.src, "r", encoding="utf-8").read()
    os.makedirs(args.out_dir, exist_ok=True)

    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        out_text, n_stash = build_variant(src_text, mode)
        out_path = os.path.join(args.out_dir, "modeling_qwen2_5_vl.py.SEL_" + mode)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out_text)
        # cheap sanity checks
        assert "BTP_SELECT_MODE" in out_text
        assert out_text.count("def div_prune") == 1
        assert out_text.count("def attn_prune") == 1
        print("wrote {}  (mode={}, attention stashes={})".format(out_path, mode, n_stash))

    print("\nNext: swap one variant over the active modeling file and run TextVQA.")
    print("Keep the retention setting identical across modes, the whole point is a matched budget.")


if __name__ == "__main__":
    sys.exit(main())
