#!/usr/bin/env python3
"""
measure_response_modes.py

Phase 5. Why can the model stop using visual tokens earlier for some answer types than for
others? Four experiments on Qwen2.5-VL-7B + TextVQA, all built from ONE sample set so every
comparison is paired: same image, same question, same correct answer. Only the response
condition changes.

CONDITIONS (--cond)
-------------------
  open      free open-ended answer (the Phase 4 TextVQA setting)
  fill      constrained generation: the model is told the answer's word count and first
            character, and fills the blank
  mc2/mc4/mc8  multiple choice with 2, 4 or 8 options. Nested: the mc2 options are a subset
            of the mc4 options, which are a subset of the mc8 options, so only the size of the
            answer space changes. Distractors are other questions' answers of the same kind
            (numbers for numbers, text for text), as in Phase 4.
  mc4_type  multiple choice with 4 options whose distractors are SAME-TYPE answers to other
            questions of the same kind (a brand for a brand question, a year for a year
            question), matched in word count. Built to stop the options giving the answer away.
  yesno     "Is the answer X?" where X is the correct answer for half the samples and the
            first same-type distractor for the other half (seeded, balanced).

  Experiment 1 = open, mc8, mc4, mc2.     Experiment 2 = mc4_type (+ no-image control).
  Experiment 3 = yesno, mc4, fill, open.  Experiment 4 = confidence, recorded in every run.

WHAT ONE RUN RECORDS, at every withdrawal layer l = 0..L (l = layers that kept the image)
------------------------------------------------------------------------------------------
  correct     per-sample correctness of the greedy answer (same scoring as Phase 4)
  pred        the greedy answer text
  conf_p      closed conditions: probability of the correct option label, renormalised over
              the option labels at the first answer position.
              generative conditions: per-token geometric-mean probability of the gold answer
              under teacher forcing.
  conf_margin closed conditions only: p(correct) minus the best wrong option
  conf_top    closed conditions only: the option label with the highest probability

ALSO, ONCE PER RUN
------------------
  noimage     the same prompts with NO image in the message at all (true text-only control,
              not just deletion before layer 0)
  lens        logit lens on the unmodified run: for every layer, the final norm and lm_head
              applied to that layer's last-position hidden state; the probability of the
              correct label (closed) or of the gold answer's first token (generative), and
              whether it is the top choice. Shows when the decision forms INSIDE the model.

USAGE (workspace root on the HPC)
---------------------------------
  python -u phase5-response-format/scripts/measure_response_modes.py --cond mc4 \
      --n 200 --out results/phase5/qwen7b_textvqa_mc4.json
Reruns with the same --out resume from the last completed layer.
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (os.path.join(ROOT, "phase3-mechanism", "scripts"),
          os.path.join(ROOT, "phase4-access-horizon", "scripts")):
    sys.path.insert(0, p)

from measure_access_horizon import (LETTERS, MC_SUFFIX, SHORT_SUFFIX, _is_num, _majority,  # noqa: E402
                                    _mc_question, _too_close, _yes_no, build_textvqa,
                                    parse_letter, save)

CONDS = ("open", "fill", "mc2", "mc4", "mc8", "mc4_type", "yesno")
CLOSED = ("mc2", "mc4", "mc8", "mc4_type", "yesno")
MAX_NEW = {"open": 10, "fill": 10, "yesno": 4}


# ===========================================================================
#  Sample construction (deterministic; every condition sees the same ids)
# ===========================================================================

def _norm(s):
    from measure_visual_depth import normalise
    return normalise(s)


def qcat(q):
    """Coarse answer type from the question wording. Used only to pick same-type distractors."""
    q = " " + str(q).lower() + " "
    rules = [("year", ("what year", "which year", " year ")),
             ("time", ("what time",)),
             ("count", ("how many",)),
             ("number", ("what number", "number on", "number of the", "how much", "price",
                         "what is the number", "jersey", "phone", "speed", "percent")),
             ("brand", ("brand", "company", "manufacturer", "make of", "what kind of beer",
                        "what airline", "logo")),
             ("name", ("who ", "name", "author", "team", "city", "country", "store", "website")),
             ("color", ("color", "colour")),
             ("letter", ("what letter",)),
             ("word", ("what word", "what does", "what is written", "say", "says", "written",
                       "title", "sign"))]
    for cat, keys in rules:
        if any(k in q for k in keys):
            return cat
    return "other"


def _rng(*parts):
    return random.Random("|".join(str(p) for p in parts))


def type_pool(max_rows=4000):
    """Question/answer pairs from TextVQA validation, without decoding any image."""
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/textvqa", split="validation", streaming=True)
    try:
        ds = ds.select_columns(["question", "answers"])
    except Exception:
        pass
    out = []
    for i, ex in enumerate(ds):
        if i >= max_rows:
            break
        refs = ex.get("answers") or []
        if not refs:
            continue
        c = _majority([str(r) for r in refs])
        if _norm(c) in ("yes", "no") or not _norm(c):
            continue
        out.append({"q": str(ex.get("question") or ""), "a": c, "cat": qcat(ex.get("question"))})
    return out


def build_all(n, seed):
    """Returns {cond: [sample]} with identical ids across conditions."""
    base = []
    for s in build_textvqa(n, seed):
        c = _majority(s["refs"])
        if _norm(c) in ("yes", "no") or not _norm(c):
            continue                      # no meaningful 8-way or same-type version
        t = dict(s)
        t["gold"], t["cat"] = c, qcat(s["question"])
        base.append(t)

    # Nested cross-image distractors, same kind (number / text), as in Phase 4.
    golds = [s["gold"] for s in base]
    pool_num = list(dict.fromkeys(g for g in golds if _is_num(g)))
    pool_txt = list(dict.fromkeys(g for g in golds if not _is_num(g)))
    for s in base:
        pool = pool_num if _is_num(s["gold"]) else pool_txt
        cand = [p for p in pool if not any(_too_close(r, p) for r in s["refs"] + [s["gold"]])]
        _rng(seed, "nest", s["id"]).shuffle(cand)
        s["dis_nested"] = cand[:7]

    # Same-type distractors from other questions of the same answer type.
    tp = type_pool()
    for s in base:
        k = len(s["gold"].split())
        num = _is_num(s["gold"])
        ok = lambda e: (not any(_too_close(r, e["a"]) for r in s["refs"] + [s["gold"]])
                        and _is_num(e["a"]) == num and _norm(e["q"]) != _norm(s["question"]))
        tiers = [[e["a"] for e in tp if e["cat"] == s["cat"] and len(e["a"].split()) == k and ok(e)],
                 [e["a"] for e in tp if e["cat"] == s["cat"] and ok(e)],
                 [e["a"] for e in tp if len(e["a"].split()) == k and ok(e)]]
        picked, src = [], []
        for ti, tier in enumerate(tiers):
            tier = [a for a in dict.fromkeys(tier) if _norm(a) not in {_norm(x) for x in picked}]
            _rng(seed, "type", ti, s["id"]).shuffle(tier)
            for a in tier[:3 - len(picked)]:
                picked.append(a)
                src.append(ti)
            if len(picked) == 3:
                break
        s["dis_type"], s["dis_type_tier"] = picked, src

    out = {c: [] for c in CONDS}
    for s in base:
        common = {"id": s["id"], "image": s["image"], "refs": s["refs"], "gold": s["gold"],
                  "cat": s["cat"], "base_question": s["question"]}
        q = s["question"]
        out["open"].append(dict(common, question=q, suffix=SHORT_SUFFIX))
        k = len(s["gold"].split())
        first = s["gold"].strip()[0]
        out["fill"].append(dict(common, question=q, suffix=(
            "\nFill in the blank with the text from the image. The answer has {} word{} and "
            "begins with \"{}\".\nAnswer: ____\nReply with only the missing word or phrase."
            .format(k, "" if k == 1 else "s", first))))
        nested = [s["gold"]] + s["dis_nested"]
        for m in (2, 4, 8):
            opts = nested[:m]
            _rng(seed, "order", m, s["id"]).shuffle(opts)
            ans = LETTERS[[_norm(o) for o in opts].index(_norm(s["gold"]))]
            out["mc%d" % m].append(dict(common, question=_mc_question(q, opts), options=opts,
                                        answer=ans, suffix=MC_SUFFIX))
        opts = [s["gold"]] + s["dis_type"]
        _rng(seed, "typeorder", s["id"]).shuffle(opts)
        ans = LETTERS[[_norm(o) for o in opts].index(_norm(s["gold"]))]
        out["mc4_type"].append(dict(common, question=_mc_question(q, opts), options=opts,
                                    answer=ans, suffix=MC_SUFFIX,
                                    type_tiers=s["dis_type_tier"]))
        true_half = int(hashlib.md5("{}|yn|{}".format(seed, s["id"]).encode()).hexdigest(), 16) % 2 == 0
        cand = s["gold"] if true_half or not s["dis_type"] else s["dis_type"][0]
        label = "yes" if (true_half or not s["dis_type"]) else "no"
        out["yesno"].append(dict(common, question='{}\nIs the answer "{}"?'.format(q, cand),
                                 suffix="\nAnswer yes or no.", options=["Yes", "No"],
                                 answer=label, candidate=cand))
    nm = [len(s["dis_nested"]) for s in base]
    nt = [s["dis_type_tier"] for s in base]
    print("samples: {} (yes/no-answer items excluded)".format(len(base)))
    print("nested distractors: min {} of 7".format(min(nm)))
    print("same-type distractors: {} of {} samples have 3 from the same category".format(
        sum(1 for t in nt if t == [0, 0, 0]), len(nt)))
    return out


# ===========================================================================
#  Scoring
# ===========================================================================

def score(cond, pred, s):
    from measure_visual_depth import is_correct
    if pred is None:
        return 0
    if cond in ("open", "fill"):
        return 1 if is_correct(pred, s["refs"]) else 0
    if cond == "yesno":
        return 1 if _yes_no(pred) == s["answer"] else 0
    return 1 if parse_letter(pred, s["options"]) == s["answer"] else 0


# ===========================================================================
#  Confidence
# ===========================================================================

class Scorer:
    def __init__(self, model, proc):
        import torch
        self.torch, self.model, self.proc, self.tok = torch, model, proc, proc.tokenizer
        self.letter_ids = {L: self._ids(L) for L in LETTERS}
        self.yn_ids = {"yes": self._ids("Yes", "yes"), "no": self._ids("No", "no")}

    def _ids(self, *words):
        ids = []
        for w in words:
            t = self.tok.encode(w, add_special_tokens=False)
            if t:
                ids.append(t[0])
        return sorted(set(ids))

    def labels(self, cond, s):
        if cond == "yesno":
            return ["yes", "no"], self.yn_ids, s["answer"]
        labs = list(LETTERS[:len(s["options"])])
        return labs, {L: self.letter_ids[L] for L in labs}, s["answer"]

    def closed_from_logits(self, cond, s, logits):
        torch = self.torch
        labs, ids, ans = self.labels(cond, s)
        lp = torch.log_softmax(logits.float(), dim=-1)
        score_l = torch.stack([torch.logsumexp(lp[ids[L]], 0) for L in labs])
        p = torch.softmax(score_l, 0).tolist()
        i = labs.index(ans)
        other = max(p[j] for j in range(len(p)) if j != i)
        return p[i], p[i] - other, labs[max(range(len(p)), key=lambda j: p[j])]

    def closed(self, cond, s, inp):
        with self.torch.inference_mode():
            out = self.model(**inp, use_cache=False)
        return self.closed_from_logits(cond, s, out.logits[0, -1])

    def gold_prob(self, s, inp):
        """Geometric-mean token probability of the gold answer under teacher forcing."""
        torch = self.torch
        g = self.tok.encode(s["gold"], add_special_tokens=False)
        if not g:
            return None
        ids = inp["input_ids"]
        gt = torch.tensor([g], device=ids.device, dtype=ids.dtype)
        full = dict(inp)
        full["input_ids"] = torch.cat([ids, gt], 1)
        full["attention_mask"] = torch.cat(
            [inp["attention_mask"], torch.ones_like(gt)], 1)
        with torch.inference_mode():
            out = self.model(**full, use_cache=False)
        # Index from the end: when image tokens are deleted the output is shorter than the
        # input, so absolute positions past the image no longer line up.
        lp = torch.log_softmax(out.logits[0, -len(g) - 1:-1].float(), dim=-1)
        tok_lp = lp[torch.arange(len(g), device=lp.device), torch.tensor(g, device=lp.device)]
        return math.exp(tok_lp.mean().item())

    def lens(self, cond, s, inp):
        """Logit lens on the unmodified model at the answer position, every layer."""
        torch = self.torch
        with torch.inference_mode():
            out = self.model(**inp, use_cache=False, output_hidden_states=True)
        hs = out.hidden_states[1:]                     # one per decoder layer
        norm = self.model.model.norm
        head = self.model.lm_head
        rows = []
        for h in hs:
            logits = head(norm(h[:, -1, :]))[0]
            if cond in CLOSED:
                p, margin, top = self.closed_from_logits(cond, s, logits)
                rows.append([p, 1 if top == s["answer"] else 0])
            else:
                g = self.tok.encode(s["gold"], add_special_tokens=False)
                if not g:
                    rows.append([None, 0])
                    continue
                pr = torch.softmax(logits.float(), -1)
                rows.append([pr[g[0]].item(), 1 if int(pr.argmax()) == g[0] else 0])
        return rows


# ===========================================================================
#  Main loop
# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--cond", choices=CONDS, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-only", action="store_true",
                    help="build the samples, print a few, and exit (no GPU needed)")
    args = ap.parse_args()

    allc = build_all(args.n, args.seed)
    samples = allc[args.cond]
    if args.build_only:
        for s in samples[:5]:
            print("---", s["id"], "| gold:", s["gold"], "| cat:", s["cat"])
            print(s["question"] + s["suffix"])
        return

    import torch
    from measure_visual_depth import VisualTokenDeleter, pick_loader
    # InternVL goes through its chat API, so only answers are recorded there: no confidence
    # and no logit lens (Experiment 4 is Qwen only).
    ivl = "internvl" in args.model.lower()
    bind = None
    if ivl:
        from internvl_adapter import load_internvl
        model, layers, run, generate, bind, _ = load_internvl(args.model)
    else:
        model, layers, run, generate = pick_loader(args.model)(args.model)
    L = len(layers)
    _, _, proc = run(samples[0]["image"], samples[0]["question"], suffix=samples[0]["suffix"])
    sc = None if ivl else Scorer(model, proc)
    closed = args.cond in CLOSED
    max_new = MAX_NEW.get(args.cond, 8)
    ids = [s["id"] for s in samples]
    print("model :", args.model, " layers:", L)
    print("cond  :", args.cond, " samples:", len(samples))
    print("prompt example:\n" + samples[0]["question"] + samples[0]["suffix"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    res = {"model": args.model, "cond": args.cond, "n_layers": L, "seed": args.seed,
           "sample_ids": ids, "meta": [{k: s.get(k) for k in
                                        ("gold", "cat", "options", "answer", "candidate",
                                         "type_tiers", "base_question")} for s in samples],
           "per_layer": {}, "predictions": {}, "conf_p": {}, "conf_margin": {}, "conf_top": {},
           "seconds_per_layer": {}}
    if os.path.exists(args.out):
        old = json.load(open(args.out))
        if old.get("sample_ids") != ids:
            raise SystemExit("FATAL: {} holds a different sample set".format(args.out))
        res = old
        print("resuming, layers done:", sorted(int(k) for k in res["per_layer"]))

    # True text-only control: no image in the message at all.
    if "noimage" not in res:
        flags, preds = [], []
        for s in samples:
            if ivl:
                inp = {"pixel_values": None, "question": s["question"] + s["suffix"]}
            else:
                msg = [{"role": "user", "content": [{"type": "text", "text": s["question"] + s["suffix"]}]}]
                text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                inp = proc(text=[text], padding=True, return_tensors="pt").to("cuda")
            pred = generate(inp, max_new_tokens=max_new)
            preds.append(pred)
            flags.append(score(args.cond, pred, s))
        res["noimage"] = {"correct": flags, "predictions": preds}
        save(res, args.out)
        print("  no image at all: acc {:.4f}".format(sum(flags) / len(flags)), flush=True)

    # Logit lens on the unmodified model.
    if "lens" not in res and not ivl:
        rows = []
        for s in samples:
            inp, _, _ = run(s["image"], s["question"], suffix=s["suffix"])
            rows.append(sc.lens(args.cond, s, inp))
        res["lens"] = rows
        save(res, args.out)
        print("  logit lens saved", flush=True)

    deleter = VisualTokenDeleter(model, layers, 0)
    if bind is not None:
        bind(deleter)       # InternVL: positions are set from inside model.generate
    for l in [L] + list(range(L)):
        if str(l) in res["per_layer"]:
            continue
        t0 = time.time()
        armed = l < L
        if armed:
            deleter.layer_index = l
            deleter.fired = 0
            deleter.arm()
        flags, preds, cp, cm, ct, failed = [], [], [], [], [], 0
        try:
            for s in samples:
                inp, positions, _ = run(s["image"], s["question"], suffix=s["suffix"])
                pred, p, m, top = None, None, None, None
                try:
                    deleter.begin_sample(positions)
                    pred = generate(inp, max_new_tokens=max_new)
                except Exception as e:
                    print("  failed on", s["id"], repr(e))
                    failed += 1
                # A confidence failure must not discard the generated answer.
                if sc is not None:
                    try:
                        deleter.begin_sample(positions)
                        if closed:
                            p, m, top = sc.closed(args.cond, s, inp)
                        else:
                            p, m, top = sc.gold_prob(s, inp), None, None
                    except Exception as e:
                        print("  confidence failed on", s["id"], repr(e))
                preds.append(pred)
                flags.append(score(args.cond, pred, s))
                cp.append(p)
                cm.append(m)
                ct.append(top)
        finally:
            if armed:
                deleter.disarm()
        if armed and deleter.fired < len(samples) - failed:
            raise SystemExit("FATAL: hook fired {} times for {} samples at layer {}".format(
                deleter.fired, len(samples), l))
        res["per_layer"][str(l)] = flags
        res["predictions"][str(l)] = preds
        res["conf_p"][str(l)] = cp
        res["conf_margin"][str(l)] = cm
        res["conf_top"][str(l)] = ct
        res["seconds_per_layer"][str(l)] = time.time() - t0
        save(res, args.out)
        acc = sum(flags) / len(flags)
        mp = [x for x in cp if x is not None]
        print("  l {:3d}  acc {:.4f}  mean conf {:.3f}  {:5.0f} s{}".format(
            l, acc, sum(mp) / max(1, len(mp)), time.time() - t0, "" if armed else "   [baseline]"),
            flush=True)
        torch.cuda.empty_cache()
    print("saved to", args.out)


if __name__ == "__main__":
    main()
