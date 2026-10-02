#!/usr/bin/env python3
"""
measure_access_horizon.py

Phase 4 pilot. Does each multimodal task have its own visual access horizon?

THE QUESTION
------------
Phase 3 showed that deleting every visual token at some decoder layer destroys TextVQA
below a sharp depth and costs almost nothing above it. That was one task. Here we measure
the same curve for several tasks and ask whether the depth at which the model stops needing
the image depends on the task, and whether that dependence is reproducible across models.

WHAT ONE RUN DOES
-----------------
For one task and one model:
  1. Fix a sample set (deterministic, saved, verified on resume).
  2. For every l in 0..L, delete all visual tokens BEFORE decoder layer l runs, so exactly
     l layers had access to the image. l = 0 is the text-only floor, l = L is the baseline.
  3. Record per-sample correctness at every l, not just the mean, so that confidence
     intervals and per-sample horizons can be computed afterwards without rerunning.

DEFINITIONS (computed by --analyse, also at the end of every run)
-----------------------------------------------------------------
  acc(l)      accuracy with l layers of visual access
  floor       acc(0): what the model scores from the question alone
  baseline    acc(L): no deletion

  normalised recovery   g(l) = (acc(l) - floor) / (baseline - floor)

  Raw accuracy is misleading for tasks with a high floor. POPE is yes/no, so it sits near
  50 per cent with no image at all, and a curve of raw accuracy would call it an early
  access task merely because it barely uses the image. g(l) measures how much of the
  image's contribution has been recovered, which is the quantity we care about.

  horizon h_alpha = the smallest l such that g(l') >= alpha for EVERY l' >= l.

  The "every l' >= l" condition makes the horizon robust to a single noisy point crossing
  alpha early and falling back. Reported for alpha in {0.5, 0.9, 0.95}, with 95 per cent
  bootstrap confidence intervals over samples.

  per-sample horizon  for a sample the model answers correctly at baseline, the smallest l
  such that it stays correct for every l' >= l. Its distribution is the starting point for
  input-specific schedules, which is the algorithmic question we do NOT address yet.

CONVENTION
----------
Zero based, deletion before the named layer, identical to measure_visual_depth.py. The
number reported is layers WITH access. BTP's one-based end_layer = 23 corresponds to l = 22.

USAGE
-----
  python -u measure_access_horizon.py --task textvqa --n 200 --out results/horizon/qwen7b_textvqa.json
  python -u measure_access_horizon.py --task pope    --n 300 --out results/horizon/qwen7b_pope.json
  python    measure_access_horizon.py --analyse results/horizon/qwen7b_textvqa.json

Reruns with the same --out resume from the last completed layer.
"""

import argparse
import io
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# Reuse the Phase 3 instrument rather than copying it. The deletion hook there was validated
# against source patching (job 423959), so it must not fork.
for cand in (os.path.join(HERE, "..", "phase3-mechanism"), os.path.join(HERE, "phase3-mechanism")):
    if os.path.isfile(os.path.join(cand, "measure_visual_depth.py")):
        sys.path.insert(0, os.path.abspath(cand))
        break

ALPHAS = (0.5, 0.9, 0.95)


# ===========================================================================
#  Tasks. Each supplies a sample builder and a scorer.
# ===========================================================================

def _yes_no(pred):
    import re
    p = re.sub(r"[^a-z ]", " ", str(pred).lower()).split()
    if not p:
        return None
    if p[0] in ("yes", "no"):
        return p[0]
    if "yes" in p and "no" not in p:
        return "yes"
    if "no" in p and "yes" not in p:
        return "no"
    return None


def build_textvqa(n, seed):
    """Deduplicated by image, first n in dataset order. Same builder as the Phase 3 tool,
    so the textvqa pilot is directly comparable with job 423959."""
    from measure_visual_depth import build_samples
    raw = build_samples("lmms-lab/textvqa", "validation", n)
    return [{"id": "textvqa_%d" % i, "image": img, "question": q, "refs": refs}
            for i, (img, q, refs) in enumerate(raw)]


def score_textvqa(pred, s):
    from measure_visual_depth import is_correct
    return is_correct(pred, s["refs"])


def build_pope(n, seed, max_scan=9000):
    """Balanced yes/no. POPE lists questions grouped by category and object, so the stream
    is shuffled with a fixed seed and buffer before sampling. Not deduplicated by image:
    several objects are asked about per COCO image, and that is the benchmark's design."""
    from datasets import load_dataset
    from measure_visual_depth import load_image
    ds = load_dataset("lmms-lab/POPE", split="test", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=3000)
    want = {"yes": n // 2, "no": n - n // 2}
    out, scanned = [], 0
    for ex in ds:
        if not any(want.values()) or scanned >= max_scan:
            break
        scanned += 1
        ans = str(ex.get("answer", "")).strip().lower()
        if want.get(ans, 0) <= 0:
            continue
        img = load_image(ex.get("image"))
        if img is None:
            continue
        want[ans] -= 1
        sid = "pope_%s_%s" % (ex.get("category", "x"), ex.get("question_id", ex.get("id", scanned)))
        out.append({"id": sid, "image": img, "question": str(ex["question"]), "answer": ans})
    return out


def score_pope(pred, s):
    return _yes_no(pred) == s["answer"]


# ---------------------------------------------------------------------------
#  Shared helpers for the Phase 4 task set
# ---------------------------------------------------------------------------

SHORT_SUFFIX = "\nAnswer the question using a single word or phrase."
MC_SUFFIX = "\nAnswer with the option's letter from the given choices directly."
LETTERS = "ABCDEFGH"


def _stream(name, config=None, split="test", seed=0, buffer=2000):
    """Streamed and shuffled with a fixed seed. These benchmarks are stored grouped by
    category, so the first n rows in file order would be one category only."""
    from datasets import load_dataset
    ds = load_dataset(name, config, split=split, streaming=True) if config else \
        load_dataset(name, split=split, streaming=True)
    return ds.shuffle(seed=seed, buffer_size=buffer)


def _take(ds, n, make, max_scan=20000):
    """Apply make(ex, k) to rows until n valid samples; make returns None to skip."""
    out, k = [], 0
    for ex in ds:
        if len(out) >= n or k >= max_scan:
            break
        k += 1
        s = make(ex, k)
        if s is not None:
            out.append(s)
    return out


def _mc_question(question, options, hint=None):
    q = ""
    if hint and str(hint).strip() and str(hint).strip().lower() != "nan":
        q += "Hint: " + str(hint).strip() + "\n"
    q += str(question).strip()
    for i, o in enumerate(options):
        q += "\n{}. {}".format(LETTERS[i], str(o).strip())
    return q


def parse_letter(pred, options):
    """First standalone option letter, else the option whose text the answer contains."""
    import re
    p = str(pred).strip()
    m = re.match(r"^\(?([A-H])[\)\.\:\s]", p + " ")
    if m and LETTERS.index(m.group(1)) < len(options):
        return m.group(1)
    for m in re.finditer(r"\b([A-H])\b", p):
        if LETTERS.index(m.group(1)) < len(options):
            return m.group(1)
    low = p.lower()
    hits = [LETTERS[i] for i, o in enumerate(options) if str(o).strip().lower() and str(o).strip().lower() in low]
    return hits[0] if len(hits) == 1 else None


def score_mc(pred, s):
    return parse_letter(pred, s["options"]) == s["answer"]


# ---------------------------------------------------------------------------
#  GQA: compositional scene questions. Questions and images live in separate configs.
# ---------------------------------------------------------------------------

def build_gqa(n, seed):
    from datasets import load_dataset
    from measure_visual_depth import load_image
    imgs = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev")
    lookup = {r["id"]: i for i, r in enumerate(imgs)}
    qs = _stream("lmms-lab/GQA", "testdev_balanced_instructions", "testdev", seed, 5000)

    def make(ex, k):
        j = lookup.get(ex["imageId"])
        if j is None:
            return None
        img = load_image(imgs[j]["image"])
        if img is None:
            return None
        return {"id": "gqa_" + str(ex["id"]), "image": img, "question": ex["question"],
                "refs": [ex["answer"]], "suffix": SHORT_SUFFIX}
    return _take(qs, n, make)


def score_exact(pred, s):
    from measure_visual_depth import normalise
    return normalise(pred) in {normalise(r) for r in s["refs"]}


# ---------------------------------------------------------------------------
#  Multiple choice: MMBench, ScienceQA-IMG, AI2D
# ---------------------------------------------------------------------------

def build_mmbench(n, seed):
    from measure_visual_depth import load_image
    ds = _stream("lmms-lab/MMBench", "en", "dev", seed)

    def make(ex, k):
        opts = [ex[c] for c in "ABCD" if ex.get(c) is not None and str(ex[c]).strip()
                and str(ex[c]).strip().lower() != "nan"]
        img = load_image(ex.get("image"))
        if img is None or len(opts) < 2:
            return None
        return {"id": "mmbench_" + str(ex["index"]), "image": img, "options": opts,
                "question": _mc_question(ex["question"], opts, ex.get("hint")),
                "answer": str(ex["answer"]).strip(), "suffix": MC_SUFFIX}
    return _take(ds, n, make)


def build_scienceqa(n, seed):
    from measure_visual_depth import load_image
    ds = _stream("lmms-lab/ScienceQA", "ScienceQA-IMG", "test", seed)

    def make(ex, k):
        img = load_image(ex.get("image"))
        if img is None:
            return None
        opts = list(ex["choices"])
        return {"id": "sqa_%d" % k, "image": img, "options": opts,
                "question": _mc_question(ex["question"], opts, ex.get("hint")),
                "answer": LETTERS[int(ex["answer"])], "suffix": MC_SUFFIX}
    return _take(ds, n, make)


def build_ai2d(n, seed):
    from measure_visual_depth import load_image
    ds = _stream("lmms-lab/ai2d", None, "test", seed)

    def make(ex, k):
        img = load_image(ex.get("image"))
        if img is None:
            return None
        opts = list(ex["options"])
        return {"id": "ai2d_%d" % k, "image": img, "options": opts,
                "question": _mc_question(ex["question"], opts),
                "answer": LETTERS[int(ex["answer"])], "suffix": MC_SUFFIX}
    return _take(ds, n, make)


# ---------------------------------------------------------------------------
#  Reading: ChartQA (relaxed accuracy), DocVQA (ANLS)
#  Document images are capped at 2048 visual tokens. Uncapped DocVQA pages can exceed
#  10k tokens, which is slow on an A30 and risks OOM. The cap is identical at every layer,
#  so the curve shape is unaffected; the baseline may sit slightly below lmms-eval's.
# ---------------------------------------------------------------------------

DOC_MAX_PIXELS = 2048 * 28 * 28


def build_chartqa(n, seed):
    from measure_visual_depth import load_image
    ds = _stream("lmms-lab/ChartQA", None, "test", seed)

    def make(ex, k):
        img = load_image(ex.get("image"))
        if img is None:
            return None
        return {"id": "chartqa_%d" % k, "image": img, "question": ex["question"],
                "refs": [str(ex["answer"])], "suffix": "\nAnswer the question with a single word.",
                "max_pixels": DOC_MAX_PIXELS}
    return _take(ds, n, make)


def score_relaxed(pred, s):
    """ChartQA relaxed accuracy: numbers within 5 per cent, otherwise exact match."""
    from measure_visual_depth import as_float, normalise
    p = str(pred).strip().rstrip(".")
    for r in s["refs"]:
        pf, rf = as_float(p), as_float(r)
        if pf is not None and rf is not None:
            if rf == 0 and pf == 0:
                return True
            if rf != 0 and abs(pf - rf) / abs(rf) <= 0.05:
                return True
        elif normalise(p) == normalise(r):
            return True
    return False


def build_docvqa(n, seed):
    from measure_visual_depth import load_image
    ds = _stream("lmms-lab/DocVQA", "DocVQA", "validation", seed, 1000)
    seen = set()

    def make(ex, k):
        # One question per document page, so page-level difficulty is not over-sampled.
        doc = ex.get("docId")
        if doc in seen:
            return None
        img = load_image(ex.get("image"))
        if img is None:
            return None
        seen.add(doc)
        return {"id": "docvqa_" + str(ex["questionId"]), "image": img, "question": ex["question"],
                "refs": [str(a) for a in ex["answers"]], "suffix": SHORT_SUFFIX,
                "max_pixels": DOC_MAX_PIXELS}
    return _take(ds, n, make)


def _levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score_anls(pred, s):
    """ANLS, the DocVQA metric: 1 - normalised edit distance, zeroed below 0.5. Continuous,
    so the per-layer accuracy is mean ANLS, as in lmms-eval."""
    p = " ".join(str(pred).strip().lower().split())
    best = 0.0
    for r in s["refs"]:
        r = " ".join(r.strip().lower().split())
        if not p and not r:
            return 1.0
        d = _levenshtein(p, r) / max(len(p), len(r), 1)
        best = max(best, 1 - d if d < 0.5 else 0.0)
    return best


# ---------------------------------------------------------------------------
#  FORMAT CONTROL: reading tasks asked as multiple choice
#
#  Every early task in the study is closed-form (multiple choice, yes/no) and every late
#  task is open-ended, so the horizon could be tracking answer FORMAT rather than the need
#  to read. These variants keep the SAME questions and images as the open-ended runs
#  (same builders, same sample ids, same resolution cap) and change only the format: the
#  correct answer plus up to three distractors, lettered, in a seeded random order.
#
#  Distractors are other samples' answers of the same kind (numbers for numbers, text for
#  text), so the options cannot be ruled out from the question alone as easily. Any
#  residual guessability raises the floor, which the floor-normalised horizon absorbs.
#  A yes/no answer gets the two options Yes / No.
#
#  If reading drives the horizon, the MC variant stays late. If format drives it, the MC
#  variant moves early, towards MMBench. Compare pairwise with format_compare.py.
# ---------------------------------------------------------------------------

def _is_num(x):
    from measure_visual_depth import as_float
    return as_float(x) is not None


def _majority(refs):
    from collections import Counter
    from measure_visual_depth import normalise
    c = Counter(normalise(r) for r in refs if str(r).strip())
    top = c.most_common(1)[0][0]
    for r in refs:
        if normalise(r) == top:
            return str(r).strip()
    return str(refs[0]).strip()


def _too_close(a, b):
    """True if b would be scored correct against a, so it cannot be a distractor."""
    from measure_visual_depth import as_float, normalise
    fa, fb = as_float(a), as_float(b)
    if fa is not None and fb is not None:
        return fa == fb or (fa != 0 and abs(fb - fa) / abs(fa) <= 0.05)
    return normalise(a) == normalise(b)


def _to_mc(samples, prefix, seed):
    from measure_visual_depth import normalise
    rng = random.Random(seed)
    correct = [_majority(s["refs"]) for s in samples]
    pool_num = [c for c in correct if _is_num(c)]
    pool_txt = [c for c in correct if not _is_num(c) and normalise(c) not in ("yes", "no")]
    out = []
    for s, c in zip(samples, correct):
        r = random.Random("{}|{}".format(seed, s["id"]))
        if normalise(c) in ("yes", "no"):
            opts = ["Yes", "No"]
        else:
            pool = pool_num if _is_num(c) else pool_txt
            cand = [p for p in dict.fromkeys(pool)
                    if not any(_too_close(ref, p) for ref in s["refs"] + [c])]
            r.shuffle(cand)
            opts = [c] + cand[:3]
            r.shuffle(opts)
        ans = LETTERS[[normalise(o) for o in opts].index(normalise(c))]
        t = dict(s)
        t.update({"question": _mc_question(s["question"], opts), "options": opts,
                  "answer": ans, "suffix": MC_SUFFIX, "mc_correct_text": c})
        out.append(t)
    return out


def build_textvqa_mc(n, seed):
    return _to_mc(build_textvqa(n, seed), "textvqa", seed)


def build_chartqa_mc(n, seed):
    return _to_mc(build_chartqa(n, seed), "chartqa", seed)


# ---------------------------------------------------------------------------
#  HARD-DISTRACTOR CONTROL for TextVQA multiple choice
#
#  In textvqa_mc the wrong options come from OTHER images, so the model can reject them by
#  noticing that the text is simply not there. That is recognition, a weaker demand than
#  reading. Here the wrong options are OTHER TEXT FROM THE SAME IMAGE (TextVQA's OCR
#  tokens), joined into n-grams with the same word count as the answer, so length does not
#  give the answer away. To choose correctly the model must know which of several strings
#  that are all visibly present answers the question.
#
#  If this variant also moves early, answer format is the driver. If it moves back
#  towards the open-ended horizon, the early MC horizon was produced by easy distractors.
#  Same questions, images and sample ids as textvqa and textvqa_mc. Samples whose image
#  has too little other text are topped up from the cross-image pool and counted.
# ---------------------------------------------------------------------------

def build_textvqa_hard(n, seed, max_scan=4000):
    import hashlib
    from datasets import load_dataset
    from measure_visual_depth import load_image, normalise
    ds = load_dataset("lmms-lab/textvqa", split="validation", streaming=True)
    raw, seen, scanned = [], set(), 0
    for ex in ds:                      # identical selection rule to build_samples
        if len(raw) >= n or scanned >= max_scan:
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
        refs = ex.get("answers") or []
        if not isinstance(refs, (list, tuple)):
            refs = [refs]
        if "ocr_tokens" not in ex:
            raise SystemExit("FATAL: lmms-lab/textvqa rows have no 'ocr_tokens' field")
        raw.append((img, str(ex.get("question") or ""), [str(r) for r in refs],
                    [str(t) for t in (ex.get("ocr_tokens") or [])]))

    samples = [{"id": "textvqa_%d" % i, "image": img, "question": q, "refs": refs}
               for i, (img, q, refs, _) in enumerate(raw)]
    base = _to_mc(samples, "textvqa", seed)        # supplies the cross-image fallback pool
    out, n_same, n_topped = [], 0, 0
    for s, b, (_, _, refs, ocr) in zip(samples, base, raw):
        c = b["mc_correct_text"]
        nref = {normalise(r) for r in refs}
        if normalise(c) in ("yes", "no"):
            out.append(b)
            continue
        k = max(1, len(str(c).split()))
        toks = [t for t in ocr if t.strip()]
        grams = [" ".join(toks[i:i + k]) for i in range(0, max(0, len(toks) - k + 1))]
        cand = []
        for g in dict.fromkeys(grams):
            ng = normalise(g)
            if not ng or ng in nref or any(ng in r or r in ng for r in nref if r):
                continue
            # No numeric tolerance here: options are scored by letter, and a near-miss
            # number printed in the same image (1999 for 1998) is exactly the hard case.
            cand.append(g)
        r = random.Random("hard|{}|{}".format(seed, s["id"]))
        r.shuffle(cand)
        dis = cand[:3]
        if len(dis) == 3:
            n_same += 1
        else:
            n_topped += 1
            extra = [o for o in b["options"] if normalise(o) != normalise(c)
                     and normalise(o) not in {normalise(d) for d in dis}]
            dis += extra[:3 - len(dis)]
        opts = [c] + dis
        r.shuffle(opts)
        ans = LETTERS[[normalise(o) for o in opts].index(normalise(c))]
        t = dict(s)
        t.update({"question": _mc_question(s["question"], opts), "options": opts, "answer": ans,
                  "suffix": MC_SUFFIX, "mc_correct_text": c,
                  "same_image_distractors": len(cand[:3])})
        out.append(t)
    print("hard distractors: {} samples with 3 same-image distractors, {} topped up from the "
          "cross-image pool".format(n_same, n_topped))
    return out


# ---------------------------------------------------------------------------
#  MIRROR FORMAT CONTROL: multiple-choice tasks asked open-ended
#
#  The reverse of textvqa_mc. MMBench and AI2D are early-horizon tasks, both multiple
#  choice. If answer format drives the horizon, asking the SAME questions without options
#  should move them late.
#
#  Not every multiple-choice question has a sensible open form, so the set is filtered
#  BEFORE sampling, and both formats run on the filtered set (same ids, same images):
#    - the question must not refer to its options ("which of the following", "option",
#      "choose", "select", "listed", "below", "above", "these")
#    - the correct option is short (at most 4 words, more than 1 character) and the
#      options are distinct
#  Open answers are scored against the correct option's text: exact or whole-phrase match,
#  and no other option may also appear in the answer (a hedge like "red or blue" fails).
# ---------------------------------------------------------------------------

_REFERS_TO_OPTIONS = ("following", "option", "choose", "select", "listed", "below", "above",
                      "these", "which one of", "which of the")


def _mirror_ok(question, hint, opts, correct):
    from measure_visual_depth import normalise
    text = (str(question) + " " + str(hint or "")).lower()
    if any(w in text for w in _REFERS_TO_OPTIONS):
        return False
    nc = normalise(correct)
    if len(nc) <= 1 or len(str(correct).split()) > 4:
        return False
    norm = [normalise(o) for o in opts]
    return len(set(norm)) == len(norm) and all(norm)


def build_mirror(which, fmt, n, seed, max_scan=20000):
    from measure_visual_depth import load_image
    ds = _stream("lmms-lab/MMBench", "en", "dev", seed) if which == "mmbench" else \
        _stream("lmms-lab/ai2d", None, "test", seed)
    out, k = [], 0
    for ex in ds:
        if len(out) >= n or k >= max_scan:
            break
        k += 1
        if which == "mmbench":
            opts = [ex[c] for c in "ABCD" if ex.get(c) is not None and str(ex[c]).strip()
                    and str(ex[c]).strip().lower() != "nan"]
            letter = str(ex["answer"]).strip()
            if letter not in LETTERS[:len(opts)]:
                continue
            correct, hint, q, sid = opts[LETTERS.index(letter)], ex.get("hint"), ex["question"], \
                "mmbench_" + str(ex["index"])
        else:
            opts = list(ex["options"])
            idx = int(ex["answer"])
            if idx >= len(opts):
                continue
            correct, hint, q, sid = opts[idx], None, ex["question"], "ai2d_%d" % k
        if len(opts) < 2 or not _mirror_ok(q, hint, opts, correct):
            continue
        img = load_image(ex.get("image"))
        if img is None:
            continue
        s = {"id": sid, "image": img, "options": opts, "mc_correct_text": str(correct),
             "refs": [str(correct)], "others": [str(o) for o in opts if str(o) != str(correct)]}
        if fmt == "mc":
            s.update({"question": _mc_question(q, opts, hint),
                      "answer": LETTERS[opts.index(correct)], "suffix": MC_SUFFIX})
        else:
            qq = ("Hint: " + str(hint).strip() + "\n" if hint and str(hint).strip()
                  and str(hint).strip().lower() != "nan" else "") + str(q).strip()
            s.update({"question": qq, "suffix": SHORT_SUFFIX})
        out.append(s)
    print("mirror {} {}: kept {} of {} scanned".format(which, fmt, len(out), k))
    return out


def _phrase_in(phrase, text):
    import re
    from measure_visual_depth import normalise
    p, t = normalise(phrase), normalise(text)
    return bool(p) and re.search(r"(^| )" + re.escape(p) + r"( |$)", t) is not None


def score_open_option(pred, s):
    from measure_visual_depth import normalise, as_float
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
             "ten", "eleven", "twelve"]
    raw = str(pred).strip().rstrip(".")
    pred = " ".join(str(words.index(w)) if w in words else w for w in raw.lower().split())
    p = normalise(pred)
    pf0 = as_float(raw)
    if pf0 is not None:
        p = normalise(str(int(pf0)) if float(pf0).is_integer() else raw)
    c = s["refs"][0]
    pf, cf = as_float(p), as_float(c)
    hit = (p == normalise(c)) or _phrase_in(c, pred) or \
        (pf is not None and cf is not None and pf == cf)
    if not hit:
        return False
    return not any(_phrase_in(o, pred) and not _phrase_in(o, c) for o in s["others"])


TASKS = {
    "textvqa":   {"build": build_textvqa,   "score": score_textvqa, "max_new_tokens": 16},
    "pope":      {"build": build_pope,      "score": score_pope,    "max_new_tokens": 4},
    "gqa":       {"build": build_gqa,       "score": score_exact,   "max_new_tokens": 16},
    "mmbench":   {"build": build_mmbench,   "score": score_mc,      "max_new_tokens": 8},
    "scienceqa": {"build": build_scienceqa, "score": score_mc,      "max_new_tokens": 8},
    "ai2d":      {"build": build_ai2d,      "score": score_mc,      "max_new_tokens": 8},
    "chartqa":   {"build": build_chartqa,   "score": score_relaxed, "max_new_tokens": 16},
    "docvqa":    {"build": build_docvqa,    "score": score_anls,    "max_new_tokens": 32},
    # format control: same questions as textvqa / chartqa, asked as multiple choice
    "textvqa_mc": {"build": build_textvqa_mc, "score": score_mc,    "max_new_tokens": 8},
    "chartqa_mc": {"build": build_chartqa_mc, "score": score_mc,    "max_new_tokens": 8},
    "textvqa_hard": {"build": build_textvqa_hard, "score": score_mc, "max_new_tokens": 8},
    # mirror control: the same filtered questions, multiple choice vs open-ended
    "mmbench_fmc":   {"build": lambda n, sd: build_mirror("mmbench", "mc", n, sd),
                      "score": score_mc, "max_new_tokens": 8},
    "mmbench_fopen": {"build": lambda n, sd: build_mirror("mmbench", "open", n, sd),
                      "score": score_open_option, "max_new_tokens": 16},
    "ai2d_fmc":      {"build": lambda n, sd: build_mirror("ai2d", "mc", n, sd),
                      "score": score_mc, "max_new_tokens": 8},
    "ai2d_fopen":    {"build": lambda n, sd: build_mirror("ai2d", "open", n, sd),
                      "score": score_open_option, "max_new_tokens": 16},
}


# ===========================================================================
#  Analysis. Pure Python, no GPU, so it can be rerun on any saved result.
# ===========================================================================

def curve_from(per_layer, idx=None):
    """per_layer: {l: [0/1 per sample]}. Returns {l: accuracy} over sample subset idx."""
    out = {}
    for l, v in per_layer.items():
        vv = v if idx is None else [v[i] for i in idx]
        out[int(l)] = sum(vv) / len(vv) if vv else float("nan")
    return out


def sustained_horizon(acc, n_layers, alpha, normalise=True):
    """Smallest l with g(l') >= alpha for every l' >= l. None if never sustained."""
    floor, base = acc[0], acc[n_layers]
    denom = (base - floor) if normalise else base
    if denom <= 0:
        return None
    ls = sorted(acc)
    h = None
    for l in reversed(ls):
        g = ((acc[l] - floor) if normalise else acc[l]) / denom
        if g >= alpha:
            h = l
        else:
            break
    return h


def bootstrap_ci(per_layer, n_layers, alpha, B=1000, seed=0, normalise=True):
    n = len(next(iter(per_layer.values())))
    rng = random.Random(seed)
    hs = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        h = sustained_horizon(curve_from(per_layer, idx), n_layers, alpha, normalise)
        if h is not None:
            hs.append(h)
    if len(hs) < 0.9 * B:
        return None, None, len(hs) / B
    hs.sort()
    return hs[int(0.025 * len(hs))], hs[int(0.975 * len(hs)) - 1], len(hs) / B


def per_sample_horizons(per_layer, n_layers):
    """Per-sample horizon, restricted to samples that NEED the image: correct at baseline
    and wrong at l = 0. Samples already right with no image (a yes/no guess that happens to
    match, a multiple-choice prior) carry no information about visual access and would drag
    the distribution towards zero, as they did for POPE in the pilot.
    A score counts as correct at >= 0.5, which is exact for 0/1 tasks and the usual
    acceptance level for ANLS."""
    ok = lambda v: v >= 0.5
    n = len(per_layer[n_layers])
    ls = sorted(per_layer)
    out, prior = [], 0
    for i in range(n):
        if not ok(per_layer[n_layers][i]):
            continue
        if ok(per_layer[0][i]):
            prior += 1
            continue
        h = None
        for l in reversed(ls):
            if ok(per_layer[l][i]):
                h = l
            else:
                break
        out.append(h)
    return out, prior


def analyse(res):
    L = res["n_layers"]
    pl = {int(k): v for k, v in res["per_layer"].items()}
    missing = [l for l in range(L + 1) if l not in pl]
    print("=" * 70)
    print("task {}   model {}   n {}   layers {}".format(
        res["task"], res["model"], len(res["sample_ids"]), L))
    if missing:
        print("INCOMPLETE, layers still to run:", missing)
        print("=" * 70)
        return {}
    acc = curve_from(pl)
    floor, base = acc[0], acc[L]
    print("floor (0 layers of access) {:.4f}   baseline {:.4f}   image contribution {:.4f}"
          .format(floor, base, base - floor))
    if base - floor < 0.05:
        print("WARNING: the image adds under 5 points on this task. Horizons below are"
              " dominated by noise and should not be interpreted.")
    print("\n  l   acc     % base   g(l)")
    for l in range(L + 1):
        g = (acc[l] - floor) / (base - floor) if base > floor else float("nan")
        print(" {:3d}  {:.4f}  {:6.1f}  {:6.3f}".format(l, acc[l], 100 * acc[l] / base if base else 0, g))

    summary = {"floor": floor, "baseline": base, "horizons": {}}
    print("\nsustained horizon, layers with access (95% bootstrap CI)")
    for a in ALPHAS:
        for norm in (True, False):
            h = sustained_horizon(acc, L, a, norm)
            lo, hi, ok = bootstrap_ci(pl, L, a, normalise=norm)
            key = "{}_{}".format("norm" if norm else "raw", a)
            summary["horizons"][key] = {"h": h, "ci": [lo, hi], "defined_frac": ok,
                                         "depth_pct": (100.0 * h / L) if h is not None else None}
            print("  {:>4} alpha {:.2f}:  h = {:>4}  ({})   CI {}".format(
                "norm" if norm else "raw", a, str(h),
                "{:.1f}% depth".format(100.0 * h / L) if h is not None else "n/a",
                "[{}, {}]".format(lo, hi) if lo is not None else "unstable (defined in {:.0%})".format(ok)))

    ps, prior = per_sample_horizons(pl, L)
    ps = [h for h in ps if h is not None]
    print("\n{} samples correct at baseline were already correct with no image (excluded)"
          .format(prior))
    if ps:
        ps.sort()
        q = lambda f: ps[min(len(ps) - 1, int(f * len(ps)))]
        summary["per_sample"] = {"n": len(ps), "excluded_prior": prior,
                                 "q25": q(0.25), "median": q(0.5), "q75": q(0.75)}
        print("per-sample horizon, {} samples that need the image: q25 {}  median {}  q75 {}"
              .format(len(ps), q(0.25), q(0.5), q(0.75)))
    t = res.get("seconds_per_layer", {})
    if t:
        m = sum(t.values()) / len(t)
        print("\ncost: {:.0f} s per layer on average, {:.1f} h for the full curve".format(m, m * (L + 1) / 3600))
    print("=" * 70)
    return summary


# ===========================================================================
#  Measurement
# ===========================================================================

def save(res, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--task", choices=sorted(TASKS))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", default="all",
                    help="'all' for 0..L, or a comma list. Baseline (l = L) is always run.")
    ap.add_argument("--out")
    ap.add_argument("--analyse", metavar="JSON", help="analyse a saved result and exit")
    args = ap.parse_args()

    if args.analyse:
        with open(args.analyse) as f:
            res = json.load(f)
        s = analyse(res)
        if s:
            res["summary"] = s
            save(res, args.analyse)
        return

    if not args.task or not args.out:
        ap.error("--task and --out are required unless --analyse is given")

    import torch
    from measure_visual_depth import VisualTokenDeleter, pick_loader

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    task = TASKS[args.task]

    bind, ivl_state = None, None
    if "internvl" in args.model.lower():
        from internvl_adapter import load_internvl
        model, layers, run, generate, bind, ivl_state = load_internvl(args.model)
    else:
        model, layers, run, generate = pick_loader(args.model)(args.model)
    L = len(layers)
    todo = list(range(L + 1)) if args.layers == "all" else \
        sorted({int(x) for x in args.layers.split(",") if x.strip()} | {0, L})

    samples = task["build"](args.n, args.seed)
    ids = [s["id"] for s in samples]
    print("model   :", args.model, "  decoder layers:", L)
    print("task    :", args.task, "  samples:", len(samples))
    if args.task == "pope":
        print("balance :", sum(s["answer"] == "yes" for s in samples), "yes /",
              sum(s["answer"] == "no" for s in samples), "no")

    # Resume, but only onto the identical sample set.
    res = {"model": args.model, "task": args.task, "n_layers": L, "seed": args.seed,
           "sample_ids": ids, "per_layer": {}, "seconds_per_layer": {}}
    if os.path.exists(args.out):
        with open(args.out) as f:
            old = json.load(f)
        if old.get("sample_ids") == ids and old.get("model") == args.model:
            res = old
            print("resuming, already done:", sorted(int(k) for k in res["per_layer"]))
        else:
            raise SystemExit("FATAL: {} exists with a different sample set or model. "
                             "Refusing to mix. Move it aside.".format(args.out))

    deleter = VisualTokenDeleter(model, layers, 0)
    if bind is not None:
        # InternVL: positions are only visible inside model.generate, so the adapter hands
        # them to the deleter from there.
        bind(deleter)

    # Baseline first, so the % column is meaningful from the first printed line.
    order = [L] + [l for l in todo if l != L]
    for l in order:
        if str(l) in res["per_layer"]:
            continue
        t0 = time.time()
        armed = l < L
        if armed:
            deleter.layer_index = l
            deleter.fired = 0
            deleter.arm()
        flags, preds, failed = [], [], 0
        try:
            for s in samples:
                inp, positions, _ = run(s["image"], s["question"],
                                     suffix=s.get("suffix", SHORT_SUFFIX),
                                     max_pixels=s.get("max_pixels"))
                deleter.begin_sample(positions)
                try:
                    pred = generate(inp, max_new_tokens=task["max_new_tokens"])
                except Exception as e:
                    print("  generation failed on", s["id"], repr(e))
                    flags.append(0)   # scored wrong, and counted so the hook check allows it
                    preds.append(None)
                    failed += 1
                    continue
                preds.append(pred)
                ok = task["score"](pred, s)
                flags.append(float(ok) if not isinstance(ok, bool) else (1 if ok else 0))

        finally:
            if armed:
                deleter.disarm()
        dt = time.time() - t0

        if armed and deleter.fired < len(samples) - failed:
            # A void measurement must never be saved as if it were real.
            raise SystemExit("FATAL: hook fired {} times for {} samples at layer {}"
                             .format(deleter.fired, len(samples), l))

        res["per_layer"][str(l)] = flags
        res["seconds_per_layer"][str(l)] = dt
        res.setdefault("failures_per_layer", {})[str(l)] = failed
        # Raw answers, so claims like "the floor is one repeated answer" can be shown, not inferred.
        res.setdefault("predictions", {})[str(l)] = preds
        save(res, args.out)
        acc = sum(flags) / len(flags)
        base = sum(res["per_layer"][str(L)]) / len(samples)
        print("  l {:3d}  acc {:.4f}  ({:5.1f}% of baseline)  {:5.0f} s{}".format(
            l, acc, 100 * acc / base if base else 0, dt, "" if armed else "   [baseline]"),
            flush=True)
        torch.cuda.empty_cache()

    s = analyse(res)
    res["summary"] = s
    save(res, args.out)
    print("saved to", args.out)


if __name__ == "__main__":
    main()
