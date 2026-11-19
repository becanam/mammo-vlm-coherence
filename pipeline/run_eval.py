"""
Evaluation runner (self-contained). For every sampled MammoVQA item it queries one vision-language
model under each condition (real image, unrelated photo, blank image, no image, reworded question,
shuffled options + fillers) and records, in one generation, the model's EXPLICIT stated answer
(the primary signal), a per-option 0-100 confidence score, and a one-sentence explanation. Works for
RSNA and EMBED and all four topics (BI-RADS, density/ACR, laterality, view). Run in place:
`cd pipeline && python run_eval.py ...`.
"""

import os
import re
import json
import random
import argparse
import glob as glob_mod

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALPHA = [chr(65 + i) for i in range(26)]           
ABSTAIN = "None of the above"
N_FILLERS = 2                                         

# for the perturbed_options condition (only using first two)
PAD_FILLERS = ["Not Applicable", "Not Assessable", "Cannot Be Determined",
               "Indeterminate", "Other", "Unclear"]

# for perturbed_questions condition: drop domain words -> generic "image"
NEUTRAL_RULES = [
    (r"\bmammograms?\b", "image"),
    (r"\bmammography\b", "imaging"),
    (r"\bbreasts?\b", "image"),
]

_RSNA_RE = re.compile(r"/Benchmark/RSNA/(.+)_img/img\.jpg")
_EMBED_RE = re.compile(r"/Benchmark/EMBED/")

# option sets per topic
TASKS = {
    "laterality": {"topic": "Laterality", "options": ["Right", "Left"], "answer_in": None},
    "view":       {"topic": "View",       "options": ["MLO", "CC"],     "answer_in": None},
    "acr":        {"topic": "ACR",
                   "options": ["Level A", "Level B", "Level C", "Level D"], "answer_in": None},
    "birads":     {"topic": "Bi-Rads",
                   "options": ["Bi-Rads 0", "Bi-Rads 1", "Bi-Rads 2", "Bi-Rads 3",
                               "Bi-Rads 4", "Bi-Rads 5", "Bi-Rads 6"], "answer_in": None},
}


# ---------------------------------------------------------------------------
# Benchmark loading + sampling
# ---------------------------------------------------------------------------
def load_benchmark(path):
    d = json.load(open(path))
    return list(d.values()) if isinstance(d, dict) else d


def sample_task(items, cfg, datasets, n, seed):
    pool = [i for i in items
            if i.get("Question topic") == cfg["topic"]
            and isinstance(i.get("Answer"), str)
            and i.get("Dataset") in datasets
            and (cfg["answer_in"] is None or i["Answer"] in cfg["answer_in"])]
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n] if n else pool


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------
def load_image(image_root, path, max_side=1024):
    import cv2
    from PIL import Image
    fp = os.path.join(image_root, path.lstrip("/"))
    im = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(fp)
    h, w = im.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return Image.fromarray(im).convert("RGB")


def load_any(path, max_side=1024):
    """Load any image file directly by path (unrelated images)."""
    import cv2
    from PIL import Image
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(path)
    h, w = im.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return Image.fromarray(im).convert("RGB")


def blank_image(size=512):
    from PIL import Image
    return Image.new("RGB", (size, size), (128, 128, 128))


def _resolve_one(path, image_root, rsna_png_root, embed_png_root):
    """Resolve a benchmark path to a PIL image, redirecting RSNA -> rsna_png/<id>.png and
    EMBED -> embed_png/<name>.jpg (the downscaled public-cohort images); else load from image_root."""
    m = _RSNA_RE.match(path)
    if m and rsna_png_root:
        p = os.path.join(rsna_png_root, m.group(1) + ".png")
        if os.path.exists(p):
            return load_any(p)
    if _EMBED_RE.match(path) and embed_png_root:
        p = os.path.join(embed_png_root, os.path.basename(path).rsplit(".", 1)[0] + ".jpg")
        if os.path.exists(p):
            return load_any(p)
    return load_image(image_root, path)


def resolve_images(path, image_root, rsna_png_root, embed_png_root):
    """Return a LIST of images (EMBED exams are multi-view lists; single-image datasets give one)."""
    paths = path if isinstance(path, list) else [path]
    return [_resolve_one(p, image_root, rsna_png_root, embed_png_root) for p in paths]


# ---------------------------------------------------------------------------
# Options + prompts + parsing
# ---------------------------------------------------------------------------
def neutralize_question(q):
    """Replace domain entity words (mammogram/breast) with neutral 'image'."""
    out = q or ""
    for pat, rep in NEUTRAL_RULES:
        out = re.sub(pat, rep, out, flags=re.I)
    out = re.sub(r"\bimage\s+image\b", "image", out, flags=re.I)
    out = re.sub(r"\bimage\s+imaging\b", "imaging", out, flags=re.I)
    return out


def build_option_set(options, abstain=True, n_fillers=0, rng=None):
    opts = list(options)
    added = set()
    for f in PAD_FILLERS:
        if len(added) >= n_fillers:
            break
        if f not in opts:
            opts.append(f)
            added.add(f)
    if abstain and ABSTAIN not in opts:
        opts.append(ABSTAIN)
        added.add(ABSTAIN)
    if rng is not None:
        rng.shuffle(opts)
    return opts, added


def conf_prompt(question, options, labels, text_only=False):
    """MCQ + abstain + a 0-100 score for EACH option + a one-line explanation."""
    lettered = "\n".join(f"{labels[i]}) {o}" for i, o in enumerate(options))
    fmt = "\n".join(f"{labels[i]}: <0-100>" for i in range(len(options)))
    return (
        "This is a mammography-related medical question with several options, only one of which "
        "is correct.\n"
        f"### Question: {question}\n"
        f"### Options:\n{lettered}\n\n"
        "For EACH option, give a confidence score from 0 to 100 for how likely THAT option is the "
        "correct answer (score each option independently; they need not sum to 100). Then give a "
        "brief one-sentence explanation of your reasoning.\n"
        "Respond in EXACTLY this format, one line per option, then the explanation, and nothing else:\n"
        f"{fmt}\n"
        "Explanation: <one sentence>"
    )


def conf_prompt_explanation_first(question, options, labels, text_only=False):
    """MCQ + abstain + explanation FIRST + 0-100 confidence for EACH option (reordered output)."""
    lettered = "\n".join(f"{labels[i]}) {o}" for i, o in enumerate(options))
    fmt = "\n".join(f"{labels[i]}: <0-100>" for i in range(len(options)))
    return (
        "This is a mammography-related medical question with several options, only one of which "
        "is correct.\n"
        f"### Question: {question}\n"
        f"### Options:\n{lettered}\n\n"
        "First, provide a brief one-sentence explanation of your reasoning. "
        "Then, for EACH option, give a confidence score from 0 to 100 for how likely THAT option "
        "is the correct answer. Score each option independently; the scores do not need to sum to 100.\n"
        "Respond in EXACTLY this format, with the explanation first, followed by one line per option, "
        "and nothing else:\n"
        "Explanation: <one sentence>\n"
        f"{fmt}"
    )


_SCORE_LINE = re.compile(r'^\s*\(?\s*([A-Za-z]{1,4})\s*[):.\-]')


def parse_explanation(raw):
    m = re.search(r'(?:explanation|reasoning)\s*[:\-]\s*(.+)', raw or "", re.I | re.S)
    if not m:
        return None
    lines = [ln.strip() for ln in m.group(1).splitlines()
             if ln.strip() and not _SCORE_LINE.match(ln)]
    return " ".join(lines) or None


def parse_choice(text, options, labels):
    t = (text or "").strip().upper()
    ul = [l.upper() for l in labels]
    for i in sorted(range(len(labels)), key=lambda i: -len(ul[i])):   # longest label first
        if re.search(rf'(?<![A-Z0-9]){re.escape(ul[i])}(?![A-Z0-9])', t):
            return i
    for i, opt in enumerate(options):
        if opt.upper() in t:
            return i
    return None


def parse_scores(raw, labels):
    lab_idx = {l.upper(): i for i, l in enumerate(labels)}
    scores = {}
    for m in re.finditer(r'\(?\s*([A-Za-z]{1,4})\s*[):.\-=]\s*(\d{1,3})', raw or ""):
        k = m.group(1).upper()
        if k in lab_idx:
            scores.setdefault(lab_idx[k], min(100, int(m.group(2))))
    for line in (raw or "").splitlines():
        m = _SCORE_LINE.match(line)
        if not m:
            continue
        k = m.group(1).upper()
        if k in lab_idx and lab_idx[k] not in scores:
            nums = re.findall(r'\d{1,3}', line)
            if nums:
                scores[lab_idx[k]] = min(100, int(nums[-1]))
    return scores


def parse_conf(raw, options, labels):
    expl = parse_explanation(raw)
    score_text = "\n".join(
        ln for ln in (raw or "").splitlines()
        if not re.match(r'\s*(?:explanation|reasoning)\s*[:\-]', ln, re.I)
    )
    sc = parse_scores(score_text, labels)
    if sc:
        scores_list = [sc.get(i) for i in range(len(options))]
        idx = max(sc, key=lambda k: (sc[k], -k))
        return scores_list, idx, options[idx], float(sc[idx]), expl
    idx = parse_choice(score_text, options, labels)
    return [None] * len(options), idx, (options[idx] if idx is not None else None), None, expl


def conf_prompt_with_answer(question, options, labels, text_only=False):
    """SINGLE prompt eliciting all three signals at once: (1) the model's EXPLICIT best answer,
    (2) a 0-100 confidence score for each option, and (3) a one-sentence explanation. The answer is
    requested FIRST so it is the model's lead commitment, then the per-option scores, then the
    reasoning -- all in one generation. This is the default prompt: the paper's primary answer
    signal is this explicitly stated answer."""
    lettered = "\n".join(f"{labels[i]}) {o}" for i, o in enumerate(options))
    fmt = "\n".join(f"{labels[i]}: <0-100>" for i in range(len(options)))
    return (
        "This is a mammography-related medical question with several options, only one of which "
        "is correct.\n"
        f"### Question: {question}\n"
        f"### Options:\n{lettered}\n\n"
        "Do THREE things, in this order:\n"
        "1) State your single best answer as the option letter.\n"
        "2) For EACH option, give a confidence score from 0 to 100 for how likely THAT option is the "
        "correct answer (score each option independently; they need not sum to 100).\n"
        "3) Give a brief one-sentence explanation of your reasoning.\n"
        "Respond in EXACTLY this format, and nothing else:\n"
        "Answer: <letter>\n"
        f"{fmt}\n"
        "Explanation: <one sentence>"
    )


def parse_answer_line(raw, options, labels):
    """Pull the EXPLICIT choice from the 'Answer: <letter>' line of a joint reply -> (idx, text).
    Falls back to a whole-response choice parse. Returns (None, None) if nothing parseable."""
    m = re.search(r'^\s*answer\s*[:\-]\s*(.+)$', raw or "", re.I | re.M)
    if m:
        idx = parse_choice(m.group(1), options, labels)
        if idx is not None:
            return idx, options[idx]
    idx = parse_choice(raw or "", options, labels)
    return (idx, options[idx]) if idx is not None else (None, None)


def parse_conf_full(raw, options, labels):
    """Joint parse of a conf_prompt_with_answer() reply. Returns everything parse_conf returns PLUS
    the explicit answer: (a_idx, a_text, scores_list, pred_idx, pred_text, confidence, explanation).
    The 'Answer:' line carries no number so it does not pollute the per-option score parse."""
    a_idx, a_text = parse_answer_line(raw, options, labels)
    scores_list, pred_idx, pred_text, conf, expl = parse_conf(raw, options, labels)
    return a_idx, a_text, scores_list, pred_idx, pred_text, conf, expl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    topics = [t.strip() for t in args.topics.split(",") if t.strip()]
    datasets = set(d.strip() for d in args.datasets.split(",") if d.strip())
    rng = random.Random(args.seed)

    items = load_benchmark(args.benchmark)
    sampled = {t: sample_task(items, TASKS[t], datasets, args.n, args.seed) for t in topics}
    for t in topics:
        print(f"  {t}: {len(sampled[t])} sampled  (options={TASKS[t]['options']})")

    unrelated_pool = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        if args.unrelated_root:
            unrelated_pool += glob_mod.glob(os.path.join(args.unrelated_root, "**", ext), recursive=True)
    if not unrelated_pool:
        print("WARNING: no --unrelated_root images; 'unrelated' condition will be skipped")
    blank_img = blank_image()

    from vlm_inference import load_model, run_inference_multi, run_text_inference
    model, processor, mtype = load_model(args.model_name)

    def infer(imgs, prompt):
        # imgs is None (no-image) or a LIST of images (>=1)
        if imgs is None:
            return run_text_inference(model, processor, mtype, prompt, max_new_tokens=384)
        return run_inference_multi(model, processor, mtype, imgs, prompt, max_new_tokens=384)

    # also have correctness computed against "None of the Above" for junk images
    abstain_conds = set(c.strip() for c in args.abstain_conditions.split(",") if c.strip())

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "results.json")

    def save(rows):
        with open(out + ".tmp", "w") as f:
            json.dump({"model": args.model_name, "datasets": sorted(datasets),
                       "topics": topics, "n": args.n, "abstain_conditions": sorted(abstain_conds),
                       "rows": rows}, f)
        os.replace(out + ".tmp", out)

    # resume: reuse rows already computed if the output exists
    rows, done = [], set()
    if os.path.exists(out):
        try:
            rows = json.load(open(out)).get("rows", [])
            done = {(r["task"], tuple(r["path"]) if isinstance(r["path"], list) else r["path"])
                    for r in rows}
            print(f"resuming: {len(done)} items already done, {len(rows)} rows loaded")
        except Exception as e:
            print(f"could not resume ({e}); starting fresh")
            rows, done = [], set()

    ucache = {}

    def draw_unrelated(k):
        """Draw k DISTINCT unrelated images so the 'unrelated' condition matches the image
        COUNT of the real condition"""
        if not unrelated_pool:
            return []
        picks = rng.sample(unrelated_pool, min(k, len(unrelated_pool)))
        imgs = []
        for up in picks:
            if up not in ucache:
                try: ucache[up] = load_any(up)
                except Exception: ucache[up] = None
            if ucache[up] is not None:
                imgs.append(ucache[up])
        return imgs

    processed = 0
    for t in topics:
        base_opts = TASKS[t]["options"]
        for it in tqdm(sampled[t], desc=t):
            path, gt = it["Path"], it["Answer"]
            pkey = tuple(path) if isinstance(path, list) else path
            if (t, pkey) in done:
                continue
            q = it["Question"]
            try:
                real = resolve_images(path, args.image_root, args.rsna_png, args.embed_png)
            except Exception as e:
                rows.append({"task": t, "path": path, "error": f"img: {e}"})
                done.add((t, pkey)); continue
            n_views = len(real)
            uimgs = draw_unrelated(n_views)   # N distinct unrelated images, matching the view count

            opts_std, add_std = build_option_set(base_opts, abstain=True)
            po_rng = random.Random(f"{pkey}|{gt}")
            opts_po, add_po = build_option_set(base_opts, abstain=True, n_fillers=N_FILLERS, rng=po_rng)
            plan = [
                ("original",           real,          q,                        opts_std, add_std),
                ("blank",              [blank_img] * n_views, q,                opts_std, add_std),
                ("noimg",              None,          q,                        opts_std, add_std),
                ("perturbed_question", real,          neutralize_question(q),   opts_std, add_std),
                ("perturbed_options",  real,          q,                        opts_po,  add_po),
            ]
            if uimgs:
                plan.insert(1, ("unrelated", uimgs, q, opts_std, add_std))

            # Default: the joint answer-first prompt, whose EXPLICIT stated answer is the paper's
            # primary signal. --explanation_first / --scores_only select the legacy score-only prompts.
            if args.scores_only:
                prompt_fn = conf_prompt_explanation_first if args.explanation_first else conf_prompt
            else:
                prompt_fn = conf_prompt_with_answer
            item_rows = []
            for cond, img, question, opts, added in plan:
                labels = ALPHA[:len(opts)]
                prompt = prompt_fn(question, opts, labels, text_only=(img is None))
                raw = infer(img, prompt)
                target = ABSTAIN if cond in abstain_conds else gt
                row = {
                    "task": t, "path": path, "condition": cond,
                    "question": question, "options": opts,
                    "gt": target, "orig_gt": gt,         # gt = scoring target; orig_gt = benchmark label
                    "raw": raw,
                }
                if args.scores_only:
                    scores, idx, pred_text, conf, expl = parse_conf(raw, opts, labels)
                else:
                    a_idx, a_text, scores, idx, pred_text, conf, expl = parse_conf_full(raw, opts, labels)
                    # explicit stated answer (primary signal)
                    row["explicit_idx"] = a_idx
                    row["explicit_answer"] = a_text
                    row["explicit_correct"] = int(a_text == target) if a_text is not None else 0
                row.update({
                    "scores": scores,
                    "pred": idx, "pred_text": pred_text,   # score arg-max (secondary signal)
                    "confidence": conf,
                    "explanation": expl,
                    "correct": int(pred_text == target) if pred_text is not None else 0,
                    "picked_new": int(pred_text in added) if pred_text is not None else 0,
                })
                item_rows.append(row)
            rows.extend(item_rows)
            done.add((t, pkey))
            processed += 1
            if processed % args.save_every == 0:      # partial save for safety
                save(rows)

    save(rows)
    parsed = sum(1 for r in rows if r.get("confidence") is not None)
    expl_ans = sum(1 for r in rows if r.get("explicit_answer") is not None)
    print(f"wrote {out}  ({len(rows)} rows, {expl_ans} with an explicit answer, "
          f"{parsed} with a parsed confidence)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="MammoVQA-Image-Bench.json")
    ap.add_argument("--image_root", default=".",
                    help="fallback root for benchmark images (RSNA resolves via --rsna_png)")
    ap.add_argument("--rsna_png", default="rsna_png",
                    help="dir of RSNA PNGs (<pid>_<iid>.png); RSNA paths resolve here")
    ap.add_argument("--embed_png", default="embed_png",
                    help="dir of EMBED JPEGs (<name>.jpg); EMBED paths resolve here")
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--datasets", default="RSNA,EMBED")
    ap.add_argument("--topics", default="birads,acr,laterality,view")
    ap.add_argument("--n", type=int, default=20000, help="max questions per topic (>= all = use all)")
    ap.add_argument("--unrelated_root", default="unrelated_diverse")
    ap.add_argument("--scores_only", action="store_true",
                    help="legacy: elicit only per-option scores + explanation (no explicit answer line); "
                         "the default joint prompt also asks for the explicit stated answer")
    ap.add_argument("--explanation_first", action="store_true",
                    help="with --scores_only, put the explanation before the per-option scores")
    ap.add_argument("--abstain_conditions", default="unrelated,blank,noimg",
                    help="conditions with no valid mammogram: ground truth becomes 'None of the above' "
                         "(so abstaining is scored correct). Set to '' to score every condition vs the label.")
    ap.add_argument("--save_every", type=int, default=20,
                    help="write partial results to disk every N completed items (crash safety + resume)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", default="output_eval")
    main(ap.parse_args())
