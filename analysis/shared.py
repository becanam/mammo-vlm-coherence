"""
Shared utilities for the analysis modules: model registry, colours/markers, the model's answer
(`out_pred` = the EXPLICIT stated answer, the paper's primary signal; keep-ties), the combined
reasoning-answer, and result loading. Keeps the per-figure boilerplate in one place so the section
modules (coherence, confidence, nota, perturbation) stay thin.
"""
import json, glob as glob_mod, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: F401  (re-exported for the modules)

from signals.answer_mapping import matchmaker_pred, ABSTAIN, is_tie
from signals.reasoning_parser import reason_pred as _reason_pred, NOTA

# ------------------------------------------------------------------ registry
MODEL_ORDER = ["medgemma-4b-it", "Lingshu-7B", "Qwen2-VL-7B-Instruct", "InternVL3-8B"]
MODEL_COL = {"medgemma-4b-it": "#E41A1C", "Lingshu-7B": "#377EB8",
             "Qwen2-VL-7B-Instruct": "#4DAF4A", "InternVL3-8B": "#FF7F00"}
MODEL_MARKER = {"medgemma-4b-it": "o", "Lingshu-7B": "s", "Qwen2-VL-7B-Instruct": "^",
                "InternVL3-8B": "P"}
TASK_LABEL = {"birads": "BI-RADS", "acr": "Density (ACR)", "laterality": "Laterality", "view": "View"}

COND = ["original", "perturbed_question", "perturbed_options", "unrelated", "blank", "noimg"]
CLAB = {"original": "original", "perturbed_question": "reworded\nQ", "perturbed_options": "shuffled\nopts",
        "unrelated": "unrelated\nphoto", "blank": "blank", "noimg": "no\nimage"}
JUNK = {"unrelated", "blank", "noimg"}

FILLERS = {"Not Applicable", "Not Assessable", "Cannot Be Determined", "Indeterminate", "Unclear"}
ABSTAIN_LIKE = {ABSTAIN} | FILLERS

# ------------------------------------------------------------------ per-row helpers
def short(m): return (m or "").split("/")[-1]
def gtof(r): return r.get("orig_gt", r.get("gt"))
def pkey(r): return tuple(r["path"]) if isinstance(r.get("path"), list) else r.get("path")


def out_pred(r, tau=1.0):
    """The model's answer = its EXPLICITLY stated single-best choice (the paper's primary signal).

    The explicit answer is always well-defined (no tie ambiguity), so it is used throughout.
    For legacy result files that predate the explicit-answer elicitation, fall back to the
    score arg-max f(A) (matchmaker) and then to the raw text choice."""
    ea = r.get("explicit_answer")
    if ea is not None:
        return ea
    p = matchmaker_pred(r.get("scores") or [], r.get("options") or [], tau)
    return p if p is not None else r.get("pred_text")


def reason_pred(r):
    """Combined reasoning-answer for a result row (or None if unparseable)."""
    return _reason_pred(r.get("explanation"), r.get("task"), r.get("options") or [])


def stated_conf(r):
    """Confidence on the model's STATED answer = the score it gave that option (0-100), or None.

    This is the per-option confidence vector read at the explicitly stated answer. If that option
    carries no parseable score, the answer contributes no confidence value (None), matching the
    paper's handling. Legacy files without an explicit answer fall back to the top score."""
    ea = r.get("explicit_answer"); opts = r.get("options") or []; sc = r.get("scores") or []
    if ea in opts:
        i = opts.index(ea)
        if i < len(sc) and sc[i] is not None:
            return float(sc[i])
        return None
    s = [x for x in sc if x is not None]
    if s: return float(max(s))
    c = r.get("confidence")
    return float(c) if c is not None else None


# ------------------------------------------------------------------ loading
def order_models(present):
    return [m for m in MODEL_ORDER if m in present] + [m for m in present if m not in MODEL_ORDER]


def load_by_model(glob_pat, exclude=(), topics=None, conditions=None):
    """glob -> {model_short: [rows]} filtered by topics/conditions, excluding model-name substrings."""
    excl = [x.lower() for x in exclude if x]
    by = defaultdict(list)
    for p in glob_mod.glob(glob_pat):
        d = json.load(open(p)); m = short(d.get("model"))
        if any(x in m.lower() for x in excl): continue
        for r in d.get("rows", []):
            if "error" in r: continue
            if topics is not None and r.get("task") not in topics: continue
            if conditions is not None and r.get("condition") not in conditions: continue
            by[m].append(r)
    return by


def add_std_args(ap, topics="birads,acr"):
    ap.add_argument("--glob", default="output_*/results.json")
    ap.add_argument("--topics", default=topics)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--discard-ties", dest="discard_ties", action="store_true", default=False,
                    help="(legacy) drop positive-tie items from the score arg-max; OFF by default, "
                         "since the explicit stated answer is always well-defined")
    ap.add_argument("--keep-ties", dest="discard_ties", action="store_false")
    return ap


def topics_of(a): return [t.strip() for t in a.topics.split(",") if t.strip()]
def exclude_of(a): return [x.strip() for x in a.exclude.split(",") if x.strip()]


def save(fig, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
