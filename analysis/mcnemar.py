"""
Paired significance of each elicitation perturbation on answer correctness, via the exact McNemar
test. For every model x task x perturbation we pair the original and perturbed responses by
(task, image path), score each with the EXPLICIT stated answer, and count the discordant pairs:

  C->W : originally correct, wrong after the perturbation
  W->C : originally wrong, correct after the perturbation

The exact two-sided McNemar p-value is a binomial test on (min(C->W, W->C), C->W + W->C) under
p = 0.5, computed with scipy.stats.binomtest. For perturbed_question we keep only items whose wording
actually changed. Keep-ties: the explicit answer is always well-defined, so no items are dropped.

  python -m analysis.mcnemar --glob "output_*/results.json" \
      --topics birads,acr --out mcnemar.csv
"""
import argparse, csv, glob as glob_mod, json
from collections import defaultdict
from scipy.stats import binomtest

from analysis.shared import (TASK_LABEL, order_models, out_pred, gtof, pkey, short,
                             topics_of, exclude_of)

PERTS = [("perturbed_question", "Question"), ("perturbed_options", "Options")]


def mcnemar_exact(b, c):
    """Two-sided exact binomial McNemar p-value on discordant counts b, c (SciPy)."""
    n = b + c
    if n == 0:
        return 1.0
    return binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue


def _load(glob_pat, exclude, topics):
    """{model: {(task, pkey): {condition: row}}} for original + the two perturbations."""
    excl = [x.lower() for x in exclude]
    by = defaultdict(lambda: defaultdict(dict))
    for p in glob_mod.glob(glob_pat):
        d = json.load(open(p)); m = short(d.get("model"))
        if any(x in m.lower() for x in excl):
            continue
        for r in d.get("rows", []):
            if "error" in r or r.get("task") not in topics:
                continue
            cond = r.get("condition")
            if cond in ("original", "perturbed_question", "perturbed_options"):
                by[m][(r["task"], pkey(r))][cond] = r
    return by


def main(a):
    topics = topics_of(a)
    by = _load(a.glob, exclude_of(a), topics)
    rows_out = []
    for m in order_models(by):
        for t in topics:
            for cond, clabel in PERTS:
                b = c = n = 0
                oa = pa = 0
                for (tt, _), cells in by[m].items():
                    if tt != t or "original" not in cells or cond not in cells:
                        continue
                    orr, pr = cells["original"], cells[cond]
                    if cond == "perturbed_question" and \
                       (orr.get("question") or "").strip() == (pr.get("question") or "").strip():
                        continue                               # wording did not change -> skip
                    gt = gtof(orr)
                    o_ok = int(out_pred(orr) == gt); p_ok = int(out_pred(pr) == gt)
                    n += 1; oa += o_ok; pa += p_ok
                    if o_ok and not p_ok: b += 1               # C->W
                    elif p_ok and not o_ok: c += 1             # W->C
                if n == 0:
                    continue
                p_val = mcnemar_exact(b, c)
                rows_out.append({
                    "model": m, "topic": t, "perturbation": clabel, "n": n,
                    "orig_acc": round(100 * oa / n, 1), "pert_acc": round(100 * pa / n, 1),
                    "net": round(100 * (pa - oa) / n, 1),
                    "C_to_W": b, "W_to_C": c, "p": round(p_val, 4),
                    "sig": "*" if p_val < 0.05 else "",
                })
    for r in rows_out:
        print(f"  {r['model']:22s} {TASK_LABEL.get(r['topic'], r['topic']):12s} {r['perturbation']:9s} "
              f"n={r['n']:5d}  orig={r['orig_acc']:5.1f}  pert={r['pert_acc']:5.1f}  "
              f"net={r['net']:+5.1f}  C→W={r['C_to_W']:4d} W→C={r['W_to_C']:4d}  p={r['p']:.4f}{r['sig']}")
    if a.out and rows_out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            w.writeheader(); w.writerows(rows_out)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="output_*/results.json")
    ap.add_argument("--topics", default="birads,acr")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--out", default="mcnemar.csv")
    main(ap.parse_args())
