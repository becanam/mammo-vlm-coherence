"""
Compare the SCORE-based prediction (arg-max of the per-option scores) to the answer stated in
the free-text EXPLANATION. Reveals cases where the model reasons correctly but its scores are
wrong (e.g. descending-by-position scores -> arg-max picks option A regardless of the reasoning).

Per model x task (condition 'original' by default) reports:
  score_acc   accuracy of the arg-max score prediction
  expl_acc    accuracy of the answer named in the explanation (over items where one was found)
  found%      fraction of explanations we could extract an answer from
  contra%     of items with both, fraction where explanation answer != score prediction

Also writes a per-row CSV (--out) with score_pred vs expl_pred side by side.

Usage:
  python3 analyze_reasoning_gap.py --glob "output_*/results.json" \
      --condition original --out reasoning_gap_rows.csv --csv reasoning_gap.csv
"""

import json
import re
import glob as glob_mod
import argparse
import csv
from collections import defaultdict
import numpy as np


def short(m):
    return m.split("/")[-1]


def expl_answer(expl, task, options):
    """Extract the option the explanation concludes with (or None)."""
    t = (expl or "").lower()
    if not t:
        return None
    if task in ("birads", "birads012"):
        nums = re.findall(r"bi[\s-]*rads?\b.{0,25}?(\d)", t)       # 'bi-rads ... 1'
        if nums:
            opt = f"Bi-Rads {nums[-1]}"
            return opt if opt in options else None
        return None
    if task == "acr":
        # require a STANDALONE letter (\b on both sides) so "and"->d, "a"->a etc. don't match
        m = re.findall(r"(?:level|category|acr|density)\b.{0,20}?\b([a-d])\b", t)
        if m:
            opt = f"Level {m[-1].upper()}"
            return opt if opt in options else None
        return None
    if task == "laterality":
        cand = [(t.rfind("left"), "Left"), (t.rfind("right"), "Right")]
        cand = [(p, o) for p, o in cand if p >= 0]
        return max(cand)[1] if cand else None
    if task == "view":
        pos = {"MLO": max(t.rfind("mlo"), t.rfind("mediolateral"), t.rfind("oblique")),
               "CC": max(t.rfind("cc"), t.rfind("craniocaudal"), t.rfind("cranio"))}
        pos = {k: v for k, v in pos.items() if v >= 0}
        return max(pos, key=pos.get) if pos else None
    return None


def main(a):
    exclude = [x.lower() for x in a.exclude.split(",")]
    by = defaultdict(list)
    for p in glob_mod.glob(a.glob):
        d = json.load(open(p))
        m = short(d["model"])
        if any(x and x in m.lower() for x in exclude):
            continue
        by[m] += [r for r in d["rows"] if r.get("condition") == a.condition]

    out_rows = []
    hdr = f"{'model':22s} {'task':10s} {'n':>6s} {'score_acc':>9s} {'expl_acc':>8s} {'found%':>7s} {'contra%':>8s}"
    print(hdr)
    summ = []
    for m in sorted(by):
        tasks = sorted({r["task"] for r in by[m]})
        for t in tasks:
            rows = [r for r in by[m] if r["task"] == t]
            n = len(rows)
            sp_correct = [int(r.get("pred_text") == r.get("orig_gt", r.get("gt"))) for r in rows]
            ep, ep_correct, both_contra, found = [], [], [], 0
            for r in rows:
                gt = r.get("orig_gt", r.get("gt"))
                ea = expl_answer(r.get("explanation"), t, r.get("options") or [])
                sp = r.get("pred_text")
                out_rows.append({"model": m, "task": t, "path": r["path"], "gt": gt,
                                 "score_pred": sp, "expl_pred": ea,
                                 "score_correct": int(sp == gt),
                                 "expl_correct": (int(ea == gt) if ea else ""),
                                 "agree": (int(sp == ea) if ea else "")})
                if ea:
                    found += 1
                    ep_correct.append(int(ea == gt))
                    both_contra.append(int(sp != ea))
            score_acc = np.mean(sp_correct) if rows else float("nan")
            expl_acc = np.mean(ep_correct) if ep_correct else float("nan")
            contra = np.mean(both_contra) if both_contra else float("nan")
            print(f"{m:22s} {t:10s} {n:6d} {100*score_acc:9.1f} "
                  f"{(100*expl_acc if ep_correct else float('nan')):8.1f} "
                  f"{100*found/max(n,1):7.1f} {(100*contra if both_contra else float('nan')):8.1f}")
            summ.append({"model": m, "task": t, "n": n,
                         "score_acc": round(float(score_acc), 3),
                         "expl_acc": round(float(expl_acc), 3) if ep_correct else "",
                         "found_pct": round(100 * found / max(n, 1), 1),
                         "contradiction_pct": round(float(100 * contra), 1) if both_contra else ""})

    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["model", "task", "path", "gt", "score_pred",
                                               "expl_pred", "score_correct", "expl_correct", "agree"])
            w.writeheader(); w.writerows(out_rows)
        print(f"\nwrote {a.out} ({len(out_rows)} rows)")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["model", "task", "n", "score_acc", "expl_acc",
                                               "found_pct", "contradiction_pct"])
            w.writeheader(); w.writerows(summ)
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="output_*/results.json")
    ap.add_argument("--condition", default="original")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--out", default="reasoning_gap_rows.csv")
    ap.add_argument("--csv", default="reasoning_gap.csv")
    main(ap.parse_args())
