"""
Validate the reasoning-answer matcher against hand annotation. Two steps:

  # 1. draw a blind, model x task stratified sample to annotate
  python3 parser_validation.py sample --glob "output_*/results.json" \
      --topics birads,acr --n 120 --out parser_sample.csv --key parser_key.csv
  #    -> edit parser_sample.csv, fill the human_label column (see legend), save.

  # 2. score agreement between your labels and the matcher
  python3 parser_validation.py score --annotated parser_sample.csv --key parser_key.csv

The sample CSV shows ONLY the id, task, and explanation (blind to the matcher output and the gold
label, both of which live in the key file), so the annotation is unbiased.

Annotation legend (type into human_label):
  BI-RADS  : a category 0-6         (or 'none' / 'unparseable')
  density  : a category A-D         (or 'none' / 'unparseable')
  none         = the explanation concludes 'none of the above'
  unparseable  = it states no category and describes no definable finding
"""
import json, glob as glob_mod, argparse, csv, random, re
from collections import defaultdict

from .answer_mapping import is_tie
from .reasoning_parser import expl_answer, semantic_answer

NOTA = "None of the above"


def short(m): return (m or "").split("/")[-1]
def gtof(r): return r.get("orig_gt", r.get("gt"))
def pkey(r):
    p = r.get("path"); return "|".join(p) if isinstance(p, list) else str(p)
def reason_pred(r):
    o = expl_answer(r.get("explanation"), r.get("task"), r.get("options") or [])
    s = semantic_answer(r.get("explanation"), r.get("task"), r.get("options") or [])
    return s if s == NOTA else (o if o is not None else s)


def read_rows(path):
    """Robust CSV read: strip a UTF-8 BOM (Excel/LibreOffice) and sniff , ; or tab delimiter."""
    with open(path, encoding="utf-8-sig", newline="") as fh:
        head = fh.read(8192); fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(head, delimiters=",;\t")
        except Exception:
            dialect = csv.excel
        rows = list(csv.DictReader(fh, dialect=dialect))
    for r in rows:                              # defensive: strip stray BOM/space from keys
        for k in list(r.keys()):
            nk = k.lstrip("﻿").strip()
            if nk != k:
                r[nk] = r.pop(k)
    return rows


def canon(v, task):
    """Normalize a matcher/human label to a comparable token."""
    if v is None: return "unparseable"
    s = str(v).strip().lower()
    if s in ("", "unparseable", "unparse", "?", "na", "n/a"): return "unparseable"
    if s in ("none", "nota") or "none of the above" in s: return "none"
    if task == "birads":
        m = re.search(r"[0-6]", s);  return m.group(0) if m else s
    m = re.search(r"\b([a-d])\b", s) or re.search(r"([a-d])", s)   # density A-D
    return m.group(1).upper() if m else s


def do_sample(a):
    topics = [t.strip() for t in a.topics.split(",") if t.strip()]
    by = defaultdict(list)
    for p in glob_mod.glob(a.glob):
        d = json.load(open(p)); m = short(d.get("model"))
        for r in d.get("rows", []):
            if r.get("condition") == "original" and r.get("task") in topics \
               and (r.get("explanation") or "").strip():
                if a.discard_ties and is_tie(r.get("scores") or [], a.tau): continue
                by[(m, r["task"])].append(r)
    cells = sorted(by)
    per = max(1, a.n // max(1, len(cells)))
    rng = random.Random(a.seed)
    sample, key = [], []
    for (m, t) in cells:
        rows = by[(m, t)][:]; rng.shuffle(rows)
        for i, r in enumerate(rows[:per]):
            rid = f"{short(m)}|{t}|{i}"
            sample.append({"id": rid, "task": t, "explanation": (r.get("explanation") or "").strip(),
                           "human_label": ""})
            key.append({"id": rid, "model": short(m), "task": t, "path": pkey(r),
                        "gold": gtof(r), "matcher_pred": reason_pred(r)})
    rng.shuffle(sample)                                    # present in random order
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "task", "explanation", "human_label"])
        w.writeheader(); w.writerows(sample)
    with open(a.key, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "model", "task", "path", "gold", "matcher_pred"])
        w.writeheader(); w.writerows(key)
    print(f"wrote {len(sample)} items to {a.out} (annotate the human_label column)")
    print(f"wrote hidden key to {a.key}")
    print("legend: BI-RADS 0-6 / density A-D / 'none' / 'unparseable'")


def kappa(pairs):
    """Observed agreement and Cohen's κ (κ via scikit-learn)."""
    from sklearn.metrics import cohen_kappa_score
    n = len(pairs)
    if n == 0: return float("nan"), float("nan")
    po = sum(1 for h, m in pairs if h == m) / n
    k = cohen_kappa_score([h for h, _ in pairs], [m for _, m in pairs])
    return po, k


def do_score(a):
    key = {row["id"]: row for row in read_rows(a.key)}
    pairs, skipped, rows = [], 0, []
    for row in read_rows(a.annotated):
        rid = row["id"]; hv = (row.get("human_label") or "").strip()
        if not hv or rid not in key: skipped += 1; continue
        t = key[rid]["task"]
        h = canon(hv, t); m = canon(key[rid]["matcher_pred"], t)
        pairs.append((h, m)); rows.append((rid, t, h, m))
    po, k = kappa(pairs)
    print(f"annotated & scored: {len(pairs)}  (skipped {skipped} blank/unmatched)")
    print(f"exact agreement: {100*po:.1f}%")
    print(f"Cohen's kappa  : {k:.3f}")
    # per-task breakdown
    for t in sorted({r[1] for r in rows}):
        sub = [(h, m) for (_, tt, h, m) in rows if tt == t]
        p2, k2 = kappa(sub)
        print(f"  [{t}] n={len(sub)}  agree={100*p2:.1f}%  kappa={k2:.3f}")
    if a.show_disagree:
        print("\ndisagreements (human vs matcher):")
        for rid, t, h, m in rows:
            if h != m: print(f"  {rid:34s} human={h!r:14s} matcher={m!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--glob", default="output_*/results.json")
    s.add_argument("--topics", default="birads,acr")
    s.add_argument("--n", type=int, default=120)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--tau", type=float, default=1.0)
    s.add_argument("--discard-ties", dest="discard_ties", action="store_true", default=True)
    s.add_argument("--keep-ties", dest="discard_ties", action="store_false")
    s.add_argument("--out", default="parser_sample.csv")
    s.add_argument("--key", default="parser_key.csv")
    s.set_defaults(func=do_sample)
    c = sub.add_parser("score")
    c.add_argument("--annotated", default="parser_sample.csv")
    c.add_argument("--key", default="parser_key.csv")
    c.add_argument("--show-disagree", action="store_true")
    c.set_defaults(func=do_score)
    a = ap.parse_args(); a.func(a)
