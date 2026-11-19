"""
Answer-reasoning coherence (condition 'original'). Subcommands:

  gap       answer accuracy vs reasoning accuracy on the SAME parseable items, with Δ = reason - answer
  taxonomy  the answer x reasoning coherence buckets A/B/C/D (C = disconnect: answer wrong, reasoning right)

  python -m analysis.coherence gap      --glob "output_*/results.json" --out fig_ar_gap.png
  python -m analysis.coherence taxonomy --glob "output_*/results.json" --out fig_ar_taxonomy.png
"""
import argparse, os, csv
import numpy as np

from analysis.shared import (plt, MODEL_ORDER, TASK_LABEL, order_models, load_by_model, out_pred,
                             gtof, short, add_std_args, topics_of, exclude_of, save)
from signals.reasoning_parser import expl_answer, semantic_answer, NOTA

C_ANS, C_REA = "#C0504D", "#1D9E75"
CATS = ["answer ✓ & reasoning ✓", "answer ✓ & reasoning ✗",
        "answer ✗ & reasoning ✓  (disconnect)", "answer ✗ & reasoning ✗"]
CAT_COL = {CATS[0]: "#1D9E75", CATS[1]: "#9DC3E6", CATS[2]: "#E8913A", CATS[3]: "#B22222"}


def reason_by(expl, task, opts, method):
    o = expl_answer(expl, task, opts); s = semantic_answer(expl, task, opts)
    if method == "label": return o
    if method == "semantic": return s
    return s if s == NOTA else (o if o is not None else s)     # combined


def _load(a):
    by = load_by_model(a.glob, exclude_of(a), topics_of(a), conditions={a.condition})
    return by, order_models(by), topics_of(a)


# ------------------------------------------------------------------ gap
def cmd_gap(a):
    from signals.answer_mapping import is_tie
    by, models, topics = _load(a)
    res = {t: {} for t in topics}; rows_out = []
    for m in models:
        for t in topics:
            rows = [r for r in by[m] if r.get("task") == t]
            if a.discard_ties:
                rows = [r for r in rows if not is_tie(r.get("scores") or [], a.tau)]
            if not rows: continue
            ans_ok, rea_ok = [], []
            for r in rows:
                gt = gtof(r)
                rp = reason_by(r.get("explanation"), t, r.get("options") or [], a.reasoning)
                if rp is None: continue
                ans_ok.append(int(out_pred(r, a.tau) == gt)); rea_ok.append(int(rp == gt))
            if not rea_ok: continue
            aacc = 100*np.mean(ans_ok); racc = 100*np.mean(rea_ok)
            res[t][m] = (aacc, racc, racc-aacc, len(rea_ok), len(rows))
            rows_out.append({"model": short(m), "topic": t, "n_parsed": len(rea_ok), "N": len(rows),
                             "answer_acc": round(aacc, 1), "reason_acc": round(racc, 1),
                             "delta": round(racc-aacc, 1)})
    _dump_csv(a.csv, rows_out, ["model", "topic", "n_parsed", "N", "answer_acc", "reason_acc", "delta"])
    for r in rows_out:
        print(f"  {r['model']:22s} {TASK_LABEL.get(r['topic'],r['topic']):12s} "
              f"ans={r['answer_acc']:5.1f}  reason={r['reason_acc']:5.1f}  Δ={r['delta']:+5.1f}  "
              f"(n_parsed={r['n_parsed']}/{r['N']})")
    fig, axes = plt.subplots(1, len(topics), figsize=(5.8*len(topics), 5), squeeze=False, sharey=True)
    for ax, t in zip(axes[0], topics):
        ms = [m for m in models if m in res[t]]; x = np.arange(len(ms)); w = 0.38
        aa = [res[t][m][0] for m in ms]; ra = [res[t][m][1] for m in ms]
        ax.bar(x-w/2, aa, w, color=C_ANS, label="Answer accuracy")
        ax.bar(x+w/2, ra, w, color=C_REA, label="Reasoning accuracy")
        for xi, m in enumerate(ms):
            d = res[t][m][2]
            ax.annotate(f"Δ{d:+.0f}", (xi, max(aa[xi], ra[xi])+2), ha="center", fontsize=9,
                        color=("#1a7a4a" if d > 0 else "#a33"), fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{short(m)}\n(n={res[t][m][3]})" for m in ms], rotation=15, ha="center", fontsize=8)
        ax.set_ylim(0, 105); ax.set_title(TASK_LABEL.get(t, t), fontsize=12); ax.grid(axis="y", alpha=.25)
    axes[0][0].set_ylabel("accuracy % (on items where reasoning is parseable)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=10, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Answer vs reasoning accuracy on the same items (Δ = reasoning − answer; "
                 f"{a.reasoning} parser)", y=1.03, fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 1]); save(fig, a.out)


# ------------------------------------------------------------------ taxonomy
def cmd_taxonomy(a):
    from signals.answer_mapping import is_tie
    by, models, topics = _load(a)
    frac = {t: {} for t in topics}; rows_out = []
    for m in models:
        for t in topics:
            rows = [r for r in by[m] if r.get("task") == t]
            if a.discard_ties:
                rows = [r for r in rows if not is_tie(r.get("scores") or [], a.tau)]
            c = {k: 0 for k in CATS}; n = 0
            for r in rows:
                gt = gtof(r)
                rp = reason_by(r.get("explanation"), t, r.get("options") or [], "combined")
                if rp is None: continue
                ac = out_pred(r, a.tau) == gt; rc = rp == gt; n += 1
                c[CATS[0 if (ac and rc) else 1 if ac else 2 if rc else 3]] += 1
            if not n: continue
            frac[t][m] = {k: 100*c[k]/n for k in CATS}
            rows_out.append({"model": short(m), "topic": t, "n_parsed": n,
                             **{["A_ans_reason", "B_ans_only", "C_reason_only_DISCONNECT",
                                 "D_both_wrong"][i]: round(100*c[CATS[i]]/n, 1) for i in range(4)}})
    _dump_csv(a.csv, rows_out, ["model", "topic", "n_parsed", "A_ans_reason", "B_ans_only",
                                "C_reason_only_DISCONNECT", "D_both_wrong"])
    for r in rows_out:
        print(f"  {r['model']:22s} {TASK_LABEL.get(r['topic'],r['topic']):12s} "
              f"A={r['A_ans_reason']:5.1f} B={r['B_ans_only']:5.1f} "
              f"C(disc)={r['C_reason_only_DISCONNECT']:5.1f} D={r['D_both_wrong']:5.1f}  (n={r['n_parsed']})")
    fig, axes = plt.subplots(1, len(topics), figsize=(5.6*len(topics), 5.2), squeeze=False, sharey=True)
    for ax, t in zip(axes[0], topics):
        ms = [m for m in models if m in frac[t]]; x = np.arange(len(ms)); bottom = np.zeros(len(ms))
        for k in CATS:
            vals = np.array([frac[t][m][k] for m in ms])
            ax.bar(x, vals, bottom=bottom, color=CAT_COL[k], label=k, edgecolor="white", width=0.7)
            if k == CATS[2]:
                for xi, v in zip(x, vals):
                    if v >= 3:
                        ax.text(xi, bottom[xi]+v/2, f"{v:.0f}", ha="center", va="center",
                                fontsize=8, color="#5a3", fontweight="bold")
            bottom += vals
        ax.set_xticks(x); ax.set_xticklabels([short(m) for m in ms], rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 100); ax.set_title(TASK_LABEL.get(t, t), fontsize=12)
    axes[0][0].set_ylabel("% of items (with parseable reasoning)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=9, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Answer × reasoning coherence (orange = wrong answer but reasoning names the correct one)",
                 y=1.02, fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 1]); save(fig, a.out)


def _dump_csv(path, rows, fields):
    if not path or not rows: return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = add_std_args(sub.add_parser("gap"))
    g.add_argument("--condition", default="original")
    g.add_argument("--reasoning", default="combined", choices=["label", "semantic", "combined"])
    g.add_argument("--out", default="fig_ar_gap.png"); g.add_argument("--csv", default="ar_gap.csv")
    g.set_defaults(func=cmd_gap)
    x = add_std_args(sub.add_parser("taxonomy"))
    x.add_argument("--condition", default="original")
    x.add_argument("--out", default="fig_ar_taxonomy.png"); x.add_argument("--csv", default="ar_taxonomy.csv")
    x.set_defaults(func=cmd_taxonomy)
    a = ap.parse_args(); a.func(a)


if __name__ == "__main__":
    main()
