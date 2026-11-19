"""
None-of-the-above (NOTA) selection analysis. Subcommands:

  rate       NOTA-selection rate across all conditions (fillers counted; --strict for exact NOTA only)
  rate_conf  NOTA rate AND the confidence placed on NOTA picks (deliberate vs fallback abstention)
  junk       NOTA rate on the real image vs junk images, with the discrimination gap Δ = junk - original
  delta      Δ_NOTA ranking: junk accuracy (NOTA correct) - base accuracy (real answer correct)

  python -m analysis.nota rate   --glob "..." --out fig_nota_rate.png
  python -m analysis.nota junk   --glob "..." --out fig_nota_junk.png
  python -m analysis.nota delta  --glob "..." --out fig_delta_nota.png
"""
import argparse, csv
from collections import defaultdict
import numpy as np

from analysis.shared import (plt, MODEL_COL, MODEL_MARKER, TASK_LABEL, COND, CLAB, JUNK, ABSTAIN,
                             ABSTAIN_LIKE, order_models, load_by_model, out_pred, gtof, short,
                             add_std_args, topics_of, exclude_of, save)
from signals.answer_mapping import is_tie


def _conds(a):
    return [c for c in COND if c in a.conditions.split(",")] if getattr(a, "conditions", "") else COND


def _csv(path, rows, fields):
    if not path or not rows: return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"wrote {path}")


# ------------------------------------------------------------------ rate
def cmd_rate(a):
    topics = topics_of(a); conds = _conds(a)
    nota_set = {ABSTAIN} if a.strict else ABSTAIN_LIKE
    by = load_by_model(a.glob, exclude_of(a), topics, set(conds))
    D = defaultdict(lambda: defaultdict(lambda: defaultdict(list))); n_tie = 0
    for m, rows in by.items():
        for r in rows:
            if a.discard_ties and is_tie(r.get("scores") or [], a.tau): n_tie += 1; continue
            D[m][r["task"]][r["condition"]].append(int(out_pred(r, a.tau) in nota_set))
    if a.discard_ties: print(f"discarded {n_tie} positive-tie items")
    models = order_models(D); x = np.arange(len(conds)); rows_out = []
    fig, axes = plt.subplots(1, len(topics), figsize=(5.8*len(topics), 4.8), squeeze=False, sharey=True)
    for ax, t in zip(axes[0], topics):
        for i, c in enumerate(conds):
            if c in JUNK: ax.axvspan(i-0.5, i+0.5, color="#eaf6ee", zorder=0)
        for m in models:
            ys = [100*np.mean(D[m][t][c]) if D[m][t][c] else np.nan for c in conds]
            ax.plot(x, ys, ls="-", marker=MODEL_MARKER.get(m, "o"), ms=6, lw=2.2,
                    color=MODEL_COL.get(m), label=short(m), zorder=3)
            for c in conds:
                rows_out.append({"model": short(m), "topic": t, "condition": c,
                                 "nota_rate": (round(100*np.mean(D[m][t][c]), 1) if D[m][t][c] else ""),
                                 "n": len(D[m][t][c])})
        ax.set_xticks(x); ax.set_xticklabels([CLAB.get(c, c) for c in conds], fontsize=8)
        ax.set_ylim(0, 105); ax.set_title(TASK_LABEL.get(t, t), fontsize=12); ax.grid(axis="y", alpha=.25)
    lab = "exact NOTA" if a.strict else "NOTA + abstain-like fillers"
    axes[0][0].set_ylabel(f"NOTA selection rate (%)\n[{lab}]")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=9, loc="lower center", ncol=len(models), bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("NOTA selection rate across conditions (green band = junk, abstaining correct)", y=1.03, fontsize=12)
    _csv(a.csv, rows_out, ["model", "topic", "condition", "nota_rate", "n"])
    fig.tight_layout(rect=[0, 0.04, 1, 1]); save(fig, a.out)


# ------------------------------------------------------------------ rate_conf
def cmd_rate_conf(a):
    topics = topics_of(a); conds = _conds(a)
    nota_set = {ABSTAIN} if a.strict else ABSTAIN_LIKE
    by = load_by_model(a.glob, exclude_of(a), topics, set(conds))
    D = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"nota": [], "conf": []}))); n_tie = 0
    for m, rows in by.items():
        for r in rows:
            if a.discard_ties and is_tie(r.get("scores") or [], a.tau): n_tie += 1; continue
            is_nota = int(out_pred(r, a.tau) in nota_set)
            cell = D[m][r["task"]][r["condition"]]; cell["nota"].append(is_nota)
            if is_nota and r.get("confidence") is not None: cell["conf"].append(r["confidence"])
    if a.discard_ties: print(f"discarded {n_tie} positive-tie items")
    models = order_models(D); x = np.arange(len(conds)); rows_out = []
    fig, axes = plt.subplots(len(topics), 2, figsize=(11.5, 4.2*len(topics)), squeeze=False)
    for ri, t in enumerate(topics):
        for ci, (key, ttl, ylab) in enumerate(
                [("nota", "NOTA selection rate", "NOTA rate (%)"),
                 ("conf", "Confidence WHEN choosing NOTA", "mean confidence of NOTA picks (0–100)")]):
            ax = axes[ri][ci]
            for i, c in enumerate(conds):
                if c in JUNK: ax.axvspan(i-0.5, i+0.5, color="#eaf6ee", zorder=0)
            for m in models:
                if key == "nota":
                    ys = [100*np.mean(D[m][t][c]["nota"]) if D[m][t][c]["nota"] else np.nan for c in conds]
                else:
                    ys = [np.mean(D[m][t][c]["conf"]) if D[m][t][c]["conf"] else np.nan for c in conds]
                ax.plot(x, ys, ls="-", marker=MODEL_MARKER.get(m, "o"), ms=6, lw=2.2,
                        color=MODEL_COL.get(m), label=short(m), zorder=3)
            ax.set_xticks(x); ax.set_xticklabels([CLAB.get(c, c) for c in conds], fontsize=8)
            ax.set_ylim(0, 105); ax.grid(alpha=.25)
            if ri == 0: ax.set_title(ttl, fontsize=11)
            ax.set_ylabel((f"{TASK_LABEL.get(t, t)}\n" if ci == 0 else "") + ylab, fontsize=9)
        for c in conds:
            for m in models:
                cell = D[m][t][c]
                rows_out.append({"model": short(m), "topic": t, "condition": c,
                                 "nota_rate": (round(100*np.mean(cell["nota"]), 1) if cell["nota"] else ""),
                                 "conf_when_nota": (round(np.mean(cell["conf"]), 1) if cell["conf"] else "")})
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=9, loc="lower center", ncol=len(models), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("NOTA: how OFTEN it is chosen (left) and how CONFIDENT the model is when choosing it (right)",
                 y=1.01, fontsize=12)
    _csv(a.csv, rows_out, ["model", "topic", "condition", "nota_rate", "conf_when_nota"])
    fig.tight_layout(rect=[0, 0.04, 1, 1]); save(fig, a.out)


# ------------------------------------------------------------------ junk
def cmd_junk(a):
    topics = topics_of(a); junk = ["unrelated", "blank", "noimg"]
    by = load_by_model(a.glob, exclude_of(a), topics, set(["original"]+junk))
    D = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for m, rows in by.items():
        for r in rows:
            D[m][r["task"]][r["condition"]].append(int(out_pred(r, a.tau) == ABSTAIN))
    models = order_models(D); rows_out = []; x = np.arange(len(models)); w = 0.38
    fig, axes = plt.subplots(1, len(topics), figsize=(5.6*len(topics), 4.8), squeeze=False, sharey=True)
    for ax, t in zip(axes[0], topics):
        def rate(m, c): return 100*np.mean(D[m][t][c]) if D[m][t][c] else np.nan
        orig = [rate(m, "original") for m in models]
        jk = [np.nanmean([rate(m, c) for c in junk]) for m in models]
        ax.bar(x-w/2, orig, w, color="#C0504D", label="real image (abstain = WRONG)")
        ax.bar(x+w/2, jk, w, color="#1D9E75", label="junk avg (abstain = CORRECT)")
        for xi, m in enumerate(models):
            g = jk[xi]-orig[xi]
            if g == g:
                ax.annotate(f"Δ{g:+.0f}", (xi, max(orig[xi], jk[xi])+2), ha="center", fontsize=9,
                            fontweight="bold", color="#333")
            rows_out.append({"model": short(m), "topic": t,
                             "original": round(orig[xi], 1) if orig[xi] == orig[xi] else "",
                             "junk_avg": round(jk[xi], 1) if jk[xi] == jk[xi] else "",
                             "gap_junk_minus_original": round(g, 1) if g == g else ""})
        ax.set_xticks(x); ax.set_xticklabels([short(m) for m in models], rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 108); ax.set_title(TASK_LABEL.get(t, t), fontsize=12); ax.grid(axis="y", alpha=.25)
    axes[0][0].set_ylabel("NOTA selection rate (%)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=9, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Abstention on the real image vs junk (Δ = junk − original; big Δ = discriminates)",
                 y=1.03, fontsize=12)
    _csv(a.csv, rows_out, ["model", "topic", "original", "junk_avg", "gap_junk_minus_original"])
    fig.tight_layout(rect=[0, 0.04, 1, 1]); save(fig, a.out)


# ------------------------------------------------------------------ delta
def cmd_delta(a):
    topics = topics_of(a); junk = [c.strip() for c in a.junk.split(",") if c.strip()]
    nota_set = {ABSTAIN} if a.strict else ABSTAIN_LIKE
    by = load_by_model(a.glob, exclude_of(a), topics)
    D = defaultdict(lambda: defaultdict(lambda: {"base": [], "nota": []})); n_tie = 0
    for m, rows in by.items():
        for r in rows:
            t = r.get("task"); c = r.get("condition")
            if a.discard_ties and is_tie(r.get("scores") or [], a.tau): n_tie += 1; continue
            pred = out_pred(r, a.tau)
            if c == a.base: D[m][t]["base"].append(int(pred == gtof(r)))
            elif c in junk: D[m][t]["nota"].append(int(pred in nota_set))
    if a.discard_ties: print(f"discarded {n_tie} positive-tie items")
    C_RED, C_GRAY, C_GREEN = "#F1808A", "#B7B7B7", "#1E7A3D"; rows_out = []
    fig, axes = plt.subplots(1, len(topics), figsize=(6.6*len(topics), 5.4), squeeze=False)
    for ax, t in zip(axes[0], topics):
        recs = []
        for m in D:
            b, nt = D[m][t]["base"], D[m][t]["nota"]
            if not b or not nt: continue
            base = 100*np.mean(b); nota = 100*np.mean(nt)
            recs.append((short(m), base, nota, nota-base, len(b), len(nt)))
        recs.sort(key=lambda r: r[3])
        for name, base, nota, dlt, nb, nn in recs:
            rows_out.append({"model": name, "topic": t, "base_acc": round(base, 1),
                             "nota_acc": round(nota, 1), "delta_nota": round(dlt, 1), "n_base": nb, "n_junk": nn})
        names = [r[0] for r in recs]; deltas = [r[3] for r in recs]; y = np.arange(len(recs))[::-1]
        cols = [C_RED if d < -a.band else (C_GREEN if d > a.band else C_GRAY) for d in deltas]
        ax.barh(y, deltas, color=cols, edgecolor="white", height=0.72); ax.axvline(0, color="#333", lw=1)
        for yi, d in zip(y, deltas):
            ax.text(d + (1.2 if d >= 0 else -1.2), yi, f"{d:+.0f}", va="center",
                    ha="left" if d >= 0 else "right", fontsize=8.5)
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
        mx = max((abs(d) for d in deltas), default=10)*1.25 + 4; ax.set_xlim(-mx, mx)
        ax.set_xlabel(r"$\Delta_{\mathrm{NOTA}}$ (pp) = NOTA acc − Base acc")
        ax.set_title(TASK_LABEL.get(t, t), fontsize=12); ax.grid(axis="x", alpha=.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_RED, C_GRAY, C_GREEN)]
    fig.legend(handles, ["Overconfident (Δ<0)", "Balanced (|Δ|≤%d)" % a.band, "Abstention-dom. (Δ>0)"],
               fontsize=9, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(r"$\Delta_{\mathrm{NOTA}}$: NOTA accuracy (junk) − Base accuracy (original)", y=1.02, fontsize=12)
    for r in rows_out:
        print(f"  {r['model']:22s} {TASK_LABEL.get(r['topic'],r['topic']):12s} "
              f"base={r['base_acc']:5.1f}  nota={r['nota_acc']:5.1f}  Δ={r['delta_nota']:+6.1f}")
    _csv(a.csv, rows_out, ["model", "topic", "base_acc", "nota_acc", "delta_nota", "n_base", "n_junk"])
    fig.tight_layout(rect=[0, 0.05, 1, 1]); save(fig, a.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = add_std_args(sub.add_parser("rate"))
    r.add_argument("--conditions", default=""); r.add_argument("--strict", action="store_true")
    r.add_argument("--out", default="fig_nota_rate.png"); r.add_argument("--csv", default="nota_rate.csv")
    r.set_defaults(func=cmd_rate)
    rc = add_std_args(sub.add_parser("rate_conf"))
    rc.add_argument("--conditions", default=""); rc.add_argument("--strict", action="store_true")
    rc.add_argument("--out", default="fig_nota_rate_conf.png"); rc.add_argument("--csv", default="nota_rate_conf.csv")
    rc.set_defaults(func=cmd_rate_conf)
    j = add_std_args(sub.add_parser("junk"))
    j.add_argument("--out", default="fig_nota_junk.png"); j.add_argument("--csv", default="nota_junk.csv")
    j.set_defaults(func=cmd_junk)
    d = add_std_args(sub.add_parser("delta"))
    d.add_argument("--base", default="original"); d.add_argument("--junk", default="unrelated,blank,noimg")
    d.add_argument("--band", type=float, default=5.0); d.add_argument("--strict", action="store_true")
    d.add_argument("--out", default="fig_delta_nota.png"); d.add_argument("--csv", default="delta_nota.csv")
    d.set_defaults(func=cmd_delta)
    a = ap.parse_args(); a.func(a)


if __name__ == "__main__":
    main()
