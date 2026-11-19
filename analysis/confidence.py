"""
Confidence signals across the evidence spectrum. Subcommands:

  gt          confidence on the CORRECT option (solid) vs the CHOSEN option (dashed guide)
  reasoning   declared numeric confidence vs reasoning-derived uncertainty
              (--subset committed | abstained | all)
  behavioral  a perturbation-stability confidence (answer flips under option shuffle) vs the stated
              confidence, scored by AUROC / ECE and acc|stable vs acc|flipped

  python -m analysis.confidence gd         --glob "..." --out fig_gt_confidence.png
  python -m analysis.confidence reasoning  --glob "..." --subset committed --out fig_conf_vs_reasoning.png
  python -m analysis.confidence behavioral --glob "..." --out fig_behavioral_conf.png
"""
import argparse, csv
import numpy as np
from matplotlib.lines import Line2D

from analysis.shared import (plt, MODEL_ORDER, MODEL_COL, MODEL_MARKER, TASK_LABEL, order_models,
                             load_by_model, out_pred, gtof, pkey, short, stated_conf, ABSTAIN,
                             ABSTAIN_LIKE, add_std_args, topics_of, exclude_of, save)
from signals.answer_mapping import is_tie
from signals.reasoning_confidence import reasoning_uncertainty
from signals.reasoning_parser import reason_pred as reason_pred_row
from matplotlib.colors import LinearSegmentedColormap

CLAB = {"original": "real\nimage", "perturbed_question": "reworded\nQ", "perturbed_options": "shuffled\nopts",
        "unrelated": "unrelated\nphoto", "blank": "blank", "noimg": "no\nimage"}


# ============================================================ gt vs chosen confidence
def gt_conf(r):
    gt = gtof(r); opts = r.get("options") or []; sc = r.get("scores") or []
    if gt in opts:
        i = opts.index(gt)
        if i < len(sc) and sc[i] is not None: return float(sc[i])
    return None


def cmd_gt(a):
    conds = [c.strip() for c in a.conditions.split(",") if c.strip()]
    topics = topics_of(a)
    by = load_by_model(a.glob, exclude_of(a), topics, set(conds))
    from collections import defaultdict
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    chosen = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    n_tie = 0
    for m, rows in by.items():
        for r in rows:
            if a.discard_ties and is_tie(r.get("scores") or [], a.tau): n_tie += 1; continue
            v = gt_conf(r)
            if v is not None: data[m][r["task"]][r["condition"]].append(v)
            cv = stated_conf(r)
            if cv is not None: chosen[m][r["task"]][r["condition"]].append(cv)
    if a.discard_ties: print(f"discarded {n_tie} positive-tie items")
    models = order_models(data)
    if not models: print("no matching rows"); return
    rows_out = []; x = np.arange(len(conds))
    fig, axes = plt.subplots(1, len(topics), figsize=(5.6*len(topics), 4.7), squeeze=False, sharey=True)
    for ax, t in zip(axes[0], topics):
        for m in models:
            ys = [np.mean(data[m][t][c]) if data[m][t][c] else np.nan for c in conds]
            yc = [np.mean(chosen[m][t][c]) if chosen[m][t][c] else np.nan for c in conds]
            mk = MODEL_MARKER.get(m, "o")
            ax.plot(x, ys, ls="-", marker=mk, ms=7, lw=2.2, color=MODEL_COL.get(m), label=short(m))
            ax.plot(x, yc, ls="--", marker=mk, ms=6, lw=1.5, mfc="none", color=MODEL_COL.get(m), alpha=0.7)
            for c, y, cc in zip(conds, ys, yc):
                rows_out.append({"model": short(m), "topic": t, "condition": c,
                                 "mean_gt_confidence": (round(y, 1) if y == y else ""),
                                 "mean_chosen_confidence": (round(cc, 1) if cc == cc else "")})
        ax.set_xticks(x); ax.set_xticklabels([CLAB.get(c, c) for c in conds], fontsize=8)
        ax.set_ylim(0, 105); ax.set_title(TASK_LABEL.get(t, t), fontsize=12)
        ax.grid(alpha=.25); ax.set_xlabel("evidence removed  →", fontsize=9)
    axes[0][0].set_ylabel("mean confidence (0–100)")
    style = [Line2D([0], [0], color="#444", lw=2.2, ls="-", marker="o", label="correct answer (GT), filled"),
             Line2D([0], [0], color="#444", lw=1.5, ls="--", marker="o", mfc="none", alpha=0.8,
                    label="chosen answer (guide), hollow")]
    h, l = axes[0][0].get_legend_handles_labels()
    leg1 = fig.legend(h, l, fontsize=9, loc="lower center", ncol=len(models), bbox_to_anchor=(0.5, -0.04))
    fig.add_artist(leg1)
    fig.legend(handles=style, fontsize=8.5, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("Confidence on the CORRECT answer (solid) vs the CHOSEN answer (dashed guide)", y=1.03, fontsize=12)
    _csv(a.csv, rows_out, ["model", "topic", "condition", "mean_gt_confidence", "mean_chosen_confidence"])
    fig.tight_layout(rect=[0, 0.04, 1, 1]); save(fig, a.out)


# ============================================================ conf vs reasoning uncertainty
def cmd_reasoning(a):
    conds = [c.strip() for c in a.conditions.split(",") if c.strip()]
    topics = topics_of(a)
    by = load_by_model(a.glob, exclude_of(a), topics, set(conds))
    from collections import defaultdict
    D = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"conf": [], "unc": []})))
    n_tie = n_excl = 0
    for m, rows in by.items():
        for r in rows:
            if a.discard_ties and is_tie(r.get("scores") or [], a.tau): n_tie += 1; continue
            op = out_pred(r, a.tau)
            if a.subset == "committed" and op in ABSTAIN_LIKE: n_excl += 1; continue
            if a.subset == "abstained" and op not in ABSTAIN_LIKE: n_excl += 1; continue
            cell = D[m][r["task"]][r["condition"]]
            if r.get("confidence") is not None: cell["conf"].append(r["confidence"])
            u = reasoning_uncertainty(r.get("explanation"))
            if u is not None: cell["unc"].append(u)
    if a.discard_ties: print(f"discarded {n_tie} positive-tie items")
    if a.subset != "all": print(f"subset={a.subset}: excluded {n_excl} items not in subset")
    models = order_models(D)
    conf_title = "Confidence in the NOTA choice" if a.subset == "abstained" else "Numeric confidence (declared)"
    conf_ylab = "NOTA-selection conf 0–100" if a.subset == "abstained" else "confidence 0–100"
    x = np.arange(len(conds))
    fig, axes = plt.subplots(len(topics), 2, figsize=(11, 4.2*len(topics)), squeeze=False)
    for ri, t in enumerate(topics):
        for ci, (key, ttl, ylab) in enumerate([("conf", conf_title, conf_ylab),
                                                ("unc", "Reasoning uncertainty (how unsure it sounds)",
                                                 "uncertainty 0–100")]):
            ax = axes[ri][ci]
            for m in models:
                ys = [np.mean(D[m][t][c][key]) if D[m][t][c][key] else np.nan for c in conds]
                ax.plot(x, ys, ls="-", marker=MODEL_MARKER.get(m, "o"), ms=6, lw=2.2,
                        color=MODEL_COL.get(m), label=short(m))
            ax.set_xticks(x); ax.set_xticklabels([CLAB.get(c, c) for c in conds], fontsize=8)
            ax.set_ylim(0, 105); ax.grid(alpha=.25)
            if ri == 0: ax.set_title(ttl, fontsize=11)
            ax.set_ylabel((f"{TASK_LABEL.get(t, t)}\n" if ci == 0 else "") + ylab, fontsize=(10 if ci == 0 else 9))
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=9, loc="lower center", ncol=len(models), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Declared confidence vs reasoning uncertainty (evidence removed →; subset={a.subset})",
                 y=1.01, fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 1]); save(fig, a.out)


# ============================================================ behavioral confidence
def auroc(scores, y):
    s = np.asarray(scores, float); y = np.asarray(y, int)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    order = s.argsort(); rs = s[order]; ranks = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j < len(s) and rs[j] == rs[i]: j += 1
        ranks[i:j] = (i + j - 1) / 2.0 + 1; i = j
    rf = np.empty(len(s)); rf[order] = ranks
    return (rf[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ece(conf, correct, bins=10):
    conf = np.asarray(conf, float); correct = np.asarray(correct, int)
    if len(conf) == 0: return float("nan")
    edges = np.linspace(0, 1, bins + 1); e = 0.0; n = len(conf)
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        m = (conf >= lo) & (conf <= hi if b == bins - 1 else conf < hi)
        if m.sum(): e += abs(correct[m].mean() - conf[m].mean()) * m.sum() / n
    return e


def hmean(a, b, eps=1e-6): return 2 * a * b / (a + b + eps)
def stated01(r): return min(max((stated_conf(r) or 0.0) / 100.0, 0.0), 1.0)


def cmd_behavioral(a):
    from collections import defaultdict
    topics = topics_of(a); excl = [x.lower() for x in exclude_of(a)]
    import json, glob as glob_mod
    orig, pert = {}, {}
    for p in glob_mod.glob(a.glob):
        d = json.load(open(p)); m = short(d.get("model"))
        if any(x in m.lower() for x in excl): continue
        for r in d.get("rows", []):
            if r.get("task") not in topics or "error" in r: continue
            k = (m, r["task"], pkey(r))
            if r.get("condition") == "original": orig[k] = r
            elif r.get("condition") == "perturbed_options": pert[k] = r
    agg = defaultdict(lambda: {"stated": [], "behav": [], "comb": [], "correct": []})
    n_tie = n_abst = 0
    for k, orr in orig.items():
        pr = pert.get(k)
        if pr is None: continue
        if a.discard_ties and (is_tie(orr.get("scores") or [], a.tau) or is_tie(pr.get("scores") or [], a.tau)):
            n_tie += 1; continue
        op = out_pred(orr, a.tau)
        if op == ABSTAIN: n_abst += 1; continue
        pp = out_pred(pr, a.tau); behav = int(op == pp); st = stated01(orr)
        g = agg[(k[0], k[1])]
        g["stated"].append(st); g["behav"].append(behav)
        g["comb"].append(hmean(st, behav)); g["correct"].append(int(op == gtof(orr)))
    print(f"discarded {n_tie} tie pairs; excluded {n_abst} abstained-original items")
    models = [m for m in MODEL_ORDER if any((m, t) in agg for t in topics)] + \
             sorted({k[0] for k in agg} - set(MODEL_ORDER))
    rows_out = []
    for m in models:
        for t in topics:
            g = agg.get((m, t))
            if not g or not g["correct"]: continue
            st = np.array(g["stated"]); bv = np.array(g["behav"]); cr = np.array(g["correct"]); cb = np.array(g["comb"])
            acc_stable = 100*cr[bv == 1].mean() if (bv == 1).any() else float("nan")
            acc_flip = 100*cr[bv == 0].mean() if (bv == 0).any() else float("nan")
            rows_out.append({"model": m, "topic": t, "n": len(cr), "flip_pct": round(100*(bv == 0).mean(), 1),
                             "acc_stable": round(acc_stable, 1), "acc_flipped": round(acc_flip, 1),
                             "gap": round(acc_stable-acc_flip, 1) if acc_flip == acc_flip else "",
                             "auroc_stated": round(auroc(st, cr), 3), "auroc_behavioral": round(auroc(bv, cr), 3),
                             "auroc_combined": round(auroc(cb, cr), 3),
                             "ece_stated": round(ece(st, cr), 3), "ece_combined": round(ece(cb, cr), 3)})
    for r in rows_out:
        print(f"  {r['model']:22s} {TASK_LABEL.get(r['topic'],r['topic']):12s} n={r['n']:5d} "
              f"flip={r['flip_pct']:5.1f}%  acc|stable={r['acc_stable']!s:>5} acc|flip={r['acc_flipped']!s:>5} "
              f"gap={r['gap']!s:>5}  AUROC stated={r['auroc_stated']} behav={r['auroc_behavioral']}")
    if rows_out: _csv(a.csv, rows_out, list(rows_out[0].keys()))
    if not rows_out: print("no matched original/perturbed_options items"); return
    fig, axes = plt.subplots(1, len(topics), figsize=(6.0*len(topics), 5.2), squeeze=False, sharey=True)
    for ax, t in zip(axes[0], topics):
        sub = [r for r in rows_out if r["topic"] == t]; ms = [r["model"] for r in sub]
        x = np.arange(len(ms)); w = 0.38
        for xi, r in enumerate(sub):
            sv = r["acc_stable"] if r["acc_stable"] == r["acc_stable"] else 0
            fv = r["acc_flipped"] if r["acc_flipped"] == r["acc_flipped"] else 0
            ax.bar(xi-w/2, sv, w, color="#1D9E75", label="acc | stable" if xi == 0 else None)
            ax.bar(xi+w/2, fv, w, color="#C0504D", label="acc | flipped" if xi == 0 else None)
            ax.annotate(f"AUROC\nstated {r['auroc_stated']}\nflip {r['auroc_behavioral']}",
                        (xi, max(sv, fv)+3), ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels([f"{m}\n(n={r['n']})" for m, r in zip(ms, sub)],
                                             rotation=15, ha="center", fontsize=8)
        ax.set_ylim(0, 100); ax.set_title(TASK_LABEL.get(t, t), fontsize=12); ax.grid(axis="y", alpha=.25)
    axes[0][0].set_ylabel("accuracy of the ORIGINAL answer (%)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=10, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Behavioral confidence: acc|stable ≫ acc|flipped ⇒ flipping predicts wrongness", y=1.03, fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 1]); save(fig, a.out)


def _csv(path, rows, fields):
    if not path or not rows: return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"wrote {path}")


# ============================================================ heatmap (confidence-experiment overview)
_HCOND = ["original", "unrelated", "blank", "noimg", "perturbed_question", "perturbed_options"]
_HCLAB = ["original", "unrelated", "blank", "no image", "pert.\nquestion", "pert.\noptions"]
_HCMAP = LinearSegmentedColormap.from_list("cf", ["#2c3e50", "#5d6d7e", "#a9746e", "#e74c3c", "#f5563c"])
_HCMAP.set_bad("#e3e3e3")


def _hblock(by, models, metric, tau, discard_ties):
    M = np.full((len(models), len(_HCOND)), np.nan)
    for i, m in enumerate(models):
        for j, c in enumerate(_HCOND):
            sub = [r for r in by[m] if r.get("condition") == c]
            if discard_ties: sub = [r for r in sub if not is_tie(r.get("scores") or [], tau)]
            if not sub: continue
            if metric == "acc":
                M[i, j] = 100*np.mean([int(out_pred(r, tau) == gtof(r)) for r in sub])
            elif metric == "conf":
                cc = [r["confidence"] for r in sub if r.get("confidence") is not None]
                M[i, j] = np.mean(cc) if cc else np.nan
            elif metric == "abstain":
                M[i, j] = 100*np.mean([int(out_pred(r, tau) in ABSTAIN_LIKE) for r in sub])
            elif metric == "reason_acc":
                vals = [int(rp == gtof(r)) for r in sub
                        for rp in [reason_pred_row(r.get("explanation"), r.get("task"), r.get("options") or [])]
                        if rp is not None]
                M[i, j] = 100*np.mean(vals) if vals else np.nan
            elif metric == "reason_unc":
                us = [u for r in sub for u in [reasoning_uncertainty(r.get("explanation"))] if u is not None]
                M[i, j] = np.mean(us) if us else np.nan
    return M


def cmd_heatmap(a):
    labels = {"acc": "Accuracy [%]", "conf": "Mean confidence", "abstain": "Abstain / filler [%]",
              "filler": "Abstain / filler [%]", "reason_acc": "Reasoning acc [%]",
              "reason_unc": "Reasoning uncertainty"}
    metrics = [(labels[x], "abstain" if x == "filler" else x) for x in a.metrics.split(",") if x in labels]
    by = load_by_model(a.glob, exclude_of(a))            # all topics/conditions
    models = order_models(by); nrow = len(models); im = None
    fig, axes = plt.subplots(len(metrics), 1, figsize=(1.15*len(_HCOND)+3, 0.55*nrow*len(metrics)+2), squeeze=False)
    for (title, metric), ax in zip(metrics, axes[:, 0]):
        M = _hblock(by, models, metric, a.tau, a.discard_ties)
        im = ax.imshow(M, cmap=_HCMAP, vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(_HCOND))); ax.set_xticklabels(_HCLAB, fontsize=8)
        ax.set_yticks(range(nrow)); ax.set_yticklabels(models, fontsize=8)
        ax.set_ylabel(title, fontsize=10, fontweight="bold")
        for i in range(nrow):
            for j in range(len(_HCOND)):
                v = M[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white" if v < 45 else "black")
                else:
                    ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#888")
    fig.colorbar(im, ax=axes[:, 0], label="Score / rate [%]", shrink=0.6, pad=0.02)
    fig.suptitle("Confidence experiment — mean scores per model × condition (explicit answer; fillers as NOTA)",
                 y=1.0, fontsize=11)
    save(fig, a.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = add_std_args(sub.add_parser("gt"))
    g.add_argument("--conditions", default="original,perturbed_question,perturbed_options,unrelated,blank,noimg")
    g.add_argument("--out", default="fig_gt_confidence.png"); g.add_argument("--csv", default="gt_confidence.csv")
    g.set_defaults(func=cmd_gt)
    r = add_std_args(sub.add_parser("reasoning"))
    r.add_argument("--conditions", default="original,unrelated,blank,noimg")
    r.add_argument("--subset", choices=["committed", "abstained", "all"], default="committed")
    r.add_argument("--out", default="fig_conf_vs_reasoning.png")
    r.set_defaults(func=cmd_reasoning)
    b = add_std_args(sub.add_parser("behavioral"))
    b.add_argument("--out", default="fig_behavioral_conf.png"); b.add_argument("--csv", default="behavioral_conf.csv")
    b.set_defaults(func=cmd_behavioral)
    hm = add_std_args(sub.add_parser("heatmap"))
    hm.add_argument("--metrics", default="acc,conf,abstain,reason_acc,reason_unc")
    hm.add_argument("--out", default="fig_heatmap.png")
    hm.set_defaults(func=cmd_heatmap)
    a = ap.parse_args(); a.func(a)


if __name__ == "__main__":
    main()
