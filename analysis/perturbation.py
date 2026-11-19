"""
Perturbation sensitivity and the option-position artifact. Subcommands:

  flips         answer-change (flip) rate vs original, split helped / hurt / neutral, with a hatched
                inset for wrong->wrong flips that retreated to NOTA (--separate or a combined grid)
  position      positional pick histogram (original), original-vs-perturbed pick compare, and
                accuracy vs the correct label's shuffled index (writes 3 figures)
  distribution  which answer each model picks most in perturbed_options (stacked by category)

  python -m analysis.perturbation flips        --glob "..." --separate --out fig_flips.png
  python -m analysis.perturbation position     --glob "..." --out fig_position.png
  python -m analysis.perturbation distribution --glob "..." --out fig_answer_distribution.png
"""
import argparse, os, csv
from collections import defaultdict, Counter
import numpy as np

from analysis.shared import (plt, MODEL_ORDER, MODEL_COL, TASK_LABEL, order_models, out_pred, gtof,
                             pkey, short, ABSTAIN, ABSTAIN_LIKE, add_std_args, topics_of, exclude_of, save)
from signals.answer_mapping import is_tie

C_HELP, C_HURT, C_NEU, C_NEU_NOTA = "#1D9E75", "#C0504D", "#B7B7B7", "#6E6E6E"
COND_TITLE = {"perturbed_question": "Question-wording perturbation (items whose wording changed)",
              "shuffle_only": "Answer-option reordering (shuffled, no distractors)",
              "perturbed_options": "Answer-option perturbation (shuffled + fillers)",
              "unrelated": "Unrelated photo (junk)", "blank": "Blank image (junk)", "noimg": "No image"}
COND_SHORT = {"perturbed_question": "reworded Q", "shuffle_only": "shuffled opts",
              "perturbed_options": "shuffled opts",
              "unrelated": "unrelated photo", "blank": "blank", "noimg": "no image"}


# ============================================================ flips
def _collect(a, cond):
    import glob as glob_mod, json
    excl = [x.lower() for x in exclude_of(a)]; topics = topics_of(a)
    out = defaultdict(lambda: defaultdict(list)); n_tie = n_nochange = 0
    for p in glob_mod.glob(a.glob):
        d = json.load(open(p)); m = short(d["model"])
        if any(x in m.lower() for x in excl): continue
        orig, pert = {}, {}
        for r in d["rows"]:
            if r.get("task") not in topics: continue
            if r.get("condition") == "original": orig[(r["task"], pkey(r))] = r
            elif r.get("condition") == cond:     pert[(r["task"], pkey(r))] = r
        for k, pr in pert.items():
            orr = orig.get(k)
            if orr is None: continue
            if a.discard_ties and (is_tie(orr.get("scores") or [], a.tau) or is_tie(pr.get("scores") or [], a.tau)):
                n_tie += 1; continue
            if cond == "perturbed_question" and \
               (orr.get("question") or "").strip() == (pr.get("question") or "").strip():
                n_nochange += 1; continue
            gt = gtof(orr); op = out_pred(orr, a.tau); pp = out_pred(pr, a.tau)
            out[m][k[0]].append({"changed": op != pp, "orig_ok": op == gt, "pert_ok": pp == gt,
                                 "pert_abstain": pp in ABSTAIN_LIKE})
    if a.discard_ties: print(f"[{cond}] discarded {n_tie} tie pairs")
    if cond == "perturbed_question": print(f"[{cond}] skipped {n_nochange} items with unchanged wording")
    return out, topics


def _summarize(items):
    n = len(items)
    if not n: return None
    ch = sum(i["changed"] for i in items)
    helped = sum(i["changed"] and (not i["orig_ok"]) and i["pert_ok"] for i in items)
    hurt = sum(i["changed"] and i["orig_ok"] and (not i["pert_ok"]) for i in items)
    neutral = ch - helped - hurt
    neutral_nota = sum(i["changed"] and (not i["orig_ok"]) and (not i["pert_ok"]) and i["pert_abstain"]
                       for i in items)
    o_acc = 100*np.mean([i["orig_ok"] for i in items]); p_acc = 100*np.mean([i["pert_ok"] for i in items])
    f = 100.0/n
    return dict(n=n, flip=ch*f, helped=helped*f, hurt=hurt*f, neutral=neutral*f,
                neutral_nota=neutral_nota*f,
                neutral_nota_share=(100.0*neutral_nota/neutral if neutral else float("nan")),
                orig_acc=o_acc, pert_acc=p_acc, net=p_acc-o_acc)


def _models_of(data): return [m for m in MODEL_ORDER if m in data] + [m for m in data if m not in MODEL_ORDER]


def _draw_panel(ax, data, cond, t, rows_out, xlabels=True):
    ms = [m for m in _models_of(data) if _summarize(data[m][t])]
    any_bar = False; bw = 0.62
    for xi, m in enumerate(ms):
        s = _summarize(data[m][t]); any_bar = True
        ax.bar(xi, s["helped"], bw, color=C_HELP, edgecolor="white",
               label="helped (wrong→right)" if xi == 0 else None)
        ax.bar(xi, s["hurt"], bw, bottom=s["helped"], color=C_HURT, edgecolor="white",
               label="hurt (right→wrong)" if xi == 0 else None)
        base_neu = s["helped"] + s["hurt"]
        ax.bar(xi, s["neutral"], bw, bottom=base_neu, color=C_NEU, edgecolor="white",
               label="neutral (wrong→wrong)" if xi == 0 else None)
        if s["neutral_nota"] > 0:
            ax.bar(xi, s["neutral_nota"], bw, bottom=base_neu, color=C_NEU, edgecolor=C_NEU_NOTA,
                   linewidth=1.1, hatch="////", label="   ↳ of which → NOTA/abstain" if xi == 0 else None)
        ax.annotate(f"{s['net']:+.0f}", (xi, s["flip"]+1.5), ha="center", fontsize=8,
                    color=("#1a7a4a" if s["net"] > 0 else "#a33" if s["net"] < 0 else "#555"), fontweight="bold")
        rows_out.append({"condition": cond, "model": short(m), "topic": t, "n": s["n"],
                         "flip_rate": round(s["flip"], 1), "helped_pct": round(s["helped"], 1),
                         "hurt_pct": round(s["hurt"], 1), "neutral_pct": round(s["neutral"], 1),
                         "neutral_to_nota_pct": round(s["neutral_nota"], 1),
                         "orig_acc": round(s["orig_acc"], 1), "pert_acc": round(s["pert_acc"], 1),
                         "net_acc": round(s["net"], 1)})
    ax.set_xticks(np.arange(len(ms)))
    ax.set_xticklabels([f"{short(m)}\n(n={_summarize(data[m][t])['n']})" for m in ms] if xlabels
                       else [short(m) for m in ms], rotation=15 if xlabels else 20,
                       ha="center" if xlabels else "right", fontsize=8 if xlabels else 7)
    if len(ms): ax.set_xlim(-0.7, len(ms)-0.3)
    ax.set_ylim(0, 108); ax.grid(axis="y", alpha=.25)
    return any_bar


def _legend_save(fig, out):
    seen, H, L = set(), [], []
    for ax in fig.axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l and l not in seen: seen.add(l); H.append(h); L.append(l)
    fig.legend(H, L, fontsize=9, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    save(fig, out)


def cmd_flips(a):
    conds = [c.strip() for c in a.conditions.split(",") if c.strip()]
    rows_out = []; per_cond = {}; topics = None
    for cond in conds:
        per_cond[cond], topics = _collect(a, cond)
    stem, ext = os.path.splitext(a.out)
    if a.separate:
        for cond in conds:
            fig, axes = plt.subplots(1, len(topics), figsize=(6.0*len(topics), 5.2), squeeze=False, sharey=True)
            any_bar = False
            for ax, t in zip(axes[0], topics):
                any_bar |= _draw_panel(ax, per_cond[cond], cond, t, rows_out)
                ax.set_title(TASK_LABEL.get(t, t), fontsize=12)
            if not any_bar: plt.close(fig); print(f"[{cond}] no matched items"); continue
            axes[0][0].set_ylabel("% of items whose answer CHANGED\n(stacked by effect on accuracy)")
            fig.suptitle(COND_TITLE.get(cond, cond) + " — bar = flip rate; hatched = retreated to NOTA; number = net acc",
                         y=1.05, fontsize=12)
            fig.tight_layout(rect=[0, 0.05, 1, 1]); _legend_save(fig, f"{stem}_{cond}{ext}")
    else:
        fig, axes = plt.subplots(len(conds), len(topics), figsize=(5.4*len(topics), 3.1*len(conds)),
                                 squeeze=False, sharey=True)
        for ri, cond in enumerate(conds):
            for cj, t in enumerate(topics):
                _draw_panel(axes[ri][cj], per_cond[cond], cond, t, rows_out, xlabels=(ri == len(conds)-1))
                if ri == 0: axes[ri][cj].set_title(TASK_LABEL.get(t, t), fontsize=12)
                if cj == 0: axes[ri][cj].set_ylabel(f"{COND_SHORT.get(cond, cond)}\n% changed", fontsize=9)
        fig.suptitle("Answer change vs original across all conditions (hatched = retreated to NOTA; number = net acc)",
                     y=1.005, fontsize=12)
        fig.tight_layout(rect=[0, 0.03, 1, 0.99]); _legend_save(fig, f"{stem}_grid{ext}")
    _csv(a.csv, rows_out, ["condition", "model", "topic", "n", "flip_rate", "helped_pct", "hurt_pct",
                           "neutral_pct", "neutral_to_nota_pct", "orig_acc", "pert_acc", "net_acc"])


# ============================================================ position
def _picked_idx(r, tau):
    opts = r.get("options") or []; p = out_pred(r, tau)
    return opts.index(p) if p in opts else None
def _gt_idx(r):
    opts = r.get("options") or []; g = gtof(r)
    return opts.index(g) if g in opts else None


def cmd_position(a):
    import glob as glob_mod, json
    excl = [x.lower() for x in exclude_of(a)]; topics = topics_of(a)
    by = defaultdict(list)
    for p in glob_mod.glob(a.glob):
        d = json.load(open(p)); m = short(d["model"])
        if any(x in m.lower() for x in excl): continue
        by[m] += [r for r in d["rows"] if r.get("task") in topics]
    models = order_models(by); stem, ext = os.path.splitext(a.out)

    # --- pick histogram (original) ---
    npos = {}; pick = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    gtc = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for m in models:
        for r in by[m]:
            if r.get("condition") != "original": continue
            if a.discard_ties and is_tie(r.get("scores") or [], a.tau): continue
            t = r["task"]; npos[t] = max(npos.get(t, 0), len(r.get("options") or []))
            pi = _picked_idx(r, a.tau); gi = _gt_idx(r)
            if pi is not None: pick[m][t][pi] += 1
            if gi is not None: gtc[m][t][gi] += 1
    fig, axes = plt.subplots(len(models), len(topics), figsize=(5.2*len(topics), 2.1*len(models)),
                             squeeze=False, sharex="col")
    for ri, m in enumerate(models):
        for cj, t in enumerate(topics):
            ax = axes[ri][cj]; K = npos.get(t, 0)
            if not K: ax.axis("off"); continue
            tot = sum(pick[m][t].values()); gtot = sum(gtc[m][t].values()); xs = np.arange(K)
            pv = [100*pick[m][t][i]/tot if tot else 0 for i in xs]
            gv = [100*gtc[m][t][i]/gtot if gtot else 0 for i in xs]
            ax.bar(xs, pv, color=MODEL_COL.get(m, "#666"), width=0.8, label="picked f(A)")
            ax.step(np.append(xs-0.5, K-0.5), np.append(gv, gv[-1]), where="post", color="black",
                    lw=1.4, label="correct-label position")
            ax.set_ylim(0, 100)
            if cj == 0: ax.set_ylabel(f"{short(m)}\n% items", fontsize=8)
            if ri == 0: ax.set_title(TASK_LABEL.get(t, t), fontsize=12)
            lab = [str(i) for i in xs]; lab[-1] = "abst"; ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=7)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=9, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Positional pick histogram (original): picked option INDEX vs correct-label position", y=1.01, fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.99]); save(fig, f"{stem}_pick_hist{ext}")

    # --- accuracy vs gold position (perturbed_options) ---
    K = {}; hit = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for m in models:
        for r in by[m]:
            if r.get("condition") != "perturbed_options": continue
            if a.discard_ties and is_tie(r.get("scores") or [], a.tau): continue
            gi = _gt_idx(r)
            if gi is None: continue
            t = r["task"]; K[t] = max(K.get(t, 0), len(r.get("options") or []))
            hit[m][t][gi].append(int(out_pred(r, a.tau) == gtof(r)))
    fig, axes = plt.subplots(1, len(topics), figsize=(5.8*len(topics), 4.8), squeeze=False, sharey=True)
    for cj, t in enumerate(topics):
        ax = axes[0][cj]; xs = np.arange(K.get(t, 0))
        for m in models:
            ys = [100*np.mean(hit[m][t][i]) if hit[m][t][i] else np.nan for i in xs]
            ax.plot(xs, ys, "o-", color=MODEL_COL.get(m, "#666"), lw=2, label=short(m))
        ax.set_xticks(xs); ax.set_ylim(0, 105)
        ax.set_xlabel("index of the CORRECT label in the shuffled option list")
        ax.set_title(TASK_LABEL.get(t, t), fontsize=12); ax.grid(alpha=.25)
    axes[0][0].set_ylabel("accuracy when the gold label sits at that index (%)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=9, loc="lower center", ncol=len(models), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Accuracy vs the correct answer's shuffled position (perturbed_options)", y=1.02, fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 1]); save(fig, f"{stem}_acc_by_pos{ext}")


# ============================================================ distribution
FILLERS_DIST = {"Not Applicable", "Not Assessable", "Cannot Be Determined", "Indeterminate", "Other", "Unclear"}
FILLER_CAT, NOTA_CAT = "(filler)", "None of the above"


def _categorize(ans):
    if ans == ABSTAIN: return NOTA_CAT
    if ans in FILLERS_DIST: return FILLER_CAT
    return ans


def cmd_distribution(a):
    import glob as glob_mod, json
    topics = topics_of(a); excl = [x.lower() for x in exclude_of(a)]
    by = defaultdict(list); gt_by = defaultdict(Counter)
    for p in glob_mod.glob(a.glob):
        d = json.load(open(p)); m = short(d["model"])
        if any(x in m.lower() for x in excl): continue
        for r in d["rows"]:
            if r.get("condition") == a.condition and r.get("task") in topics:
                by[(m, r["task"])].append(r); gt_by[r["task"]][gtof(r)] += 1
    models = [m for m in MODEL_ORDER if any((m, t) in by for t in topics)] + \
             [m for m in {k[0] for k in by} if m not in MODEL_ORDER]
    dist = {}; rows_out = []
    for (m, t), rs in by.items():
        c = Counter()
        for r in rs:
            if a.discard_ties and is_tie(r.get("scores") or [], a.tau): continue
            c[_categorize(out_pred(r, a.tau))] += 1
        dist[(m, t)] = c
    fig, axes = plt.subplots(1, len(topics), figsize=(7.2*len(topics), 0.7*len(models)+2.4), squeeze=False)
    for ax, t in zip(axes[0], topics):
        labels = sorted(gt_by[t]); cats = labels + [NOTA_CAT, FILLER_CAT]
        cmap = plt.cm.tab10(np.linspace(0, 1, max(len(labels), 3)))
        colr = {lab: cmap[i] for i, lab in enumerate(labels)}
        colr[NOTA_CAT] = "#222222"; colr[FILLER_CAT] = "#c9c9c9"
        rowlab, rowcnt = ["gold (truth)"], [gt_by[t]]
        for m in models:
            if (m, t) in dist: rowlab.append(short(m)); rowcnt.append(dist[(m, t)])
        y = np.arange(len(rowlab))
        for yi, (rl, cnt) in enumerate(zip(rowlab, rowcnt)):
            tot = sum(cnt.values()) or 1; left = 0.0
            for cat in cats:
                v = 100*cnt.get(cat, 0)/tot
                if v <= 0: continue
                ax.barh(yi, v, left=left, color=colr[cat], edgecolor="white",
                        hatch=("////" if cat == FILLER_CAT else None),
                        label=cat if yi == 0 or (cat in (NOTA_CAT, FILLER_CAT)) else None)
                left += v
            if rl != "gold (truth)" and cnt:
                top, tn = cnt.most_common(1)[0]
                ax.plot(101, yi, "o", color=colr.get(top, "#666"), ms=7, clip_on=False)
                rows_out.append({"model": rl, "topic": t, "top_answer": top,
                                 "top_pct": round(100*tn/(sum(cnt.values()) or 1), 1), "n": sum(cnt.values())})
        ax.set_yticks(y); ax.set_yticklabels(rowlab, fontsize=9); ax.invert_yaxis()
        ax.set_xlim(0, 100); ax.set_xlabel("% of items answered with this option")
        ax.set_title(TASK_LABEL.get(t, t), fontsize=12)
        hs, ls = ax.get_legend_handles_labels(); seen = dict(zip(ls, hs))
        ax.legend([seen[k] for k in cats if k in seen], [k for k in cats if k in seen],
                  fontsize=7.5, ncol=2, loc="lower right", framealpha=.9)
    fig.suptitle(f"Which answer each model picks most ({a.condition}); top row = true label distribution",
                 y=1.02, fontsize=12)
    _csv(a.csv, rows_out, ["model", "topic", "top_answer", "top_pct", "n"])
    fig.tight_layout(rect=[0, 0, 1, 0.99]); save(fig, a.out)


def _csv(path, rows, fields):
    if not path or not rows: return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = add_std_args(sub.add_parser("flips"))
    f.add_argument("--conditions", default="perturbed_question,perturbed_options,unrelated,blank,noimg")
    f.add_argument("--separate", action="store_true")
    f.add_argument("--out", default="fig_perturbation_flips.png"); f.add_argument("--csv", default="perturbation_flips.csv")
    f.set_defaults(func=cmd_flips)
    p = add_std_args(sub.add_parser("position"))
    p.add_argument("--out", default="fig_position.png"); p.add_argument("--csv", default="position_bias.csv")
    p.set_defaults(func=cmd_position)
    d = add_std_args(sub.add_parser("distribution"))
    d.add_argument("--condition", default="perturbed_options")
    d.add_argument("--out", default="fig_answer_distribution.png"); d.add_argument("--csv", default="answer_distribution.csv")
    d.set_defaults(func=cmd_distribution)
    a = ap.parse_args(); a.func(a)


if __name__ == "__main__":
    main()
