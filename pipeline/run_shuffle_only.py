"""
Shuffle-only experiment (paper Figure 5): the effect of reordering the answer options ALONE, with no
added distractors. Kept separate from the main experiment (run_eval.py) so that run stays exactly as
used; this script reuses the same building blocks and the same explicit-answer prompt.

For each sampled MammoVQA item the model is queried under two conditions on the real image:
  original      -- the standard option set (+ "None of the above")
  shuffle_only  -- the SAME options reordered (seeded per item), still no distractors

Rows are written in the same format as run_eval.py (explicit stated answer, per-option scores,
explanation, …). Pair original vs shuffle_only to get the answer-change (flip) rate:

  cd pipeline && python run_shuffle_only.py --model_name <hf> --rsna_png ... --embed_png ... \
      --topics birads,acr --output_dir shuffle_<model>
  python -m analysis.perturbation flips --conditions shuffle_only \
      --glob "shuffle_*/results.json" --out fig_shuffle_flips.png
"""
import os
import json
import random
import argparse

from tqdm import tqdm

from run_eval import (ALPHA, TASKS, load_benchmark, sample_task, resolve_images, build_option_set,
                      conf_prompt_with_answer, conf_prompt, conf_prompt_explanation_first,
                      parse_conf_full, parse_conf)


def main(args):
    topics = [t.strip() for t in args.topics.split(",") if t.strip()]
    datasets = set(d.strip() for d in args.datasets.split(",") if d.strip())

    items = load_benchmark(args.benchmark)
    sampled = {t: sample_task(items, TASKS[t], datasets, args.n, args.seed) for t in topics}
    for t in topics:
        print(f"  {t}: {len(sampled[t])} sampled  (options={TASKS[t]['options']})")

    from vlm_inference import load_model, run_inference_multi
    model, processor, mtype = load_model(args.model_name)

    def infer(imgs, prompt):
        return run_inference_multi(model, processor, mtype, imgs, prompt, max_new_tokens=384)

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "results.json")

    def save(rows):
        with open(out + ".tmp", "w") as f:
            json.dump({"model": args.model_name, "datasets": sorted(datasets),
                       "topics": topics, "n": args.n, "experiment": "shuffle_only",
                       "rows": rows}, f)
        os.replace(out + ".tmp", out)

    # resume
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

            opts_std, add_std = build_option_set(base_opts, abstain=True)
            so_rng = random.Random(f"{pkey}|{gt}|shuffle")   # reorder std options, NO fillers
            opts_so, add_so = build_option_set(base_opts, abstain=True, n_fillers=0, rng=so_rng)
            plan = [
                ("original",     q, opts_std, add_std),
                ("shuffle_only", q, opts_so,  add_so),
            ]

            if args.scores_only:
                prompt_fn = conf_prompt_explanation_first if args.explanation_first else conf_prompt
            else:
                prompt_fn = conf_prompt_with_answer

            for cond, question, opts, added in plan:
                labels = ALPHA[:len(opts)]
                prompt = prompt_fn(question, opts, labels, text_only=False)
                raw = infer(real, prompt)
                row = {"task": t, "path": path, "condition": cond,
                       "question": question, "options": opts,
                       "gt": gt, "orig_gt": gt, "raw": raw}
                if args.scores_only:
                    scores, idx, pred_text, conf, expl = parse_conf(raw, opts, labels)
                else:
                    a_idx, a_text, scores, idx, pred_text, conf, expl = parse_conf_full(raw, opts, labels)
                    row["explicit_idx"] = a_idx
                    row["explicit_answer"] = a_text
                    row["explicit_correct"] = int(a_text == gt) if a_text is not None else 0
                row.update({"scores": scores, "pred": idx, "pred_text": pred_text,
                            "confidence": conf, "explanation": expl,
                            "correct": int(pred_text == gt) if pred_text is not None else 0,
                            "picked_new": int(pred_text in added) if pred_text is not None else 0})
                rows.append(row)
            done.add((t, pkey))
            processed += 1
            if processed % args.save_every == 0:
                save(rows)

    save(rows)
    expl_ans = sum(1 for r in rows if r.get("explicit_answer") is not None)
    print(f"wrote {out}  ({len(rows)} rows, {expl_ans} with an explicit answer)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="MammoVQA-Image-Bench.json")
    ap.add_argument("--image_root", default=".")
    ap.add_argument("--rsna_png", default="rsna_png")
    ap.add_argument("--embed_png", default="embed_png")
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--datasets", default="RSNA,EMBED")
    ap.add_argument("--topics", default="birads,acr,laterality,view")
    ap.add_argument("--n", type=int, default=20000, help="max questions per topic")
    ap.add_argument("--scores_only", action="store_true",
                    help="legacy: score-only prompt (no explicit answer line)")
    ap.add_argument("--explanation_first", action="store_true",
                    help="with --scores_only, put the explanation before the per-option scores")
    ap.add_argument("--save_every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", default="shuffle_output",
                    help="use a 'shuffle_*' name so it is NOT matched by the main 'output_*' glob")
    main(ap.parse_args())
