#!/usr/bin/env bash
# End-to-end pipeline: preprocess DICOMs -> run each model -> analyse.
# Edit the paths/models below, then: bash run_all.sh
set -euo pipefail

# ----------------------------------------------------------------- config
BENCHMARK="MammoVQA-Image-Bench.json"      # or MammoVQA-Exam-Bench.json for EMBED exams
RSNA_PNG="rsna_png"                         # output of prepare_images.py for RSNA
EMBED_PNG="embed_png"                       # output of prepare_images.py for EMBED
UNRELATED="unrelated_diverse"              # folder of out-of-domain images (ImageNet sample)
DATASETS="RSNA,EMBED"
TOPICS="birads,acr"

# The four evaluated models (Hugging Face ids). Pin revisions as needed with @<commit>.
MODELS=(
  "google/medgemma-4b-it"
  "lingshu-medical-mllm/Lingshu-7B"
  "Qwen/Qwen2-VL-7B-Instruct"
  "OpenGVLab/InternVL3-8B"
)

# Optional DICOM->image conversion. Set these to your DICOM folders to enable; leave empty to skip
# (e.g. if the images are already prepared).
RSNA_DICOM_DIR=""
EMBED_DICOM_DIR=""

# ----------------------------------------------------------------- 1. preprocess
if [[ -n "$RSNA_DICOM_DIR" ]]; then
  echo "== preprocessing RSNA DICOMs =="
  python preprocessing/prepare_images.py --dataset rsna --dicom_dir "$RSNA_DICOM_DIR" --out_dir "$RSNA_PNG"
fi
if [[ -n "$EMBED_DICOM_DIR" ]]; then
  echo "== preprocessing EMBED DICOMs =="
  python preprocessing/prepare_images.py --dataset embed --dicom_dir "$EMBED_DICOM_DIR" --out_dir "$EMBED_PNG"
fi

# ----------------------------------------------------------------- 2. run each model
for M in "${MODELS[@]}"; do
  TAG="output_$(basename "$M")"
  echo "== evaluating $M -> $TAG =="
  ( cd pipeline && python run_eval.py \
      --benchmark "$BENCHMARK" \
      --rsna_png "../$RSNA_PNG" --embed_png "../$EMBED_PNG" \
      --unrelated_root "../$UNRELATED" \
      --model_name "$M" --datasets "$DATASETS" --topics "$TOPICS" \
      --output_dir "../$TAG" )
  echo "== shuffle-only (Figure 5) $M -> shuffle_$(basename "$M") =="
  ( cd pipeline && python run_shuffle_only.py \
      --benchmark "$BENCHMARK" \
      --rsna_png "../$RSNA_PNG" --embed_png "../$EMBED_PNG" \
      --model_name "$M" --datasets "$DATASETS" --topics "$TOPICS" \
      --output_dir "../shuffle_$(basename "$M")" )
done

# ----------------------------------------------------------------- 3. analyse
GLOB="output_*/results.json"
echo "== analysis =="
python -m analysis.coherence    gap   --glob "$GLOB" --topics "$TOPICS" --out fig_ar_gap.png
python -m analysis.coherence    taxonomy --glob "$GLOB" --topics "$TOPICS" --out fig_ar_taxonomy.png
python -m analysis.confidence   gt    --glob "$GLOB" --topics "$TOPICS" --out fig_gt_confidence.png
python -m analysis.nota         rate  --glob "$GLOB" --topics "$TOPICS" --out fig_nota_rate.png
python -m analysis.perturbation flips --glob "$GLOB" --topics "$TOPICS" --separate --out fig_flips.png
python -m analysis.perturbation flips --glob "shuffle_*/results.json" --topics "$TOPICS" \
    --conditions shuffle_only --out fig_shuffle_flips.png
python -m analysis.mcnemar             --glob "$GLOB" --topics "$TOPICS" --out mcnemar.csv

echo "== done =="
