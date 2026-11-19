# Evaluating Coherence Among Answers, Confidence, and Reasoning in Mammography Vision-Language Models for Breast Cancer Assessment

![Perturbations figure](images/perturbations_figure.png)

Code for the paper *Evaluating Coherence Among Answers, Confidence, and Reasoning in Mammography
Vision-Language Models for Breast Cancer Assessment*. For each multiple-choice mammography question a
model gives an explicit answer, a 0–100 confidence per option, and a one-sentence explanation; we
measure whether these agree and how they hold up under controlled perturbations. The explicit answer
is the primary signal. Tasks: BI-RADS and breast density (ACR).

## Data

Images are **not distributed here** — RSNA and EMBED are governed by data-use agreements.

| Input | Download | Reference | Access |
|---|---|---|---|
| **RSNA** screening mammograms | [Link](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data) | [RSNA Screening Mammography Breast Cancer Detection AI Challenge](https://www.rsna.org/rsnai/ai-image-challenge/screening-mammography-breast-cancer-detection-ai-challenge) | Open |
| **EMBED** mammograms | [Link](https://registry.opendata.aws/emory-breast-imaging-dataset-embed/) | [The EMBED dataset (3.4M mammographic images)](https://pubs.rsna.org/doi/pdf/10.1148/ryai.220047) | Credentialed |
| **MammoVQA** benchmark | [Link](https://github.com/PiggyJerry/MammoVQA) | [A Benchmark for Breast Cancer Screening and Diagnosis in Mammogram VQA](https://doi.org/10.1038/s41467-025-66507-z) | Open |
| **Unrelated images** (ImageNet sample) | [Link](https://image-net.org) | Deng et al., 2009 | Open |

## Usage

```bash
pip install -r requirements.txt
```

Three steps — preprocess, run, analyze. `run_all.sh` chains them for one model.

**1. Preprocess** DICOMs to 8-bit images (VOI-LUT → invert `MONOCHROME1` → 0–255 grayscale → breast
crop → ≤1024 px → JPEG q90):

```bash
python preprocessing/prepare_images.py --dicom_dir /path/to/dicoms --out_dir embed_png --format jpg
```

Name each output to match its MammoVQA path (`<rsna_png>/<pid>_<iid>.png`, `<embed_png>/<name>.jpg`).
EMBED examinations are multi-view; the runner feeds an exam's views together.

**2. Run a model** over the seven conditions (original; unrelated / blank / no image; reworded
question; shuffled options; shuffled + distractors):

```bash
cd pipeline
python run_eval.py \
    --benchmark MammoVQA-Image-Bench.json \
    --rsna_png /path/to/rsna_png --embed_png /path/to/embed_png \
    --unrelated_root /path/to/imagenet_sample \
    --model_name <hf-model> --datasets RSNA,EMBED --topics birads,acr \
    --output_dir output_<model>
```

Reordering options alone (Figure 5) is a separate run; the `shuffle_*` name keeps it out of the main
`output_*` glob:

```bash
python run_shuffle_only.py \
    --rsna_png /path/to/rsna_png --embed_png /path/to/embed_png \
    --model_name <hf-model> --datasets RSNA,EMBED --topics birads,acr \
    --output_dir shuffle_<model>
```

**3. Analyze** from the repo root (default glob `output_*/results.json`, `--topics birads,acr`; `--help`
for options):

| Command | Output |
|---|---|
| `python -m analysis.coherence gap` | answer vs reasoning accuracy, with the disconnect taxonomy |
| `python -m analysis.confidence gt` | confidence as visual evidence is removed |
| `python -m analysis.nota rate` | *None of the above* selection across conditions |
| `python -m analysis.perturbation flips` | answer-change rate under perturbations |
| `python -m analysis.mcnemar` | exact McNemar significance (SciPy) |

Figure 5: `python -m analysis.perturbation flips --conditions shuffle_only --glob "shuffle_*/results.json"`.
