# Evaluating Coherence Among Answers, Confidence, and Reasoning in Mammography Vision-Language Models for Breast Cancer Assessment

![Perturbations figure](images/perturbations_figure.png)

Code for the paper *Evaluating Coherence Among Answers, Confidence, and Reasoning in Mammography
Vision-Language Models for Breast Cancer Assessment*. We test whether a model's **answer, confidence,
and reasoning** agree with each other and with the ground truth on multiple-choice mammography VQA,
and whether they change sensibly under controlled perturbations.

Each question is answered in one generation. The model gives: (1) an **explicit single-best answer**,
(2) a 0–100 confidence score for every option (including *None of the above*), and (3) a one-sentence
explanation. The **explicit answer is the primary signal** — it is always well-defined, so no items
are dropped for score ties. Tasks are BI-RADS and breast density (ACR); laterality and view are also
supported.

## Data

Images are **not distributed here** — RSNA and EMBED are governed by data-use agreements. Obtain the
four inputs below and point the runner at them.

| Input | Download | Reference | Access |
|---|---|---|---|
| **RSNA** screening mammograms | [Link](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data) | [RSNA Screening Mammography Breast Cancer Detection AI Challenge](https://www.rsna.org/rsnai/ai-image-challenge/screening-mammography-breast-cancer-detection-ai-challenge) | Open |
| **EMBED** mammograms | [Link](https://registry.opendata.aws/emory-breast-imaging-dataset-embed/) | [The EMBED dataset (3.4M mammographic images)](https://pubs.rsna.org/doi/pdf/10.1148/ryai.220047) | Credentialed |
| **MammoVQA** benchmark (questions, options, answers) | [Link](https://github.com/PiggyJerry/MammoVQA) | [A Benchmark for Breast Cancer Screening and Diagnosis in Mammogram VQA](https://doi.org/10.1038/s41467-025-66507-z) | Open |
| **Unrelated images** for the unrelated-photo perturbation (ImageNet sample) | [Link](https://image-net.org) | Deng et al., 2009 | Open |

## Pipeline

Three steps — **preprocess → run → analyze**. `run_all.sh` chains all three for one model (edit the
paths at the top, then `bash run_all.sh`); the sections below explain each. Install dependencies
first:

```bash
pip install -r requirements.txt
```

### 1. Preprocess

`preprocessing/prepare_images.py` converts each DICOM to an 8-bit image, following MammoVQA's
conversion (paper, Appendix B):

1. apply VOI-LUT (header windowing); invert `MONOCHROME1`
2. rescale to 0–255 grayscale; crop to the breast region
3. downscale the longer side to ≤1024 px; save as JPEG (quality 90)

```bash
python preprocessing/prepare_images.py --dicom_dir /path/to/dicoms --out_dir embed_png --format jpg
```

Name each output to match its MammoVQA path so the runner can find it — RSNA →
`<rsna_png>/<pid>_<iid>.png`, EMBED → `<embed_png>/<name>.jpg`. EMBED examinations are multi-view:
each view is converted separately, and the runner feeds an exam's views together. (The grayscale→RGB
channel replication happens at inference, not here.)

### 2. Run a model

Queries one model over the seven conditions (original; unrelated / blank / no image; reworded
question; shuffled options; shuffled options + distractors):

```bash
cd pipeline
python run_eval.py \
    --benchmark MammoVQA-Image-Bench.json \
    --rsna_png /path/to/rsna_png --embed_png /path/to/embed_png \
    --unrelated_root /path/to/imagenet_sample \
    --model_name <hf-model> --datasets RSNA,EMBED --topics birads,acr \
    --output_dir output_<model>
```

Writes `output_<model>/results.json` — per item and condition: the explicit answer
(`explicit_answer`, primary), the per-option scores and score-argmax (`scores`/`pred_text`,
secondary), the explanation, and the raw output. Resumes if interrupted. Model backends live in
`pipeline/vlm_inference.py`.

**Reordering alone (Figure 5)** is a separate run that reshuffles the options without distractors:

```bash
python run_shuffle_only.py \
    --rsna_png /path/to/rsna_png --embed_png /path/to/embed_png \
    --model_name <hf-model> --datasets RSNA,EMBED --topics birads,acr \
    --output_dir shuffle_<model>
```

The `shuffle_*` name keeps these results out of the main `output_*` glob.

### 3. Analyze

All analysis uses the explicit answer (keep-ties). Run from the repo root; `--help` lists options.
Each module defaults to `--glob "output_*/results.json"` and `--topics birads,acr`.

| Command | Output |
|---|---|
| `python -m analysis.coherence gap` | answer accuracy vs reasoning accuracy, with the disconnect taxonomy |
| `python -m analysis.confidence gt` | confidence trajectories as visual evidence is removed |
| `python -m analysis.nota rate` | *None of the above* selection across conditions |
| `python -m analysis.perturbation flips` | answer-change rate under option/wording perturbations |
| `python -m analysis.mcnemar` | exact McNemar significance (SciPy) of each perturbation |

Figure 5 uses the shuffle-only results:
`python -m analysis.perturbation flips --conditions shuffle_only --glob "shuffle_*/results.json"`.

`signals/` holds the shared extractors: the reasoning-answer matcher (`reasoning_parser.py`),
reasoning certainty (`reasoning_confidence.py`), matcher-vs-human validation with Cohen's κ via
scikit-learn (`parser_validation.py`), and the secondary score-argmax map (`answer_mapping.py`).
