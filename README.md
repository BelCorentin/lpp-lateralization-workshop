# Le Petit Prince — LLM → fMRI encoding and left lateralization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BelCorentin/lpp-lateralization-workshop/blob/main/notebooks/lpp_lateralization.ipynb)

A hands-on workshop replicating the core result of

> Bonnasse-Gahot, L. & Pallier, C. (2024).
> *fMRI predictors based on language models of increasing complexity recover
> brain left lateralization.* NeurIPS 37. https://arxiv.org/abs/2405.17992

on the **average-subject** Le Petit Prince fMRI data
([chrplr/lpp_average_subject_en](https://github.com/chrplr/lpp_average_subject_en)),
using [`neuralset`](https://pypi.org/project/neuralset/) for the event/feature
pipeline.

The pipeline in one line:

```
LPP study  ->  word events  ->  LLM embeddings  ->  HRF convolution  ->  ridge (leave-one-run-out)  ->  per-voxel r  ->  left vs right
```

## Quickest start: Colab

Click the badge above. The first cell installs everything and downloads the
data; nothing to set up, **no GPU needed**. The whole notebook — brain maps,
model sweeps, statistics — runs end to end in **under 3 minutes** on a
workstation, or roughly 10 on a free Colab box, plus the one-off 700 MB data
download.

That is possible because two expensive steps ship precomputed (see
[Why it is fast](#why-it-is-fast) below). Everything that constitutes the
*result* is still computed on your machine.

Everything below is for running it on your own machine instead.

## 1. Install

You need Python **3.12+** and `git`. Everything else comes from PyPI.

With [uv](https://docs.astral.sh/uv/) (recommended, fastest):

```bash
git clone <this repo> lpp-lateralization-workshop
cd lpp-lateralization-workshop
uv venv --python 3.12
uv pip install -e .
```

Or with plain pip:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

A GPU is **not** required — the small models run on CPU in a few minutes.
If you have one, PyTorch will use it automatically.

> On Linux, `pip install torch` pulls the CUDA build (~2.5 GB). For a
> laptop-only workshop room, install the CPU wheel first and the rest after:
>
> ```bash
> uv pip install torch --index-url https://download.pytorch.org/whl/cpu
> uv pip install -e .
> ```

## 2. Get the data (~700 MB)

```bash
source .venv/bin/activate
python -c "from lppws.study import download_data; download_data('data')"
```

This fetches only what the analysis reads — BOLD, word annotations, brain mask,
ROI masks — into `data/lpp_average_subject_en/`, over plain HTTPS: no `git`, no
Git-LFS, no OpenNeuro account, no credentials. Re-running it skips what is
already there, so an interrupted download just needs the same command again.

It deliberately leaves out the repository's `stimuli/` folder: that is 467 MB of
*French* audio, and the fMRI here is from English listeners. (`clone_data()` is
available if you would rather have a `git` checkout you can `pull`, but it costs
about three times the disk, because git stores every file a second time inside
`.git`.)

## 3. Run

```bash
jupyter lab notebooks/lpp_lateralization.ipynb
```

`notebooks/lpp_lateralization.executed.ipynb` is the same notebook with every
output filled in, so you can see where you should be landing before running
anything. **Read `RESULTS.md` before presenting any of this.** Short version: the
lateralization replicates, and so does its growth *with training* (pythia-70m
from random init to fully trained, p=0.003) — but not with model *size*, and
the reason (which layer you read) matters.

or, headless, the same analysis as a script:

```bash
python verify.py                   # gpt2, static embeddings, ~2 min
python verify.py '[["pythia-70m","EleutherAI/pythia-70m",true]]'
```

## Why it is fast

Two steps in this pipeline are slow and neither is where the science lives:

1. **Running the language models.** Embedding 15,406 words *in context* takes a
   couple of minutes per model on a GPU — and close to an hour per model on the
   2-vCPU machine a free Colab session usually gives you. With eleven models
   that is a non-starter.
2. **Fitting the ridge.** 25,870 voxels x 9 folds is 40-120 s per configuration
   on a 16-core box, several times that on a small VM, and there are fourteen
   configurations.

So the workshop downloads both: the HRF-convolved design matrices (41 MB) and
the per-fold encoding maps (12 MB), from the `features-v1` release. Section 4
still fits one model live, so the regression is not a black box, and every
number the notebook *reports* — lateralization, across-run intervals, trend
tests, every figure — is computed locally from those maps.

The precomputed path reproduces the full GPU pipeline to five decimal places
(p=0.00056 and p=0.0034 for the two training-axis trends, identical either way).
Two switches in section 3 turn the shortcuts off:

```python
USE_PRECOMPUTED = False    # run the language models yourself (wants a GPU)
FIT_SWEEPS_LIVE = True     # refit the ridge for all 14 configurations
```

Regenerate the packs with `tools_dump_features.py` and `tools_dump_folds.py`.

## What is in here

| path | what |
|---|---|
| `lppws/study.py` | `Li2022PetitAverage` — the average subject as a `neuralset` `Study`, plus the data downloader |
| `lppws/pipeline.py` | masking, HRF-convolved LLM features, leave-one-run-out ridge, lateralization metrics |
| `notebooks/lpp_lateralization.ipynb` | the guided workshop notebook |
| `verify.py` | headless end-to-end run of any model list |
| `verify2.py` | same, keeping the per-run maps (for the across-run error bar) |
| `RESULTS.md` | what the pipeline actually produced here, with the numbers |
| `results/` | the raw measurements behind `RESULTS.md` |
| `lppws/viz.py` | the word-timing figure (words on the fMRI sampling grid) |
| `lppws/cached.py` | download + load the precomputed feature and result packs |
| `tools_dump_*.py` | rebuild those packs from scratch |
| `notebooks/lpp_lateralization.executed.ipynb` | the same notebook with all outputs, as a reference run |

`Li2022PetitAverage` lives here rather than inside `neuralfetch` because the
published `neuralfetch` only ships the full 112-subject OpenNeuro study
(`Li2022Petit`) and a one-subject sample (`Li2022PetitSample`). It subclasses
`Li2022Petit`, so everything else in `neuralset` treats it as a normal study.

## The data

One pseudo-subject `sub-avgEN`, 9 runs. Per run: ~1700 `Word` events with
onsets from the TextGrid annotation, and one 4D BOLD file (TR = 2 s) that is
the voxel-wise average of the 49 English-speaking participants of
[ds003643](https://openneuro.org/datasets/ds003643). Averaging across subjects
is what makes a laptop-scale workshop possible — it removes most of the noise,
at the cost of any subject-level error bar (see *Caveats*).

The brain mask holds **25,870 voxels**, exactly symmetric: 12,425 left,
12,425 right, and 100% of them have a mirror homolog in the other hemisphere.
`roi_masks/` gives 7 left-hemisphere language-network ROIs (BA44, BA45, BA47,
aSTS, pSTS, TP, AG/TPJ) totalling 430 voxels.

## Caveats worth stating out loud in the workshop

- **There is one subject.** Every confidence interval here is a bootstrap over
  *voxels*, not over subjects. Neighbouring voxels are strongly correlated, so
  these intervals are anti-conservative: treat them as a display of spread, not
  as a significance test. A real test needs the per-subject data.
- **The stimulus audio in the repository is French** (and only section 1),
  while the fMRI is from English listeners. It is therefore not loaded by
  default (`Li2022PetitAverage(load_audio=False)`). Do not use it as an
  acoustic regressor.
- **Ridge alphas are selected inside the training runs only.** Selecting the
  penalty on the fold you evaluate on inflates encoding scores substantially.
- **Static vs contextual embeddings are not the same experiment.** With
  `contextualized=False` each occurrence of a word gets the same vector; the
  paper's claim is about models that read the word *in context*. The notebook
  runs both so the difference is visible.
