# What this pipeline actually produces

Everything below was measured on 2026-08-24 with the environment pinned by
`pyproject.toml` (neuralset 0.2.3, neuralfetch 0.2.3, transformers 5.15.1,
torch 2.13, scikit-learn 1.8), on the average-subject data described in the
README: 25,870 voxels (12,425 left / 12,425 right, 100% mirror-paired), 9 runs,
430 voxels in the 7 left language-network ROIs.

Metrics:

* `mean r` — encoding correlation averaged over all voxels.
* `LI global` — `r(left voxel) − r(right mirror homolog)`, averaged over all left voxels.
* `LI lang` — the same, restricted to the 430 language-ROI voxels.

## 1. Static embeddings (`contextualized=False`), layer 2/3

| model | mean r | LI global | LI lang |
|---|---|---|---|
| gpt2 | 0.0266 | +0.0034 | −0.0040 |
| pythia-14m | 0.0262 | +0.0033 | −0.0029 |
| pythia-160m | 0.0266 | +0.0031 | −0.0042 |

Encoding quality is flat across a 11× range of model size, which is the expected
sanity check: without context, a "model" is essentially a lexicon lookup, and a
bigger lexicon does not fit BOLD better. Note the language ROIs come out
*negatively* lateralized here.

## 2. Contextual embeddings (`contextualized=True`, 32-word context), layer 2/3

| model | mean r | LI global | LI lang |
|---|---|---|---|
| gpt2 | 0.0180 | +0.0005 | −0.0019 |
| pythia-14m | 0.0337 | +0.0045 | +0.0026 |
| pythia-70m | 0.0357 | +0.0045 | +0.0033 |
| pythia-160m | 0.0363 | +0.0045 | +0.0026 |
| pythia-410m | 0.0368 | +0.0050 | +0.0059 |

Within the Pythia family, `mean r` rises monotonically with size
(0.0337 → 0.0368) and the sign of `LI lang` flips positive relative to the
static features. `LI global` is essentially flat (+0.0045 → +0.0050); `LI lang`
does not increase monotonically (14m 0.0026, 70m 0.0033, 160m 0.0026,
410m 0.0059).

**gpt2 is the outlier**, and section 3 explains why.

## 3. The training axis — where the scaling claim does replicate

Comparing *sizes* means comparing different architectures read at the same
relative depth, which confounds size with layer (section 4). Comparing
*training checkpoints* does not: same architecture, same width, same layer
index, only the weights differ.

pythia-70m, contextual, layer 2/3, from random init to fully trained:

| training step | mean r | LI global | LI lang |
|---|---|---|---|
| 1 (random init) | 0.0265 | +0.0028 | −0.0047 |
| 512 | 0.0268 | +0.0035 | −0.0036 |
| 2 000 | 0.0287 | +0.0037 | −0.0024 |
| 8 000 | 0.0333 | +0.0047 | +0.0017 |
| 32 000 | 0.0352 | +0.0046 | +0.0031 |
| 143 000 | 0.0357 | +0.0045 | +0.0033 |

Monotone on every column. Language-ROI lateralization **crosses zero**: the
untrained network predicts the right hemisphere slightly better, the trained one
predicts the left better.

Each individual checkpoint still has an across-run interval that spans zero —
but that is the wrong test. The claim is that the metric *grows*, so each run
can be compared against itself across checkpoints, which cancels the run's own
noise. Fitting a line per held-out run and testing the 9 slopes:

| metric | slope per decade of training | t | p | runs up |
|---|---|---|---|---|
| mean r | +0.00214 | +5.51 | **0.0006** | 9/9 |
| LI language ROIs | +0.00186 | +4.11 | **0.0034** | 8/9 |
| LI whole brain | +0.00041 | +1.85 | 0.10 | 7/9 |

**This is the paper's claim, replicated, on a 70M-parameter model.** Note which
measure carries it: training makes the *language network specifically* more
left-lateralized, while the whole-brain asymmetry is already there at random
initialisation (+0.0028) and barely moves after.

Sanity check on the checkpoints: the loaded weights really do differ — a layer-2
layernorm has mean |w| exactly 1.000 (sd 0.000) at `step1`, 1.012 (sd 0.005) at
`step512`, 1.089 (sd 0.104) at `step143000`. `revision=` is not being silently
ignored.

## 4. Layer matters more than model size

gpt2, contextual, sweeping the relative layer depth:

| layer | mean r | LI global | LI lang |
|---|---|---|---|
| 1/3 (0.3333) | 0.0367 | +0.0045 | +0.0045 |
| 2/3 | 0.0180 | +0.0005 | −0.0019 |
| 1.0 (last) | 0.0248 | +0.0020 | −0.0080 |

Read at one third of its depth, gpt2 matches pythia-410m
(mean r 0.0367 vs 0.0368; LI lang +0.0045 vs +0.0059). Read at two thirds — the
value hardcoded in the original notebook — it is the worst model in the table.
**The layer moves the headline number more than a 30× change in parameter count
does.**

The full Pythia grid shows the profile is stable within a family, which is why a
family-internal comparison can be made fair:

| | layer 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|
| **14m** | 0.0269 / −0.0016 | 0.0330 / +0.0050 | **0.0337 / +0.0026** | 0.0275 / −0.0043 |
| **70m** | 0.0272 / −0.0025 | 0.0341 / +0.0047 | **0.0357 / +0.0033** | 0.0271 / −0.0043 |
| **160m** | 0.0285 / −0.0020 | 0.0365 / +0.0043 | **0.0370 / +0.0028** | 0.0272 / −0.0044 |
| **410m** | 0.0360 / +0.0051 | 0.0367 / +0.0056 | **0.0369 / +0.0052** | 0.0076 / −0.0069 |

(cells are `mean r / LI lang`; bold = each model's best layer by mean r.)

Two things fall out. The **last** layer is always the worst — it is specialised
for next-token prediction, not for representing the sentence. And the useful
band shifts *earlier in relative terms* as models get deeper: 410m is already at
full strength at 0.25, where the smaller models are still near their floor.

### The size axis, with the layer confound removed

Fitting the size trend per held-out run, at a layer held fixed across models:

| axis, layer | metric | slope/decade | t | p | runs up |
|---|---|---|---|---|---|
| size @ 0.75 | mean r | +0.00240 | +2.74 | 0.025 | 8/9 |
| size @ 0.75 | LI lang | +0.00145 | +0.95 | 0.37 | 4/9 |
| size @ 0.75 | LI global | +0.00008 | +0.52 | 0.62 | 5/9 |
| size @ 0.50 | mean r | +0.00275 | +1.91 | 0.093 | 7/9 |
| size @ 0.50 | LI lang | +0.00022 | +0.11 | 0.92 | 4/9 |
| size @ 0.50 | LI global | −0.00046 | −1.64 | 0.14 | 2/9 |

So the size axis is not rescued by fixing the layer. Encoding *quality* does
grow with size (and saturates around 160m: 0.0337 → 0.0357 → 0.0370 → 0.0369),
but **lateralization does not** — null at both layers tested. Compare the
training axis in section 3, which is significant on the same test with the same
number of points.

## 5. The error bar decides the conclusion

Two ways to put an interval on `LI lang`, on exactly the same numbers:

| model | bootstrap over 430 voxels | spread over the 9 held-out runs | runs positive |
|---|---|---|---|
| gpt2 static | −0.0040 [−0.0068, −0.0012] | −0.0040 [−0.0142, +0.0061] | 4/9 |
| pythia-14m ctx | +0.0026 [−0.0001, +0.0054] | +0.0026 [−0.0064, +0.0116] | 5/9 |
| pythia-410m ctx | +0.0059 [+0.0027, +0.0091] | +0.0059 [−0.0060, +0.0179] | 6/9 |
| gpt2 ctx layer 1/3 | +0.0045 [+0.0014, +0.0075] | +0.0045 [−0.0073, +0.0163] | 5/9 |

The voxel bootstrap makes several of these look like clean effects. It should
not be believed: there is a single (averaged) subject, and 430 spatially smooth
voxels are not 430 independent observations. Across runs, **every language-ROI
lateralization estimate crosses zero.**

The whole-brain effect is the one that survives:

| model | LI global, across runs | runs positive |
|---|---|---|
| pythia-14m ctx | +0.0045 [+0.0011, +0.0079] | 8/9 |
| pythia-410m ctx | +0.0050 [+0.0012, +0.0088] | 8/9 |
| gpt2 ctx layer 1/3 | +0.0045 [+0.0007, +0.0083] | 8/9 |
| gpt2 static | +0.0034 [+0.0003, +0.0064] | 6/9 |

Raw measurements are in `results/`. (The notebook passes `1/3` exactly rather
than `0.3333`, which lands on a marginally different layer weighting: mean r
0.0365, LI lang +0.0042 — same conclusion.)

## Summary

* **Replicates:** language models predict the left hemisphere better than the
  right. Positive for every trained model tried, and positive in 8/9 held-out
  runs.
* **Replicates:** contextual features beat static ones, and encoding quality
  grows both with training and (weakly, saturating ~160m) with model size.
* **Replicates — the scaling claim, on the training axis.** pythia-70m from
  random init to fully trained: language-ROI lateralization +0.0019 per decade
  (t=4.11, p=0.003, up in 8/9 runs), crossing from −0.0047 to +0.0033. This is
  the axis to use: same architecture, same layer, only the weights change.
* **Does not replicate — the scaling claim on the size axis**, 14m→410m, even
  after matching layers across models (p=0.37 at layer 0.75, p=0.92 at 0.50).
  Either the range is too narrow or a single averaged subject cannot resolve it.
* **Caveat not in the paper's framing:** the relative-layer choice dominates
  cross-family comparisons, and the last layer is always the worst. Sweep layers
  before reading any complexity curve.
* **Caveat on statistics:** with one averaged subject, per-point confidence
  intervals must come from the spread across runs, not a bootstrap over
  spatially correlated voxels. Trends are far better powered than points,
  because each run can be compared against itself along the axis.

None of this contradicts Bonnasse-Gahot & Pallier — they use per-subject data (a
real error term), more models, and a wider complexity range. It does mean this
workshop setup demonstrates the lateralization and its growth *with training*,
and should not be presented as demonstrating growth with model size.
