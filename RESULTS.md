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

## 3. Layer matters more than model size

gpt2, contextual, sweeping the relative layer depth:

| layer | mean r | LI global | LI lang |
|---|---|---|---|
| 1/3 | 0.0367 | +0.0045 | +0.0045 |
| 2/3 | 0.0180 | +0.0005 | −0.0019 |
| 1.0 (last) | 0.0248 | +0.0020 | −0.0080 |

Read at one third of its depth, gpt2 matches pythia-410m
(mean r 0.0367 vs 0.0368; LI lang +0.0045 vs +0.0059). Read at two thirds — the
value hardcoded in the original notebook — it is the worst model in the whole
table. **The choice of layer moves the headline number more than a 30× change
in parameter count does.** Any "complexity" axis measured at one fixed relative
depth is confounded with where in the network that depth happens to land.

## 4. The error bar decides the conclusion

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

## Summary

* **Replicates:** language models predict the left hemisphere better than the
  right. The whole-brain paired effect is positive for every model tried and
  positive in 8 of 9 held-out runs for the contextual ones.
* **Replicates:** contextual features beat static ones, and within a family,
  encoding quality grows with model size.
* **Does not replicate at this scale:** lateralization *growing* with model
  complexity. `LI global` is flat from 14m to 410m, and `LI lang` is
  non-monotonic and never clears a run-level error bar.
* **New caveat, not in the paper's framing:** the relative-layer choice
  dominates. Before reading a size or training-step curve, sweep layers.

None of this contradicts Bonnasse-Gahot & Pallier — they use per-subject data
(a real error term), more models, and a wider complexity range. It does mean
this workshop setup can demonstrate *the lateralization*, but should not be
presented as demonstrating *the scaling*.
