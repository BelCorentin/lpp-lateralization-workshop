"""Encoding pipeline: word features -> HRF -> ridge -> per-voxel r -> lateralization.

Small, dependency-light helpers shared by the workshop notebook. Everything
operates on the event DataFrame produced by
:class:`lppws.study.Li2022PetitAverage`.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker
from sklearn.linear_model import RidgeCV

import neuralset as ns

TR = 2.0  #: fMRI repetition time (s)
FMRI_FREQ = 1.0 / TR
INNER_FREQ = 10.0  #: sampling rate of the word-feature signal *before* HRF convolution


# --------------------------------------------------------------------------
# fMRI target
# --------------------------------------------------------------------------
@dataclass
class Brain:
    """The masked fMRI target plus the geometry needed for hemisphere analyses."""

    masker: NiftiMasker
    mask_img: nib.Nifti1Image
    ijk: np.ndarray  #: (n_vox, 3) voxel indices, in masker order
    world: np.ndarray  #: (n_vox, 3) MNI coordinates
    left: np.ndarray  #: (n_vox,) bool
    right: np.ndarray  #: (n_vox,) bool
    pair: np.ndarray  #: (n_vox,) index of the mirror homolog, -1 if none
    valid: np.ndarray  #: (n_vox,) bool, True where `pair` is defined

    @property
    def n_voxels(self) -> int:
        return self.left.size


def load_brain(mask_path: str | Path) -> Brain:
    """Build the masker and the left/right + mirror-homolog geometry."""
    mask_path = str(mask_path)
    masker = NiftiMasker(
        mask_img=mask_path, detrend=True, standardize=True, high_pass=1 / 128, t_r=TR
    ).fit()
    mask_img = nib.load(mask_path)
    ijk = np.argwhere(mask_img.get_fdata() > 0)  # masker voxel order
    world = nib.affines.apply_affine(mask_img.affine, ijk)

    # Map each voxel to its left/right mirror homolog (the MNI grid is symmetric in x).
    inv = np.linalg.inv(mask_img.affine)
    mirrored = world.copy()
    mirrored[:, 0] *= -1
    mijk = np.rint(nib.affines.apply_affine(inv, mirrored)).astype(int)
    lut = {tuple(v): i for i, v in enumerate(ijk)}
    pair = np.array([lut.get(tuple(v), -1) for v in mijk])
    return Brain(
        masker=masker,
        mask_img=mask_img,
        ijk=ijk,
        world=world,
        left=world[:, 0] < 0,
        right=world[:, 0] > 0,
        pair=pair,
        valid=pair >= 0,
    )


def load_bold(brain: Brain, events: pd.DataFrame, timeline: str) -> np.ndarray:
    """(n_TR, n_voxels) detrended/standardized BOLD for one run."""
    fp = events[(events.timeline == timeline) & (events.type == "Fmri")]["filepath"].iloc[0]
    return brain.masker.transform(fp)


def roi_membership(roi_dir: str | Path, brain: Brain) -> dict[str, np.ndarray]:
    """Boolean masks (in masker voxel order) for every ROI nifti in ``roi_dir``.

    The ROIs ship on the same grid as the brain mask, so a direct boolean index
    into the mask is enough.
    """
    inside = brain.mask_img.get_fdata() > 0
    return {
        p.name.split(".")[0]: (nib.load(p).get_fdata()[inside] > 0)
        for p in sorted(Path(roi_dir).glob("*.nii.gz"))
    }


# --------------------------------------------------------------------------
# Predictors
# --------------------------------------------------------------------------
def hf_features(
    events: pd.DataFrame,
    model_name: str,
    *,
    layers: float = 2 / 3,
    contextualized: bool = False,
    device: str = "cpu",
    cache_dir: str | Path | None = None,
    **hf_kwargs: tp.Any,
) -> tp.Any:
    """HRF-convolved LLM word embeddings, as a prepared neuralset extractor.

    ``layers`` is a *relative* depth in [0, 1]. Extra keyword arguments are
    forwarded to :class:`neuralset.extractors.HuggingFaceText` (e.g.
    ``hf_config=`` to pin a training-step checkpoint).
    """
    infra = dict(folder=Path(cache_dir)) if cache_dir else {}
    text = ns.extractors.HuggingFaceText(
        model_name=model_name,
        frequency=INNER_FREQ,
        contextualized=contextualized,
        aggregation="mean",  # combine words falling in the same time bin
        layers=layers,
        layer_aggregation="mean",
        device=device,
        infra=infra,
        **hf_kwargs,
    )
    hrf = ns.extractors.HrfConvolve(extractor=text, frequency=FMRI_FREQ, infra=infra)
    hrf.prepare(events)
    return hrf


def spacy_features(
    events: pd.DataFrame,
    *,
    language: str = "english",
    cache_dir: str | Path | None = None,
) -> tp.Any:
    """Static-embedding baseline via spaCy vectors (needs ``en_core_web_md``)."""
    infra = dict(folder=Path(cache_dir)) if cache_dir else {}
    emb = ns.extractors.SpacyEmbedding(
        language=language, frequency=INNER_FREQ, aggregation="mean", infra=infra
    )
    hrf = ns.extractors.HrfConvolve(extractor=emb, frequency=FMRI_FREQ, infra=infra)
    hrf.prepare(events)
    return hrf


def design_matrix(hrf: tp.Any, events: pd.DataFrame, timeline: str) -> np.ndarray:
    """(n_TR, n_features) HRF-convolved predictors for one run."""
    tle = events[events.timeline == timeline]
    dur = float(tle[tle.type == "Fmri"]["duration"].iloc[0])
    return np.asarray(hrf(tle, start=0.0, duration=dur)).T


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------
ALPHAS = np.logspace(2, 7, 8)


def _align(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(x), len(y))
    return x[:n], y[:n]


def _corr(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pred = (pred - pred.mean(0)) / (pred.std(0) + 1e-8)
    true = (true - true.mean(0)) / (true.std(0) + 1e-8)
    return (pred * true).mean(0)


def encode_corr(
    hrf: tp.Any,
    events: pd.DataFrame,
    timelines: list[str],
    bold: dict[str, np.ndarray],
    *,
    alphas: np.ndarray = ALPHAS,
    per_fold: bool = False,
) -> np.ndarray:
    """Leave-one-run-out ridge encoding; per-voxel Pearson r.

    Returns ``(n_voxels,)`` averaged over folds, or ``(n_runs, n_voxels)`` when
    ``per_fold=True`` — the per-fold maps are what you need for an error bar
    that is not a bootstrap over (correlated) voxels.

    The ridge penalty is selected by RidgeCV *inside* the training runs only, so
    the held-out run is never used for model selection.
    """
    X = {tl: design_matrix(hrf, events, tl) for tl in timelines}
    r = np.zeros((len(timelines), next(iter(bold.values())).shape[1]))
    for i, test_tl in enumerate(timelines):
        xs, ys = zip(*(_align(X[tl], bold[tl]) for tl in timelines if tl != test_tl))
        model = RidgeCV(alphas=alphas).fit(np.vstack(xs), np.vstack(ys))
        xte, yte = _align(X[test_tl], bold[test_tl])
        r[i] = _corr(model.predict(xte), yte)
    return r if per_fold else r.mean(0)


# --------------------------------------------------------------------------
# Lateralization
# --------------------------------------------------------------------------
def hemisphere_means(r: np.ndarray, brain: Brain) -> tuple[float, float, float]:
    """Mean r in each hemisphere and the normalized lateralization index."""
    rl, rr = float(r[brain.left].mean()), float(r[brain.right].mean())
    return rl, rr, (rl - rr) / (abs(rl) + abs(rr) + 1e-8)


def bootstrap_ci(d: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of ``d`` with a percentile bootstrap 95% CI over its entries."""
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, d.size, size=(n, d.size))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def paired_lateralization(
    r: np.ndarray, brain: Brain, subset: np.ndarray | None = None
) -> np.ndarray:
    """r(left voxel) - r(its right mirror homolog), over left voxels.

    This paired contrast cancels any global shift in encoding quality, which the
    raw hemisphere means do not.
    """
    sel = brain.left & brain.valid
    if subset is not None:
        sel = sel & subset
    return r[sel] - r[brain.pair[sel]]


def metrics(r: np.ndarray, brain: Brain, lang_mask: np.ndarray) -> dict[str, float]:
    """Summary row: overall fit, global lateralization, language-ROI lateralization."""
    mean_li, lo, hi = bootstrap_ci(paired_lateralization(r, brain, lang_mask))
    return dict(
        mean_r=float(r.mean()),
        LI_global=float(paired_lateralization(r, brain).mean()),
        LI_lang=mean_li,
        LI_lang_lo=lo,
        LI_lang_hi=hi,
    )


def add_running_context(
    events: pd.DataFrame, timelines: list[str], n_words: int = 32
) -> pd.DataFrame:
    """Fill the ``context`` column of each Word with the last ``n_words`` words.

    ``HuggingFaceText(contextualized=True)`` embeds a word *inside* its context
    string; without this the column is empty and every word gets a static,
    context-free vector.
    """
    events = events.copy()
    ctx = events["context"].astype(object)
    for tl in timelines:
        idx = events.index[(events.timeline == tl) & (events.type == "Word")]
        buf: list[str] = []
        for i in idx:
            buf.append(events.at[i, "text"])
            ctx.at[i] = " ".join(buf[-n_words:])
    events["context"] = ctx
    return events


def across_run_ci(
    r_folds: np.ndarray, brain: Brain, subset: np.ndarray | None = None
) -> dict[str, float]:
    """Mean paired lateralization with a spread taken *across runs*.

    ``r_folds`` is the ``(n_runs, n_voxels)`` output of ``encode_corr(...,
    per_fold=True)``. For each held-out run we average
    ``r(left) - r(right homolog)`` over the voxels of interest, then report the
    mean and a t-based 95% interval over those n_runs values.

    This is a far more honest error bar than a bootstrap over voxels, which
    treats 430 spatially smooth voxels as 430 independent observations. It is
    still not a clean test — leave-one-run-out folds share 8/9 of their training
    data — so read it as "how much does this move between runs", not as a
    p-value.
    """
    from scipy import stats

    per_run = np.array(
        [paired_lateralization(r, brain, subset).mean() for r in r_folds]
    )
    n = per_run.size
    sem = per_run.std(ddof=1) / np.sqrt(n)
    half = stats.t.ppf(0.975, n - 1) * sem
    return dict(
        mean=float(per_run.mean()),
        lo=float(per_run.mean() - half),
        hi=float(per_run.mean() + half),
        n_runs=n,
        runs_positive=int((per_run > 0).sum()),
    )


def trend_test(
    folds_by_x: dict[float, np.ndarray],
    brain: Brain,
    *,
    subset: np.ndarray | None = None,
    metric: str = "LI",
    log_x: bool = True,
) -> dict[str, tp.Any]:
    """Does a metric grow along a complexity axis? Tested across held-out runs.

    ``folds_by_x`` maps the axis value (training step, parameter count, ...) to
    that model's ``(n_runs, n_voxels)`` per-fold map. For each run we fit a line
    through its own points and test the 9 resulting slopes against zero.

    This is much more powerful than asking whether each point separately differs
    from zero, because every run contributes a *paired* comparison across the
    axis: the run's own noise cancels out of its slope. It is the right test for
    a claim of the form "X grows with model complexity".

    ``metric`` is ``"LI"`` (paired lateralization over ``subset``) or
    ``"mean_r"`` (overall encoding quality).
    """
    from scipy import stats

    xs = sorted(folds_by_x)
    x = np.log10(np.asarray(xs, float) + 1) if log_x else np.asarray(xs, float)
    n_runs = next(iter(folds_by_x.values())).shape[0]

    curve = np.zeros((n_runs, len(xs)))
    for j, xv in enumerate(xs):
        for i in range(n_runs):
            r = folds_by_x[xv][i]
            curve[i, j] = (
                r.mean() if metric == "mean_r" else paired_lateralization(r, brain, subset).mean()
            )

    slopes = np.array([np.polyfit(x, curve[i], 1)[0] for i in range(n_runs)])
    t, p = stats.ttest_1samp(slopes, 0)
    return dict(
        x=xs,
        curve=curve.mean(0),
        slope=float(slopes.mean()),
        t=float(t),
        p=float(p),
        runs_positive=int((slopes > 0).sum()),
        n_runs=n_runs,
    )
