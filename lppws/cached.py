"""Precomputed design matrices, so the workshop does not need a GPU.

Extracting contextual embeddings means one forward pass per batch of words for
every model — a few minutes on a GPU, but close to an hour *per model* on the
2-vCPU machine a free Colab session usually hands you. That is fine for a
research run and hopeless for a 90-minute workshop.

This module downloads the same matrices, already convolved with the HRF and
resampled to the fMRI TR, as one 41 MB archive. Everything downstream — the
ridge regression, the lateralization, the statistics — still runs live on the
attendee's machine; only the language-model forward passes are skipped.

Stored as float16, which is lossless at the precision that matters here: the
encoding correlations agree with the float32 pipeline to better than 1e-4.
"""

from __future__ import annotations

import typing as tp
import urllib.request
from pathlib import Path

import numpy as np

VERSION = "features-v1"
_BASE = (
    "https://github.com/BelCorentin/lpp-lateralization-workshop/"
    f"releases/download/{VERSION}/"
)
#: HRF-convolved design matrices, one per (model config, run). 41 MB.
FEATURES_URL = _BASE + "lpp_features_v1.npz"
#: Per-fold encoding maps, ``(9 runs, n_voxels)`` per config. ~13 MB.
FOLDS_URL = _BASE + "lpp_folds_v1.npz"

#: Tags available in the pack, and what they are.
TAGS = {
    "gpt2-static": "gpt2, non-contextual (a lexicon lookup), layer 2/3",
    "size-14m": "pythia-14m, contextual, layer 2/3",
    "size-70m": "pythia-70m, contextual, layer 2/3",
    "size-160m": "pythia-160m, contextual, layer 2/3",
    "size-410m": "pythia-410m, contextual, layer 2/3",
    "step-1": "pythia-70m at random initialisation",
    "step-512": "pythia-70m after 512 training steps",
    "step-2000": "pythia-70m after 2 000 training steps",
    "step-8000": "pythia-70m after 8 000 training steps",
    "step-32000": "pythia-70m after 32 000 training steps",
    "step-143000": "pythia-70m fully trained",
    "gpt2-layer-33": "gpt2 contextual, layer 1/3",
    "gpt2-layer-66": "gpt2 contextual, layer 2/3",
    "gpt2-layer-100": "gpt2 contextual, last layer",
}


def download(cache_dir: str | Path = "cache", url: str = FEATURES_URL) -> Path:
    """Fetch the feature pack once into ``cache_dir``; returns its path."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / Path(url).name
    if dest.exists():
        return dest
    tmp = dest.with_suffix(".part")
    print(f"downloading precomputed features (~41 MB) -> {dest}")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def load_features(
    timelines: tp.Sequence[str],
    *,
    cache_dir: str | Path = "cache",
    url: str = FEATURES_URL,
) -> dict[str, dict[str, np.ndarray]]:
    """Return ``{tag: {timeline: (n_TR, n_features) float32}}``.

    ``timelines`` must be the study's timeline list, in order — the pack stores
    runs by index, and this maps them back onto the names your events use.
    """
    path = download(cache_dir, url)
    with np.load(path) as z:
        keys = list(z.keys())
        out: dict[str, dict[str, np.ndarray]] = {}
        for k in keys:
            tag, idx = k.rsplit("|", 1)
            out.setdefault(tag, {})[timelines[int(idx)]] = z[k].astype(np.float32)
    missing = [t for t, d in out.items() if len(d) != len(timelines)]
    if missing:
        raise RuntimeError(f"incomplete feature pack for {missing}")
    return out


def load_folds(
    *, cache_dir: str | Path = "cache", url: str = FOLDS_URL
) -> dict[str, np.ndarray]:
    """Return ``{tag: (9, n_voxels) float32}`` — the leave-one-run-out maps.

    These are the *output* of the ridge, not its input. Fitting one
    configuration takes 40-120 s on a 16-core machine and several times that on
    a small cloud VM; with fourteen of them the sweeps alone would outlast a
    workshop. Loading the maps skips only the regression — every statistic the
    notebook reports (lateralization, across-run intervals, trend tests) is
    recomputed from them locally, and section 4 still fits one model live so the
    regression is not a black box.
    """
    path = download(cache_dir, url)
    with np.load(path) as z:
        return {k: z[k].astype(np.float32) for k in z.keys()}
