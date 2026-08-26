"""Dump the per-fold encoding maps for every config in the feature pack.

The ridge itself is the second bottleneck: ~40-120 s per configuration on a
16-core box, several times that on a 2-vCPU Colab. Shipping the 9 held-out-run
maps lets the notebook run the regression live once (so attendees see it) and
load the rest, while every statistic downstream is still computed on their
machine from these maps.
"""
import sys
from pathlib import Path
import numpy as np
from lppws.study import Li2022PetitAverage
from lppws import pipeline as pl, cached

D = Path('/home/co/tmp/li-avg/lpp_average_subject_en')
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else 'lpp_folds_v1.npz')

ev = Li2022PetitAverage(path=str(D), query='timeline_index < 9').run()
tls = list(dict.fromkeys(ev['timeline']))
brain = pl.load_brain(next(D.glob('**/mask_lpp_en.nii.gz')))
bold = {t: pl.load_bold(brain, ev, t) for t in tls}
F = cached.load_features(tls, cache_dir='/home/co/tmp')

out = {}
for tag in cached.TAGS:
    folds = pl.encode_corr_from_X(F[tag], tls, bold, per_fold=True)
    out[tag] = folds.astype(np.float32)
    print(f"{tag:18s} {folds.shape}  mean r {folds.mean():+.5f}", flush=True)

np.savez_compressed(OUT, **out)
print("wrote", OUT, f"{OUT.stat().st_size/1e6:.1f} MB")
