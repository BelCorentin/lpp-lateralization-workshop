"""Complexity axes that are not confounded with layer choice.

Usage:  python verify3.py steps   # training-step sweep, one small model
        python verify3.py layers  # size x layer grid
"""
import json, sys
from pathlib import Path
import numpy as np, torch, pandas as pd
from neuralset.extractors.text import HuggingFaceTextConfig
from lppws.study import Li2022PetitAverage
from lppws import pipeline as pl

D = Path('/home/co/tmp/li-avg/lpp_average_subject_en'); CACHE = Path('/home/co/tmp/lppws_cache')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
study = Li2022PetitAverage(path=str(D), query='timeline_index < 9')
events = study.run(); timelines = list(dict.fromkeys(events['timeline']))
brain = pl.load_brain(next(D.glob('**/mask_lpp_en.nii.gz')))
bold = {tl: pl.load_bold(brain, events, tl) for tl in timelines}
rois = pl.roi_membership(next(D.glob('**/roi_masks')), brain)
lang = np.zeros(brain.n_voxels, bool)
for m in rois.values(): lang |= m
ctx = pl.add_running_context(events, timelines)


def cell(tag, model, layers, revision=None):
    f = CACHE / f'folds_{tag}.npy'
    if f.exists():
        rf = np.load(f)
    else:
        kw = {}
        if revision:
            kw['hf_config'] = HuggingFaceTextConfig(model_kwargs={'revision': revision})
        h = pl.hf_features(ctx, model, contextualized=True, layers=layers,
                           device=DEV, cache_dir=CACHE, **kw)
        rf = pl.encode_corr(h, ctx, timelines, bold, per_fold=True)
        np.save(f, rf)
        del h
        torch.cuda.is_available() and torch.cuda.empty_cache()
    r = rf.mean(0)
    row = pl.metrics(r, brain, lang)
    row |= {f'run_{k}': v for k, v in pl.across_run_ci(rf, brain, lang).items()}
    row |= {f'glob_{k}': v for k, v in pl.across_run_ci(rf, brain).items()}
    print(tag, json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    return row


mode = sys.argv[1]
out = {}
if mode == 'steps':
    MODEL = 'EleutherAI/pythia-70m'
    for st in [1, 512, 2000, 8000, 32000, 143000]:
        out[st] = cell(f'p70m-step{st}', MODEL, 2/3, revision=f'step{st}')
elif mode == 'layers':
    SIZES = {'14m': 'EleutherAI/pythia-14m', '70m': 'EleutherAI/pythia-70m',
             '160m': 'EleutherAI/pythia-160m', '410m': 'EleutherAI/pythia-410m'}
    for s, name in SIZES.items():
        for lay in [0.25, 0.5, 0.75, 1.0]:
            out[f'{s}@{lay}'] = cell(f'p{s}-L{int(lay*100)}', name, lay)
Path(f'results/{mode}.json').write_text(json.dumps(out, indent=2))
