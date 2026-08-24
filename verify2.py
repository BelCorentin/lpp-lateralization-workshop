"""Per-fold runs (honest error bar) + a layer probe for the gpt2 anomaly."""
import json, sys
from pathlib import Path
import numpy as np, torch
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
ctx_events = pl.add_running_context(events, timelines)

out = {}
for tag, name, ctx, layers in json.loads(sys.argv[1]):
    ev = ctx_events if ctx else events
    h = pl.hf_features(ev, name, contextualized=ctx, layers=layers, device=DEV, cache_dir=CACHE)
    rf = pl.encode_corr(h, ev, timelines, bold, per_fold=True)
    np.save(CACHE / f'folds_{tag}.npy', rf)
    r = rf.mean(0)
    row = pl.metrics(r, brain, lang)
    row |= {f'run_{k}': v for k, v in pl.across_run_ci(rf, brain, lang).items()}
    row |= {f'glob_{k}': v for k, v in pl.across_run_ci(rf, brain).items()}
    out[tag] = row
    print(tag, json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    del h; torch.cuda.is_available() and torch.cuda.empty_cache()
Path('verify2_results.json').write_text(json.dumps(out, indent=2))
