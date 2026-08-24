"""End-to-end check: does the left-lateralization result hold?"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from lppws.study import Li2022PetitAverage
from lppws import pipeline as pl

D = Path('/home/co/tmp/li-avg/lpp_average_subject_en')
CACHE = Path('/home/co/tmp/lppws_cache')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

study = Li2022PetitAverage(path=str(D), query='timeline_index < 9')
events = study.run()
timelines = list(dict.fromkeys(events['timeline']))
brain = pl.load_brain(next(D.glob('**/mask_lpp_en.nii.gz')))
bold = {tl: pl.load_bold(brain, events, tl) for tl in timelines}
rois = pl.roi_membership(next(D.glob('**/roi_masks')), brain)
lang = np.zeros(brain.n_voxels, bool)
for m in rois.values(): lang |= m
print(f'voxels={brain.n_voxels} L={brain.left.sum()} R={brain.right.sum()} '
      f'paired={brain.valid.mean():.0%} langROI={lang.sum()}', flush=True)

ctx_events = pl.add_running_context(events, timelines)

MODELS = json.loads(sys.argv[1]) if len(sys.argv) > 1 else [
    ["gpt2-static", "openai-community/gpt2", False],
]
out = {}
for tag, name, ctx in MODELS:
    t0 = time.time()
    ev = ctx_events if ctx else events
    hrf = pl.hf_features(ev, name, contextualized=ctx, device=DEV, cache_dir=CACHE)
    r = pl.encode_corr(hrf, ev, timelines, bold)
    m = pl.metrics(r, brain, lang)
    rl, rr, li = pl.hemisphere_means(r, brain)
    m.update(L=rl, R=rr, LI_norm=li, secs=round(time.time()-t0))
    out[tag] = m
    np.save(f'/home/co/tmp/lppws_cache/r_{tag}.npy', r)
    print(tag, {k: (round(v,5) if isinstance(v,float) else v) for k,v in m.items()}, flush=True)
    del hrf
    torch.cuda.empty_cache()
Path('verify_results.json').write_text(json.dumps(out, indent=2))
