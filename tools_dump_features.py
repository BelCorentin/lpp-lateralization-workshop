"""Dump every design matrix the notebook needs into one compressed archive.

Reads the local feature cache (so no model forward passes happen if the
embeddings are already there) and writes float16 design matrices, one per
(model config, run). Uploaded as a release asset so the notebook can skip the
expensive extraction entirely.
"""
import sys
from pathlib import Path
import numpy as np
from neuralset.extractors.text import HuggingFaceTextConfig
from lppws.study import Li2022PetitAverage
from lppws import pipeline as pl
import torch

D = Path('/home/co/tmp/li-avg/lpp_average_subject_en')
CACHE = Path('/home/co/tmp/lppws_cache')
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else 'lpp_features_v1.npz')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

study = Li2022PetitAverage(path=str(D), query='timeline_index < 9')
events = study.run()
timelines = list(dict.fromkeys(events['timeline']))
ctx = pl.add_running_context(events, timelines)

CONFIGS = [("gpt2-static", "openai-community/gpt2", False, 2/3, None)]
CONFIGS += [(f"size-{s}", f"EleutherAI/pythia-{s}", True, 2/3, None)
            for s in ["14m", "70m", "160m", "410m"]]
CONFIGS += [(f"step-{st}", "EleutherAI/pythia-70m", True, 2/3, f"step{st}")
            for st in [1, 512, 2000, 8000, 32000, 143000]]
CONFIGS += [(f"gpt2-layer-{int(l*100)}", "openai-community/gpt2", True, l, None)
            for l in [1/3, 2/3, 1.0]]

out = {}
for tag, model, contextual, layers, revision in CONFIGS:
    kw = {}
    if revision:
        kw['hf_config'] = HuggingFaceTextConfig(model_kwargs={'revision': revision})
    ev = ctx if contextual else events
    h = pl.hf_features(ev, model, contextualized=contextual, layers=layers,
                       device=DEV, cache_dir=CACHE, **kw)
    for i, tl in enumerate(timelines):
        out[f"{tag}|{i}"] = pl.design_matrix(h, ev, tl).astype(np.float16)
    print(f"{tag:18s} {out[f'{tag}|0'].shape} x9", flush=True)
    del h
    torch.cuda.is_available() and torch.cuda.empty_cache()

np.savez_compressed(OUT, **out)
print("wrote", OUT, f"{OUT.stat().st_size/1e6:.1f} MB", len(out), "arrays")
