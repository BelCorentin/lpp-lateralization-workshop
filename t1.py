import numpy as np, nibabel as nib, torch
from pathlib import Path
from nilearn.maskers import NiftiMasker
import neuralset as ns
from lppws.study import Li2022PetitAverage

D = Path('/home/co/tmp/li-avg/lpp_average_subject_en')
TR=2.0; FMRI_FREQ=1/TR; INNER=10.0
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device',DEV)
study = Li2022PetitAverage(path=str(D), query='timeline_index < 9')
events = study.run()
timelines = list(dict.fromkeys(events['timeline']))

mask_path = next(D.glob('**/mask_lpp_en.nii.gz'))
masker = NiftiMasker(mask_img=str(mask_path), detrend=True, standardize=True, high_pass=1/128, t_r=TR).fit()
mask_img = nib.load(str(mask_path))
ijk = np.argwhere(mask_img.get_fdata()>0)
world = nib.affines.apply_affine(mask_img.affine, ijk)
left = world[:,0]<0; right = world[:,0]>0
print('vox',left.size,'L',left.sum(),'R',right.sum())

def load_bold(tl):
    fp = events[(events.timeline==tl)&(events.type=='Fmri')]['filepath'].iloc[0]
    return masker.transform(fp)
Y = {tl: load_bold(tl) for tl in timelines}
print({k.split(',')[-1]: v.shape for k,v in list(Y.items())[:3]})

CACHE = Path('/tmp/claude-1000/-home-co/baa9cd65-7fa7-42cf-a31f-b09bf0c96b9c/scratchpad/wscache')
infra = dict(folder=CACHE)
text = ns.extractors.HuggingFaceText(model_name='openai-community/gpt2', frequency=INNER,
    contextualized=False, aggregation='mean', layers=2/3, layer_aggregation='mean',
    device=DEV, infra=infra)
hrf = ns.extractors.HrfConvolve(extractor=text, frequency=FMRI_FREQ, infra=infra)
hrf.prepare(events)
tle = events[events.timeline==timelines[0]]
dur = float(tle[tle.type=='Fmri']['duration'].iloc[0])
X = np.asarray(hrf(tle, start=0.0, duration=dur))
print('X', X.shape, 'Y', Y[timelines[0]].shape)
