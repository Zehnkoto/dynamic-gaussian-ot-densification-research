#!/usr/bin/env python3
"""Independent SSE and saved evaluation checks for the E05 offline pilot."""
import argparse,csv,json,os,pathlib,zipfile
import numpy as np
import torch
from run_ctd_comparison import digest,dump
ap=argparse.ArgumentParser();ap.add_argument('input',type=pathlib.Path);a=ap.parse_args();inp=a.input.resolve();root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();assert inp.is_relative_to(root);out=inp/'validation'
c=json.loads((out/'completion.json').read_text());assert c['status']=='completed'
m=json.loads((inp/'manifest.json').read_text());s=json.loads((out/'summary.json').read_text());vm=json.loads((out/'manifest.json').read_text());proposal=json.loads((inp/'proposal-summary.json').read_text());completion=json.loads((inp/'completion.json').read_text());assert digest(inp/'source.zip')==completion['source_sha256']
for p,h in m['inputs'].items():assert digest(pathlib.Path(p))==h
for n,h in completion['artifacts'].items():assert digest(inp/n)==h
cache=pathlib.Path(m['source_cache']);x=np.load(cache/'states.npy',mmap_mode='r');w=np.load(cache/'opacity.npy').astype(float).reshape(-1);w/=w.sum();old=np.load(m['parent_states'],mmap_mode='r');parents=np.load(root/'outputs/ot/e02-ctd-refined-20261103T082856Z/budget-10000/shared-grouping.npz')['labels'];ch=np.load(inp/'children.npz');child=ch['source_child'];gain=ch['gains'];checks={}
budget=m['protocol']['base_budget'];splits=m['protocol']['splits'];final=budget+splits
for name,r in proposal['methods'].items():
 ids=np.array(r['selected']);labels=parents.copy();assert len(np.unique(ids))==splits
 for j,parent in enumerate(ids):labels[(parents==parent)&(child==1)]=budget+j
 assert len(np.unique(labels))==final
 new=np.load(inp/(name+'-states.npy'),mmap_mode='r');errors=[]
 for t in [0,75,149]:
  before=(w[:,None]*(x[t].astype(float)-old[t,parents])**2).sum();after=(w[:,None]*(x[t].astype(float)-new[t,labels])**2).sum();expected=gain[t,ids].sum();relative=abs(before-after-expected)/max(abs(expected),1e-30);assert relative<1e-4;errors.append(relative)
 checks[name]={'exact_budget_nonempty_clusters':True,'independent_SSE_gain_max_relative_error':max(errors)}
views=json.loads((out/'gt-per-view.json').read_text());training=json.loads((out/'training.json').read_text());rows=[]
for name in ['continue','window','global','first','random']:
 ck=torch.load(out/(name+'-appearance.pt'),map_location='cpu');count=budget if name=='continue' else final
 assert ck['steps']==1520 and ck['state_dict']['dc'].shape[0]==count and all(torch.isfinite(v).all() for v in ck['state_dict'].values())
 assert len([r for r in training if r['arm']==name])==20
 for phase in ['before','after']:
  v=[r for r in views if r['arm']==name and r['phase']==phase];assert sorted(r['test_index'] for r in v)==list(range(20))
  for metric in ['PSNR','SSIM','LPIPS_alex_v01']:assert abs(np.mean([r[metric] for r in v])-s['official_20_view_gt'][name+'_'+phase][metric])<1e-12
 row={'arm':name,'gaussians':count,'holdout_CTD_reduction_percent':0 if name=='continue' else proposal['methods'][name]['holdout_CTD_reduction_percent'],'PSNR_before':s['official_20_view_gt'][name+'_before']['PSNR'],'PSNR_after':s['official_20_view_gt'][name+'_after']['PSNR'],'LPIPS_after':s['official_20_view_gt'][name+'_after']['LPIPS_alex_v01'],'temporal_after':s['fixed_camera_temporal_delta_residual_256'][name+'_after'],'holdout_PSNR_to_full':s['fidelity_to_full_400'][name+'_after']['holdout']['PSNR_to_full']};row['PSNR_delta_vs_continue']=row['PSNR_after']-s['official_20_view_gt']['continue_after']['PSNR'];rows.append(row)
analysis=out/'analysis';analysis.mkdir(exist_ok=False)
with (analysis/'comparison.csv').open('x',newline='') as f:writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
report={'rows':rows,'checks':checks,'input_hashes_unchanged':True,'source_hash_verified':True,'finite_checkpoints':True,'fixed1520steps_each':True,'20view_means_verified':True,'scope':'Offline fixed extra budget; not training-time adaptive densification. Scoring-only windows; first-frame score still uses fit76 shared child partition.'};dump(analysis/'report.json',report)
with zipfile.ZipFile(analysis/'source.zip','x') as z:z.write(pathlib.Path(__file__).resolve(),'summarize_split_validation.py')
print(json.dumps(report,indent=2))
