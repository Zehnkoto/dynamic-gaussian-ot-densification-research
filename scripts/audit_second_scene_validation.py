#!/usr/bin/env python3
"""Independently verify E19 geometry, checkpoints and all reported means."""
import argparse,hashlib,json,math,pathlib
import numpy as np
import torch
from covariance_moment_core import group_moments

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def same(a,b,tol=1e-8):
 assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tol,(a,b)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('output',type=pathlib.Path);args=ap.parse_args();out=args.output.resolve()
 m=json.loads((out/'manifest.json').read_text());r=json.loads((out/'report.json').read_text());t=json.loads((out/'training-complete.json').read_text())
 assert len(m['fit_indices'])==101 and len(m['holdout_indices'])==99
 assert set(m['fit_indices']).isdisjoint(m['holdout_indices'])
 assert set(m['fit_indices'])|set(m['holdout_indices'])==set(range(200))
 assert m['fit_indices']==list(range(0,200,2))+[199] and m['holdout_indices']==list(range(1,199,2))
 rng=np.random.default_rng(42)
 assert len(m['orders'])==20 and all(order==rng.permutation(m['fit_indices']).tolist() for order in m['orders'])
 for p,h in m['inputs'].items():assert digest(pathlib.Path(p))==h,p
 n=m['n'];k=m['k'];assert k==round(n*10000/26936) and r['shared_budget']==k
 assert t['all4_frozen_before_evaluation'] and t['new_gradient_updates']==8080 and len(t['checkpoint_hashes'])==4
 for name,h in t['checkpoint_hashes'].items():
  p=out/name;assert digest(p)==h
  ck=torch.load(p,map_location='cpu');assert ck['steps']==2020
  assert all(int(v['step'])==2020 for v in ck['optimizer']['state'].values())
  assert all(torch.isfinite(v).all() for v in ck['state_dict'].values())
 grouping=np.load(out/'shared-grouping.npz');labels=grouping['labels'];reps=grouping['representatives'];group_mass=grouping['normalized_mass']
 assert labels.shape==(n,) and reps.shape==(k,) and len(np.unique(labels))==k
 assert labels.min()==0 and labels.max()==k-1
 full=np.load(out/'teacher-states.npy',mmap_mode='r');mean=np.load(out/'mean-states.npy',mmap_mode='r');moment=np.load(out/'moment-states.npy',mmap_mode='r')
 assert full.shape==(200,n,12) and mean.shape==moment.shape==(200,k,12)
 assert np.array_equal(mean[:,:,:3],moment[:,:,:3])
 from plyfile import PlyData
 from scipy.special import expit
 source=PlyData.read(pathlib.Path(m['baseline'])/'point_cloud/iteration_20000/point_cloud.ply')['vertex']
 source_weights=expit(np.asarray(source['opacity'],dtype=np.float64));source_weights/=source_weights.sum()
 assert source_weights.shape==(n,) and np.all(source_weights>0)
 assert np.max(abs(np.bincount(labels,weights=source_weights,minlength=k)-group_mass))<1e-10
 max_mean=0.;max_cov=0.;max_moment=0.
 for i in range(200):
  assert np.isfinite(full[i]).all() and np.isfinite(mean[i]).all() and np.isfinite(moment[i]).all()
  mu,cov,scatter,_=group_moments(full[i],labels,source_weights,k)
  max_mean=max(max_mean,float(np.max(abs(mu-mean[i,:,:3]))))
  max_cov=max(max_cov,float(np.max(abs(cov-mean[i,:,3:].reshape(k,3,3)))))
  max_moment=max(max_moment,float(np.max(abs((cov+scatter)-moment[i,:,3:].reshape(k,3,3)))))
 assert max_mean<1e-6 and max_cov<1e-6 and max_moment<2e-6,(max_mean,max_cov,max_moment)
 logs=json.loads((out/'training.json').read_text());assert len(logs)==80
 arms={f'{c}__beta{b:g}' for c in ['mean_covariance','moment_covariance'] for b in [0.,0.1]}
 assert {x['arm'] for x in logs}==arms
 assert all([x['updates'] for x in logs if x['arm']==a]==list(range(101,2021,101)) for a in arms)
 gt=json.loads((out/'gt-per-view.json').read_text());fid=json.loads((out/'fidelity-per-view.json').read_text());tmp=json.loads((out/'temporal-per-pair.json').read_text())
 assert len(gt)==80 and len(fid)==800 and len(tmp)==796
 rows={x['arm']:x for x in r['rows']};assert set(rows)==arms
 for a in arms:
  assert {x['view'] for x in gt if x['arm']==a}==set(range(20))
  assert {x['time_index'] for x in fid if x['arm']==a}==set(range(200))
  assert {x['time_index'] for x in tmp if x['arm']==a}==set(range(1,200))
  assert sum(x['split']=='fit' for x in fid if x['arm']==a)==101
  for metric in ['PSNR','SSIM','LPIPS_alex_v01']:
   same(rows[a][metric],float(np.mean([x[metric] for x in gt if x['arm']==a])))
  same(rows[a]['temporal_residual256'],float(np.mean([x['residual'] for x in tmp if x['arm']==a])))
  for split in ['fit','holdout']:
   records=[x for x in fid if x['arm']==a and x['split']==split]
   for metric in ['PSNR_to_teacher','L1_to_teacher','mixed_loss_to_teacher']:
    same(rows[a][split][metric],float(np.mean([x[metric] for x in records])))
 for c,p in r['paired_beta01_minus_beta0'].items():
  for metric,d in p.items():same(d,rows[f'{c}__beta0.1'][metric]-rows[f'{c}__beta0'][metric])
 for b,p in r['paired_moment_minus_mean'].items():
  for metric,d in p.items():same(d,rows[f'moment_covariance__beta{float(b):g}'][metric]-rows[f'mean_covariance__beta{float(b):g}'][metric])
 completion=json.loads((out/'completion.json').read_text());assert completion['status']=='completed'
 assert completion['inputs_unchanged'] and completion['checkpoints_verified'] and completion['source_sha256']==digest(out/'source.zip')
 report={'status':'passed','input_hashes':len(m['inputs']),'teacher_gaussians':n,'budget':k,'checkpoints':4,'training_rows':80,'GT_rows':80,'teacher_fidelity_rows':800,'temporal_rows':796,'paired_beta':r['paired_beta01_minus_beta0'],'paired_shape':r['paired_moment_minus_mean'],'source_group_mean_max_abs_error':max_mean,'source_covariance_max_abs_error':max_cov,'second_moment_max_abs_error':max_moment}
 (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
