#!/usr/bin/env python3
"""Independent E20 input, assignment, geometry, checkpoint and metric audit."""
import argparse,hashlib,json,math,pathlib
import numpy as np
import torch
from plyfile import PlyData
from scipy.special import expit
from covariance_moment_core import group_moments

ARMS=['fixed_E19_mean_beta0','window_full_geometry','window_position_only']
WINDOW=['window_full_geometry','window_position_only']

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def close(a,b,tol=1e-8):
 assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tol,(a,b)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('output',type=pathlib.Path);args=ap.parse_args();out=args.output.resolve()
 m=json.loads((out/'manifest.json').read_text());p=m['protocol'];assert p['window_size']==5 and p['max_switch_fraction_per_window']==.02
 n=m['n'];k=m['k'];cap=m['cap_per_window'];assert n==24985 and k==9276 and cap==500
 fit=m['fit_indices'];held=m['holdout_indices'];assert fit==list(range(0,200,2))+[199] and held==list(range(1,199,2))
 rng=np.random.default_rng(42);assert len(m['orders'])==20 and all(x==rng.permutation(fit).tolist() for x in m['orders'])
 for name,h in m['inputs'].items():assert digest(pathlib.Path(name))==h,name
 e19=pathlib.Path(m['E19_control']).parent;old=json.loads((e19/'report.json').read_text());control=next(x for x in old['rows'] if x['arm']=='mean_covariance__beta0')
 g=np.load(e19/'shared-grouping.npz');fixed=g['labels'];assert fixed.shape==(n,)
 full=np.load(e19/'teacher-states.npy',mmap_mode='r');fixed_states=np.load(e19/'mean-states.npy',mmap_mode='r')
 base=pathlib.Path(next(x for x in m['inputs'] if x.endswith('/cfg_args'))).parent
 ply=base/'point_cloud/iteration_20000/point_cloud.ply';source=PlyData.read(ply)['vertex'];w=expit(np.asarray(source['opacity'],dtype=np.float64));w/=w.sum()
 assert w.shape==(n,) and full.shape==(200,n,12) and fixed_states.shape==(200,k,12)
 events=json.loads((out/'switch-events.json').read_text());labels={a:np.load(out/(a+'-labels.npy'),mmap_mode='r') for a in WINDOW};states={a:np.load(out/(a+'-states.npy'),mmap_mode='r') for a in WINDOW}
 max_geometry_error=0.;max_cost_error=0.;total_switches={};mean_disagreement={};empty_groups=0
 distribution=json.loads((out/'distribution-per-time.json').read_text());assert len(distribution)==600
 D={(x['arm'],x['time_index']):x for x in distribution}
 for arm in WINDOW:
  assert labels[arm].shape==(200,n) and states[arm].shape==(200,k,12) and len(events[arm])==40
  previous=fixed;switch_count=0;disagreement=[]
  for window,e in enumerate(events[arm]):
   block=list(range(5*window,5*window+5));expected_fit=[i for i in block if i in fit]
   assert e['window']==window and e['fit_indices']==expected_fit
   now=np.asarray(labels[arm][block[0]],dtype=np.int64)
   assert all(np.array_equal(labels[arm][i],now) for i in block)
   changed=int(np.count_nonzero(now!=previous));assert changed==e['switches'] and changed<=cap
   count=np.bincount(now,minlength=k);assert count.min()>=1 and e['minimum_group_size']==int(count.min())
   assert e['disagreement_with_fixed']==int(np.count_nonzero(now!=fixed))
   switch_count+=changed;disagreement.append(e['disagreement_with_fixed']/n);previous=now
  total_switches[arm]=switch_count;mean_disagreement[arm]=float(np.mean(disagreement))
 for i in range(200):
  x=np.asarray(full[i],dtype=np.float64)
  for arm in ARMS:
   lab=fixed if arm==ARMS[0] else labels[arm][i]
   centers=fixed_states[i] if arm==ARMS[0] else states[arm][i]
   delta=x-centers[lab];fullcost=float(np.sum(w*np.sum(delta*delta,axis=1)));poscost=float(np.sum(w*np.sum(delta[:,:3]**2,axis=1)))
   max_cost_error=max(max_cost_error,abs(fullcost-D[(arm,i)]['full12_weighted_cost']),abs(poscost-D[(arm,i)]['position_weighted_cost']))
   assert D[(arm,i)]['split']==('fit' if i in fit else 'holdout')
   if arm in WINDOW:
    mu,cov,_,_=group_moments(x,lab,w,k)
    max_geometry_error=max(max_geometry_error,float(np.max(abs(mu-centers[:,:3]))),float(np.max(abs(cov-centers[:,3:].reshape(k,3,3)))))
 assert max_cost_error<1e-9 and max_geometry_error<1e-6,(max_cost_error,max_geometry_error)
 frozen=json.loads((out/'training-complete.json').read_text());assert frozen['two_new_models_frozen_before_GT'] and frozen['control_reused_and_hash_verified'] and frozen['new_gradient_updates']==4040
 assert len(frozen['checkpoint_hashes'])==2
 for name,h in frozen['checkpoint_hashes'].items():
  path=out/name;assert digest(path)==h
  ck=torch.load(path,map_location='cpu');assert ck['steps']==2020 and ck['arm'] in WINDOW
  assert all(int(s['step'])==2020 for s in ck['optimizer']['state'].values())
  assert all(torch.isfinite(v).all() for v in ck['state_dict'].values())
 training=json.loads((out/'training.json').read_text());assert len(training)==40
 assert all([x['updates'] for x in training if x['arm']==arm]==list(range(101,2021,101)) for arm in WINDOW)
 gt=json.loads((out/'gt-per-view.json').read_text());fid=json.loads((out/'fidelity-per-view.json').read_text());tmp=json.loads((out/'temporal-per-pair.json').read_text())
 assert len(gt)==60 and len(fid)==600 and len(tmp)==597
 report=json.loads((out/'report.json').read_text());rows={x['arm']:x for x in report['rows']};assert set(rows)==set(ARMS)
 for arm in ARMS:
  assert {x['view'] for x in gt if x['arm']==arm}==set(range(20))
  assert {x['time_index'] for x in fid if x['arm']==arm}==set(range(200))
  assert {x['time_index'] for x in tmp if x['arm']==arm}==set(range(1,200))
  for metric in ['PSNR','SSIM','LPIPS_alex_v01']:
   close(rows[arm][metric],float(np.mean([x[metric] for x in gt if x['arm']==arm])))
  close(rows[arm]['temporal_residual256'],float(np.mean([x['residual'] for x in tmp if x['arm']==arm])))
  for split in ['fit','holdout']:
   sub=[x for x in fid if x['arm']==arm and x['split']==split]
   assert len(sub)==(101 if split=='fit' else 99)
   for metric in ['PSNR_to_teacher','L1_to_teacher','mixed_loss_to_teacher']:
    close(rows[arm][split][metric],float(np.mean([x[metric] for x in sub])))
   sub=[x for x in distribution if x['arm']==arm and x['split']==split]
   for metric in ['full12_weighted_cost','position_weighted_cost']:
    close(rows[arm]['distribution'][split][metric],float(np.mean([x[metric] for x in sub])))
 for metric in ['PSNR','SSIM','LPIPS_alex_v01','temporal_residual256']:
  close(rows[ARMS[0]][metric],control[metric],1e-6)
  for arm in WINDOW:close(report['paired_new_minus_fixed'][arm][metric],rows[arm][metric]-rows[ARMS[0]][metric])
 for arm in WINDOW:
  assert rows[arm]['switches_total']==total_switches[arm]
  close(rows[arm]['mean_disagreement_with_fixed'],mean_disagreement[arm])
 completion=json.loads((out/'completion.json').read_text());assert completion['status']=='completed' and completion['inputs_unchanged'] and completion['checkpoints_verified']
 assert completion['source_sha256']==digest(out/'source.zip')
 base_fit=rows[ARMS[0]]['distribution']['fit'];cov_fraction=(base_fit['full12_weighted_cost']-base_fit['position_weighted_cost'])/base_fit['full12_weighted_cost']
 cross_disagreement=float(np.mean([np.mean(labels[WINDOW[0]][i]!=labels[WINDOW[1]][i]) for i in range(200)]))
 result={'status':'passed','input_hashes':len(m['inputs']),'new_checkpoints':2,'training_rows':40,'GT_rows':60,'teacher_fidelity_rows':600,'temporal_rows':597,'distribution_rows':600,'switches':total_switches,'mean_disagreement_with_fixed':mean_disagreement,'cross_method_label_disagreement':cross_disagreement,'baseline_covariance_fraction_of_full12_fit_cost':cov_fraction,'max_geometry_error':max_geometry_error,'max_cost_error':max_cost_error,'paired_new_minus_fixed':report['paired_new_minus_fixed']}
 (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
