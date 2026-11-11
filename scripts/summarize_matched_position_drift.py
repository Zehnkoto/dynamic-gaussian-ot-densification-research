#!/usr/bin/env python3
"""Independent E15 audit: fit-only calibration, frozen weights and paired evaluation."""
import argparse,csv,json,os,pathlib,zipfile,math
import numpy as np
import torch
from position_ctd_core import SharedTranslation
from run_ctd_comparison import dump,digest
ap=argparse.ArgumentParser();ap.add_argument('output',type=pathlib.Path);out=ap.parse_args().output.resolve();root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();assert out.is_relative_to(root)
read=lambda p:json.loads(p.read_text());manifest=read(out/'manifest.json');p=manifest['protocol'];design=read(out/'design.json');freeze=read(out/'training-complete.json');done=read(out/'completion.json');selection=read(out/'lambda-selection-frozen.json');calibration=read(out/'calibration.json');attempts=read(out/'calibration-attempts.json');training=read(out/'training.json');summary=read(out/'validation/summary.json');gt=read(out/'validation/gt-per-view.json');fid=read(out/'validation/fidelity-per-view.json');temporal=read(out/'validation/temporal-per-pair.json')
assert done['status']=='completed' and digest(out/'source.zip')==done['source_sha256'] and freeze['all12_frozen_before_evaluation'] and done['all12_frozen_before_evaluation']
assert digest(out/'lambda-selection-frozen.json')==freeze['lambda_selection_sha256'] and freeze['calibration_updates']==len(attempts)*380==selection['calibration_updates'] and freeze['confirmation_updates']==3040
assert freeze['new_gradient_updates']==done['new_gradient_updates']==(len(attempts)+8)*380 and len(training)==(len(attempts)+8)*5
for path,h in manifest['inputs'].items():assert digest(pathlib.Path(path))==h
old=root/p['reuse_runs']['ctd'];assert design==read(old/'design.json');geo=np.load(out/'base-geometry.npy',mmap_mode='r');assert np.array_equal(geo,np.load(old/'base-geometry.npy',mmap_mode='r')) and geo.shape==(150,10256,12)
proposal=root/'outputs/ot/e05-split-20261104T103723Z';pm=read(proposal/'manifest.json');cache=pathlib.Path(pm['source_cache']);full=np.load(cache/'states.npy',mmap_mode='r');weights=np.load(cache/'opacity.npy').astype(np.float64).reshape(-1);weights/=weights.sum();original=np.load(root/'outputs/ot/e02-ctd-refined-20261103T082856Z/budget-10000/shared-grouping.npz')['labels'];child=np.load(proposal/'children.npz')['source_child'];labels=original.copy()
for j,parent in enumerate(design['first_parents']+design['second_parents']):labels[(original==parent)&(child==1)]=10000+j
stats=np.load(out/'ctd-statistics.npz');mass=np.bincount(labels,weights=weights,minlength=10256);assert np.array_equal(labels,stats['labels']) and np.allclose(mass,stats['mass'],atol=1e-15,rtol=0) and np.all(mass>0)
base=[];linear=[]
for t in range(150):
 x=np.asarray(full[t],dtype=np.float64);g=np.asarray(geo[t],dtype=np.float64);base.append(float((weights[:,None]*(x-g[labels])**2).sum()));sums=np.stack([np.bincount(labels,weights=weights*x[:,d],minlength=10256) for d in range(3)],1);linear.append(mass[:,None]*g[:,:3]-sums)
base=np.array(base);linear=np.stack(linear);assert np.allclose(base,stats['base_ctd'],atol=1e-15,rtol=0)
for indices,key,basekey in [(manifest['fit'],'fit_linear','fit_base'),(manifest['holdout'],'held_linear','held_base')]:assert np.allclose(linear[indices].mean(0),stats[key],atol=1e-15,rtol=0) and abs(base[indices].mean()-stats[basekey])<1e-15
assert len({tuple(o) for orders in design['future_orders'].values() for o in orders})==15

def replay(path):
 ck=torch.load(path,map_location='cpu');field=SharedTranslation(torch.tensor(mass),torch.tensor(stats['fit_linear']),float(stats['fit_base']),p['position_radius']);field.load_state_dict(ck['position']);assert np.array_equal(field.mass.numpy(),stats['mass']) and np.array_equal(field.linear.numpy(),stats['fit_linear']) and float(field.baseline)==float(stats['fit_base'])
 for v in list(ck['position'].values())+list(ck['appearance'].values()):assert torch.isfinite(v).all()
 for state in ck['appearance_optimizer']['state'].values():assert float(state['step'])==1520 and torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()
 for state in ck['position_optimizer']['state'].values():assert float(state['step'])==380 and torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()
 delta=field.offset().detach().numpy().astype(np.float64);assert np.linalg.norm(delta,axis=1).max()<=p['position_radius']+1e-7
 fitdrift=float((mass[:,None]*delta**2+2*stats['fit_linear']*delta).sum()/stats['fit_base']);assert abs(fitdrift-float(field.relative_excess().detach()))<1e-12
 return ck,delta,fitdrift
# Replay all calibration choices independently; only lambda and fit drift enter this audit rule.
for recipe,c in calibration.items():
 records=c['records'];target=c['budget'];assert target in p['drift_budgets'];assert len(records)>=3 and len(records)-3<=p['max_new_candidates_per_recipe']
 for j,r in enumerate(records):
  path=pathlib.Path(r['checkpoint']);assert path.is_relative_to(root) and digest(path)==r['sha256'];ck,delta,fitdrift=replay(path);assert abs(fitdrift-r['fit_drift'])<1e-8
  if j<3:
   assert not r['new'] and ck['info']['seed']==42 and ck['info']['ctd_weight']==r['lambda'] and r['sha256']==manifest['inputs'][str(path)]
  else:
   assert r['new'] and ck['info']['seed']==42 and ck['info']['method']==c['method'] and ck['info']['regularizer_weight']==r['lambda']
   previous=records[:j];assert not any(abs(x['fit_drift']/target-1)<=p['calibration_relative_tolerance'] for x in previous)
   pairs=[(a,b) for a in previous for b in previous if a['fit_drift']>target>b['fit_drift'] and a['lambda']<b['lambda']];a,b=min(pairs,key=lambda ab:math.log(ab[1]['lambda']/ab[0]['lambda']));fraction=max(.1,min(.9,math.log(a['fit_drift']/target)/math.log(a['fit_drift']/b['fit_drift'])));expected=math.exp(math.log(a['lambda'])+fraction*math.log(b['lambda']/a['lambda']));assert abs(expected-r['lambda'])<1e-15
   arm=path.name.removesuffix('-state.pt');logs=[v for v in training if v['arm']==arm];assert len(logs)==5 and abs(logs[-1]['relative_CTD_end']-fitdrift)<1e-8
 passing=[v for v in records if abs(v['fit_drift']/target-1)<=p['calibration_relative_tolerance']];chosen=passing[0] if passing else min(records,key=lambda r:(abs(math.log(max(r['fit_drift'],1e-30)/target)),r['lambda']));assert chosen==c['selected'];s=selection['choices'][recipe];assert s['lambda']==chosen['lambda'] and s['sha256']==chosen['sha256'] and s['matched']==bool(passing)==c['matched']
assert len(summary)==len(freeze['model_info'])==len(freeze['checkpoint_hashes'])==12 and len(gt)==240 and len(fid)==150 and len(temporal)==149
rows=[];max_error=0.;selection_time=(out/'lambda-selection-frozen.json').stat().st_mtime;freeze_time=(out/'training-complete.json').stat().st_mtime
for name,info in freeze['model_info'].items():
 recipe=info['condition'];seed=info['seed'];chosen=selection['choices'][recipe];assert info['regularizer_weight']==chosen['lambda'] and info['budget']==chosen['budget'] and info['initial_digest']==design['initial_appearance_digest']
 path=out/'validation'/(name+'-state.pt');assert digest(path)==freeze['checkpoint_hashes'][path.name] and selection_time<=path.stat().st_mtime<=freeze_time;ck,delta,fitdrift=replay(path)
 for k in ['seed','method','regularizer_weight','radius','initial_digest','appearance_steps','geometry_steps']:assert ck['info'][k]==info[k]
 if seed==42:
  source=torch.load(chosen['checkpoint'],map_location='cpu')
  for key in ['appearance','position']:
   for k,v in source[key].items():assert torch.equal(v,ck[key][k])
 else:
  logs=[r for r in training if r['arm']==name];assert len(logs)==5 and all(r['phase']=='confirmation' and r['lambda']==chosen['lambda'] for r in logs) and abs(logs[-1]['relative_CTD_end']-fitdrift)<1e-8
 direct=[]
 for t in range(150):
  before=np.asarray(geo[t,labels,:3],dtype=np.float64)-np.asarray(full[t,:,:3],dtype=np.float64);after=before+delta[labels];direct.append(float((weights[:,None]*(after**2-before**2)).sum()))
 direct=np.array(direct);fit_change=float(direct[manifest['fit']].mean());held_change=float(direct[manifest['holdout']].mean());held_quad=float((mass[:,None]*delta**2+2*stats['held_linear']*delta).sum());err=max(abs(fit_change-fitdrift*float(stats['fit_base'])),abs(held_change-held_quad));max_error=max(max_error,err);assert err<1e-12
 assert abs(fitdrift-info['fit_drift'])<1e-8 and info['matched']==(abs(fitdrift/info['budget']-1)<=p['calibration_relative_tolerance'])
 views=[r for r in gt if r['arm']==name];assert sorted(v['test_index'] for v in views)==list(range(20))
 for metric in ['PSNR','SSIM','LPIPS_alex_v01']:assert abs(np.mean([r[metric] for r in views])-summary[name]['GT'][metric])<1e-12
 for split in ['fit','holdout']:
  values=[r['arms'][name] for r in fid if r['split']==split];assert len(values)==(76 if split=='fit' else 74)
  for key,value in summary[name]['fidelity'][split].items():assert abs(np.mean([r[key] for r in values])-value)<1e-12
 assert abs(np.mean([r[name] for r in temporal])-summary[name]['temporal_residual256'])<1e-12
 norm=np.linalg.norm(delta,axis=1)
 rows.append({'arm':name,'method':info['method'],'condition':recipe,'seed':seed,'role':'calibration' if seed==42 else 'confirmation','budget':info['budget'],'lambda':info['regularizer_weight'],'fit_CTD_relative_increase':fitdrift,'held_CTD_relative_increase':held_change/float(stats['held_base']),'budget_relative_error':fitdrift/info['budget']-1,'matched':info['matched'],'gaussians':10256,**summary[name]['GT'],'temporal_residual256':summary[name]['temporal_residual256'],'holdout_native400_mix':summary[name]['fidelity']['holdout']['native400_mix'],'holdout_native400_mse':summary[name]['fidelity']['holdout']['native400_mse'],'offset_weighted_RMS':float(np.sqrt((mass*norm**2).sum())),'offset_uniform_RMS':float(np.sqrt(np.mean(norm**2))),'offset_max':float(norm.max())})
for file in ['fidelity-per-view.json','gt-per-view.json','temporal-per-pair.json','summary.json']:assert (out/'validation'/file).stat().st_mtime>=freeze_time
keys=['PSNR','SSIM','LPIPS_alex_v01','temporal_residual256','fit_CTD_relative_increase','held_CTD_relative_increase','holdout_native400_mix','holdout_native400_mse','offset_weighted_RMS','offset_uniform_RMS'];aggregates=[];paired=[]
for budget in p['drift_budgets']:
 for method in p['methods']:
  for scope,seeds in [('all_orders',[42,43,44]),('confirmation_orders',[43,44])]:
   chosen=[r for r in rows if r['budget']==budget and r['method']==method and r['seed'] in seeds];assert len(chosen)==len(seeds)
   aggregates.append({'budget':budget,'method':method,'scope':scope,'n':len(chosen),'lambda':chosen[0]['lambda'],'all_matched':all(r['matched'] for r in chosen),'metrics':{k:{'mean':float(np.mean([r[k] for r in chosen])),'sample_std':float(np.std([r[k] for r in chosen],ddof=1))} for k in keys}})
 for seed in p['order_seeds']:
  c=next(r for r in rows if r['budget']==budget and r['method']=='ctd' and r['seed']==seed);u=next(r for r in rows if r['budget']==budget and r['method']=='l2' and r['seed']==seed)
  paired.append({'budget':budget,'seed':seed,'role':c['role'],'both_matched':c['matched'] and u['matched'],'difference':'CTD minus uniform L2',**{k+'_difference':c[k]-u[k] for k in keys}})
analysis=out/'analysis';analysis.mkdir()
for filename,data in [('comparison.csv',rows),('paired-effects.csv',paired)]:
 with (analysis/filename).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
# Strip local checkpoint paths from the public report, retain all calibration outcomes.
cal_public={key:{'method':c['method'],'budget':c['budget'],'selected_lambda':c['selected']['lambda'],'selected_fit_drift':c['selected']['fit_drift'],'matched':c['matched'],'records':[{k:v for k,v in r.items() if k!='checkpoint'} for r in c['records']]} for key,c in calibration.items()}
report={'rows':rows,'aggregates':aggregates,'paired_effects':paired,'calibration':cal_public,'new_calibration_candidates':len(attempts),'new_gradient_updates':done['new_gradient_updates'],'confirmation_updates':3040,'max_direct_vs_quadratic_CTD_error':max_error,'verification':{'all_input_source_checkpoint_hashes':True,'same_E13_initial_geometry_targets_orders':True,'fit_only_calibration_and_first_pass_rule_replayed':True,'lambda_choices_frozen_before_confirmation':True,'all12_frozen_before_evaluation':True,'source_partition_mass_baseline_linear_verified':True,'all12_direct_source_CTD_deltas_verified':True,'all240_GT_and_fidelity_temporal_means_verified':True,'all_optimizer_steps_and_radius_verified':True},'all_calibration_recipes_matched':all(c['matched'] for c in calibration.values()),'all_confirmation_arms_matched':all(r['matched'] for r in rows if r['role']=='confirmation'),'scope':'Fit-only seed42 calibration, frozenlambda future-order43/44 comparison on previously explored bouncingballs; no new scene or OT-plan/adaptive split learning.','limits':p['limits']};dump(analysis/'report.json',report)
with zipfile.ZipFile(analysis/'source.zip','x') as z:z.write(pathlib.Path(__file__).resolve(),'summarize_matched_position_drift.py')
dump(root/'projects/dynamic-gaussian-ot-densification-research/experiments/e15-status-20261105.json',{'run':str(out),'report':report,'remote_publish_requested':False,'full4DGS_training_densification_implemented':False})
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
for ax,key,title in zip(axes,['PSNR','fit_CTD_relative_increase','temporal_residual256'],['GT PSNR (higher)','Actual fit CTD increase (target1%,4%)','Temporal residual256 (lower)']):
 for method,label,marker in [('ctd','Source-mass CTD','o'),('l2','Uniform L2','s')]:
  selected=[r for r in aggregates if r['method']==method and r['scope']=='confirmation_orders'];ax.errorbar([r['budget']*100 for r in selected],[r['metrics'][key]['mean'] for r in selected],yerr=[r['metrics'][key]['sample_std'] for r in selected],fmt=marker+'-',capsize=3,label=label)
 if key=='fit_CTD_relative_increase':ax.plot([1,4],[.01,.04],color='gray',ls='--',label='Target')
 ax.set_xticks([1,4]);ax.set_xlabel('Predeclared drift target (%)');ax.set_title(title);ax.grid(alpha=.3);ax.legend(fontsize=8)
fig.suptitle('E15: lambda calibrated on fit-only order42; frozen for43/44; mean +/- sample SD, n=2 orders');fig.savefig(analysis/'matched-position-drift.png',dpi=140);plt.close(fig)
print(json.dumps({'aggregates':aggregates,'calibration':cal_public,'all_confirmation_arms_matched':report['all_confirmation_arms_matched'],'new_gradient_updates':report['new_gradient_updates'],'verification':report['verification']},indent=2))
