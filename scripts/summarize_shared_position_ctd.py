#!/usr/bin/env python3
"""Audit E13 shared-position states and direct source-wise fixed-plan CTD changes."""
import argparse,csv,json,os,pathlib,zipfile
import numpy as np
import torch
from position_ctd_core import SharedTranslation
from run_ctd_comparison import dump,digest
ap=argparse.ArgumentParser();ap.add_argument('output',type=pathlib.Path);out=ap.parse_args().output.resolve();root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();assert out.is_relative_to(root);read=lambda p:json.loads(p.read_text());m=read(out/'manifest.json');p=m['protocol'];design=read(out/'design.json');freeze=read(out/'training-complete.json');done=read(out/'completion.json');summary=read(out/'validation/summary.json');gt=read(out/'validation/gt-per-view.json');fidelity=read(out/'validation/fidelity-per-view.json');temporal=read(out/'validation/temporal-per-pair.json');training=read(out/'training.json')
assert done['status']=='completed' and digest(out/'source.zip')==done['source_sha256'] and freeze['all15_frozen_before_evaluation'] and done['all15_frozen_before_evaluation'];assert freeze['new_gradient_updates']==done['new_gradient_updates']==5700
for path,h in m['inputs'].items():assert digest(pathlib.Path(path))==h
source=root/'outputs/ot/e12-teacher-rendering-control-20261104T163644Z';e12design=read(source/'design.json');assert design['future_orders']==e12design['future_orders'] and design['initial_appearance_digest']==e12design['initial_digests']['ctd128'];assert len({tuple(o) for orders in design['future_orders'].values() for o in orders})==15
geo=np.load(out/'base-geometry.npy',mmap_mode='r');assert np.array_equal(geo,np.load(source/'geometry-ctd128.npy',mmap_mode='r')) and geo.shape==(150,10256,12)
proposal=root/'outputs/ot/e05-split-20261104T103723Z';pm=read(proposal/'manifest.json');cache=pathlib.Path(pm['source_cache']);full=np.load(cache/'states.npy',mmap_mode='r');weights=np.load(cache/'opacity.npy').astype(np.float64).reshape(-1);weights/=weights.sum();original=np.load(root/'outputs/ot/e02-ctd-refined-20261103T082856Z/budget-10000/shared-grouping.npz')['labels'];child=np.load(proposal/'children.npz')['source_child'];labels=original.copy();splits=design['first_parents']+design['second_parents']
for j,parent in enumerate(splits):labels[(original==parent)&(child==1)]=10000+j
stats=np.load(out/'ctd-statistics.npz');assert np.array_equal(labels,stats['labels']) and len(np.unique(labels))==10256;mass=np.bincount(labels,weights=weights,minlength=10256);assert np.allclose(mass,stats['mass'],atol=1e-15,rtol=0) and np.isclose(mass.sum(),1)
base=[]
for t in range(150):base.append(float((weights[:,None]*(np.array(full[t],dtype=np.float64)-np.array(geo[t,labels],dtype=np.float64))**2).sum()))
base=np.array(base);assert np.allclose(base,stats['base_ctd'],atol=1e-15,rtol=0) and abs(base[m['fit']].mean()-stats['fit_base'])<1e-15 and abs(base[m['holdout']].mean()-stats['held_base'])<1e-15
assert len(summary)==len(freeze['model_info'])==len(freeze['checkpoint_hashes'])==15 and len(training)==75 and len(gt)==300 and len(fidelity)==150 and len(temporal)==149
rows=[];max_ctd_error=0.
for name,info in freeze['model_info'].items():
 condition=info['condition'];recipe=p['conditions'][condition];seed=info['seed'];assert info['initial_digest']==design['initial_appearance_digest'] and info['appearance_steps']==1520 and info['geometry_steps']==(380 if recipe['move'] else 0) and info['ctd_weight']==recipe['ctd_weight']
 path=out/'validation'/(name+'-state.pt');assert digest(path)==freeze['checkpoint_hashes'][path.name] and path.stat().st_mtime<=(out/'training-complete.json').stat().st_mtime;ck=torch.load(path,map_location='cpu');assert ck['info']==info and len(ck['appearance']['dc'])==10256==summary[name]['gaussians']==done['counts'][name]
 for v in list(ck['appearance'].values())+list(ck['position'].values()):assert torch.isfinite(v).all()
 for state in ck['appearance_optimizer']['state'].values():assert float(state['step'])==1520 and torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()
 if recipe['move']:
  assert len(ck['position_optimizer']['state'])==1
  for state in ck['position_optimizer']['state'].values():assert float(state['step'])==380 and torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()
 else:assert ck['position_optimizer'] is None and torch.count_nonzero(ck['position']['raw'])==0
 field=SharedTranslation(torch.tensor(stats['mass']),torch.tensor(stats['fit_linear']),float(stats['fit_base']),info['radius']);field.load_state_dict(ck['position']);assert np.array_equal(field.mass.numpy(),stats['mass']) and np.array_equal(field.linear.numpy(),stats['fit_linear']) and float(field.baseline)==float(stats['fit_base']);delta=field.offset().detach().numpy().astype(np.float64);norm=np.linalg.norm(delta,axis=1);assert norm.max()<=p['position_radius']+1e-7
 direct=[]
 for t in range(150):
  before=np.array(geo[t,labels,:3],dtype=np.float64)-np.array(full[t,:,:3],dtype=np.float64);after=before+delta[labels];direct.append(float((weights[:,None]*(after**2-before**2)).sum()))
 direct=np.array(direct);fit_change=float(direct[m['fit']].mean());held_change=float(direct[m['holdout']].mean());fit_quad=float((mass[:,None]*delta**2+2*stats['fit_linear']*delta).sum());held_quad=float((mass[:,None]*delta**2+2*stats['held_linear']*delta).sum());err=max(abs(fit_change-fit_quad),abs(held_change-held_quad));max_ctd_error=max(max_ctd_error,err);assert err<1e-12
 assert abs(float(field.excess().detach())-fit_change)<1e-12
 post=[r for r in training if r['arm']==name];assert len(post)==5
 for e,r in enumerate(post):assert r['epoch']==e+1 and r['main_steps']==1140+(e+1)*76 and np.isfinite([r['render_loss'],r['total_loss'],r['relative_CTD_end'],r['max_offset']]).all() and r['max_offset']<=p['position_radius']+1e-7
 assert abs(post[-1]['relative_CTD_end']-fit_change/float(stats['fit_base']))<1e-5
 views=[r for r in gt if r['arm']==name];assert sorted(r['test_index'] for r in views)==list(range(20))
 for metric in ['PSNR','SSIM','LPIPS_alex_v01']:assert abs(np.mean([r[metric] for r in views])-summary[name]['GT'][metric])<1e-12
 for split in ['fit','holdout']:
  values=[r['arms'][name] for r in fidelity if r['split']==split];assert len(values)==(76 if split=='fit' else 74)
  for key,v in summary[name]['fidelity'][split].items():assert np.isfinite([r[key] for r in values]).all() and abs(np.mean([r[key] for r in values])-v)<1e-12
 assert abs(np.mean([r[name] for r in temporal])-summary[name]['temporal_residual256'])<1e-12
 rows.append({'arm':name,'seed':seed,'condition':condition,'ctd_weight':recipe['ctd_weight'],'gaussians':10256,**summary[name]['GT'],'temporal_residual256':summary[name]['temporal_residual256'],'fit_CTD':float(stats['fit_base'])+fit_change,'held_CTD':float(stats['held_base'])+held_change,'fit_CTD_relative_increase':fit_change/float(stats['fit_base']),'held_CTD_relative_increase':held_change/float(stats['held_base']),'offset_mean':float(norm.mean()),'offset_weighted_RMS':float(np.sqrt((mass*norm**2).sum())),'offset_max':float(norm.max()),'fraction_near_bound95pct':float((norm>.95*p['position_radius']).mean())})
aggregates=[];paired=[]
for condition in p['conditions']:
 selected=[r for r in rows if r['condition']==condition];assert sorted(r['seed'] for r in selected)==[42,43,44];aggregates.append({'condition':condition,'ctd_weight':p['conditions'][condition]['ctd_weight'],'n_order_seeds':3,'metrics':{k:{'mean':float(np.mean([r[k] for r in selected])),'sample_std':float(np.std([r[k] for r in selected],ddof=1))} for k in ['PSNR','SSIM','LPIPS_alex_v01','temporal_residual256','fit_CTD_relative_increase','held_CTD_relative_increase','offset_weighted_RMS','offset_max']}})
for seed in p['order_seeds']:
 selected={r['condition']:r for r in rows if r['seed']==seed}
 for condition in p['conditions']:
  if condition=='appearance_only':continue
  for reference in ['appearance_only','position_render']:
   if condition==reference:continue
   paired.append({'seed':seed,'condition':condition,'reference':reference,**{k+'_difference':selected[condition][k]-selected[reference][k] for k in ['PSNR','LPIPS_alex_v01','temporal_residual256','fit_CTD_relative_increase','offset_weighted_RMS']}})
old=read(source/'validation/summary.json');repeats=[]
for r in rows:
 if r['condition']=='appearance_only':
  target=old[f'native400__ctd128__seed{r["seed"]}']['GT'];repeats.append({'seed':r['seed'],**{k+'_delta':r[k]-target[k] for k in ['PSNR','SSIM','LPIPS_alex_v01']}})
analysis=out/'analysis';analysis.mkdir()
for filename,data in [('comparison.csv',rows),('paired-effects.csv',paired)]:
 with (analysis/filename).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
report={'rows':rows,'aggregates':aggregates,'paired_effects':paired,'appearance_control_E12_repeat_deltas':repeats,'base_CTD':{'fit':float(stats['fit_base']),'holdout':float(stats['held_base'])},'max_direct_vs_quadratic_CTD_error':max_ctd_error,'verification':{'all_input_and_source_hashes':True,'paired_initial_state_and_orders_match_E12':True,'base_geometry_and_source_partition_verified':True,'all15_direct_source_CTD_deltas_checked':True,'all_offsets_within_radius_and_covariance_fixed':True,'appearance_Adam1520_geometry_Adam380or0':True,'all15_models_frozen_before_evaluation':True,'all75_epochs_and300_GT_rows_verified':True,'all_fidelity_and_temporal_means_verified':True},'scope':'First shared-position pilot with differentiable fixed-plan CTD excess; constant world shifts, frozen covariance/assignment/deformation. No OT-plan or new adaptive split learning.','limits':p['limits']};dump(analysis/'report.json',report)
with zipfile.ZipFile(analysis/'source.zip','x') as z:z.write(pathlib.Path(__file__).resolve(),'summarize_shared_position_ctd.py')
dump(root/'projects/dynamic-gaussian-ot-densification-research/experiments/e13-status-20261105.json',{'run':str(out),'report':report,'remote_publish_requested':False,'full4DGS_training_densification_implemented':False})
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(16,4.8),layout='constrained');labels=['Appearance','Position/render','CTD0.001','CTD0.01','CTD0.1']
for ax,key,title in zip(axes,['PSNR','fit_CTD_relative_increase','temporal_residual256'],['20-view GT PSNR (higher)','Fixed-plan fit CTD relative increase','Temporal residual256 (lower)']):
 values=[r['metrics'][key]['mean'] for r in aggregates];std=[r['metrics'][key]['sample_std'] for r in aggregates];ax.errorbar(range(5),values,yerr=std,fmt='o-',capsize=3);ax.set_xticks(range(5),labels,rotation=30,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.3)
fig.suptitle('E13: K10256, bounded shared positions, native400; mean +/- SD over3 future-order seeds');fig.savefig(analysis/'shared-position-ctd.png',dpi=140);plt.close(fig)
print(json.dumps({'aggregates':aggregates,'repeats':repeats,'max_direct_CTD_error':max_ctd_error,'verification':report['verification']},indent=2))
