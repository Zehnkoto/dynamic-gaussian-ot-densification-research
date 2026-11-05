#!/usr/bin/env python3
"""Audit paired E12 states/orders/geometry and all evaluation means; report all cells."""
import argparse,csv,json,os,pathlib,zipfile
import numpy as np
import torch
from appearance_model import Appearance
from training_split_core import digest_state,RATES
from run_ctd_comparison import dump,digest
ap=argparse.ArgumentParser();ap.add_argument('output',type=pathlib.Path);out=ap.parse_args().output.resolve();root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();assert out.is_relative_to(root);read=lambda p:json.loads(p.read_text());m=read(out/'manifest.json');p=m['protocol'];design=read(out/'design.json');freeze=read(out/'training-complete.json');done=read(out/'completion.json');summary=read(out/'validation/summary.json');gt=read(out/'validation/gt-per-view.json');fidelity=read(out/'validation/fidelity-per-view.json');temporal=read(out/'validation/temporal-per-pair.json');training=read(out/'training.json')
assert done['status']=='completed' and digest(out/'source.zip')==done['source_sha256'] and freeze['all18_frozen_before_evaluation'] and done['all18_frozen_before_evaluation'];assert freeze['new_gradient_updates']==done['new_gradient_updates']==6840
for path,h in m['inputs'].items():assert digest(pathlib.Path(path))==h
source=root/'outputs/ot/e09-current-state-reranking-20261104T133944Z';sm=read(source/'manifest.json');selection=read(source/'selection.json');shared=torch.load(source/'before-second.pt',map_location='cpu');model=Appearance(shared['model']['source_sh'],shared['model']['source_logits'],10128);model.load_state_dict(shared['model']);opt=model.optimizer(RATES);opt.load_state_dict(shared['optimizer']);assert digest_state(model,opt)==shared['digest']==design['shared_digest']==design['initial_digests']['no_second']
assert design['first_parents']==selection['first_parents'];second=np.array(selection['candidate_batches'])[selection['chosen_commit_order']['fixed_ctd']].reshape(-1).tolist();assert design['second_parents']==second and not set(second)&set(design['first_parents'])
assert design['future_orders']['42']==sm['orders'][15:]
for seed in [43,44]:assert design['future_orders'][str(seed)]==[np.random.default_rng(seq).permutation(m['fit']).tolist() for seq in np.random.SeedSequence(seed).spawn(5)]
allorders=[tuple(order) for orders in design['future_orders'].values() for order in orders];assert len(allorders)==len(set(allorders))==15 and all(sorted(order)==sorted(m['fit']) for order in allorders)
proposal=root/'outputs/ot/e05-split-20261104T103723Z';pm=read(proposal/'manifest.json');parents=np.load(pm['parent_states'],mmap_mode='r');children=np.load(proposal/'child-states.npy',mmap_mode='r')
for cap in p['capacities']:
 ids=design['first_parents']+(second if cap=='ctd128' else []);expected=np.concatenate([np.array(parents,copy=True),children[:,np.array(ids)*2+1]],1);expected[:,ids]=children[:,np.array(ids)*2];assert np.array_equal(expected,np.load(out/('geometry-'+cap+'.npy'),mmap_mode='r'));del expected
assert len(summary)==len(freeze['model_info'])==len(freeze['checkpoint_hashes'])==18 and len(training)==90 and len(gt)==360 and len(fidelity)==150 and len(temporal)==149
assert [r['time_index'] for r in fidelity]==list(range(150)) and [r['time_index'] for r in temporal]==list(range(1,150));assert (out/'training-complete.json').stat().st_mtime<=(out/'validation/gt-per-view.json').stat().st_mtime
rows=[]
for name,info in freeze['model_info'].items():
 cap=info['capacity'];condition=info['condition'];seed=info['seed'];count=p['capacities'][cap];assert info['initial_digest']==design['initial_digests'][cap] and info['new_updates']==380 and info['main_steps']==1520 and info['student_operator']==p['conditions'][condition]['student'] and info['teacher_operator']==p['conditions'][condition]['target']
 path=out/'validation'/(name+'-appearance.pt');assert digest(path)==freeze['checkpoint_hashes'][path.name] and path.stat().st_mtime<=(out/'training-complete.json').stat().st_mtime;ck=torch.load(path,map_location='cpu');assert ck['info']==info and len(ck['state_dict']['dc'])==count==summary[name]['gaussians']==done['counts'][name]
 for v in ck['state_dict'].values():assert torch.isfinite(v).all()
 for state in ck['optimizer']['state'].values():assert float(state['step'])==1520 and torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()
 post=[r for r in training if r['arm']==name];assert len(post)==5
 for e,r in enumerate(post):assert r['epoch']==e+1 and r['main_steps']==1140+(e+1)*76 and r['gaussians']==count and np.isfinite(r['loss'])
 views=[r for r in gt if r['arm']==name];assert sorted(r['test_index'] for r in views)==list(range(20))
 for metric in ['PSNR','SSIM','LPIPS_alex_v01']:assert abs(np.mean([r[metric] for r in views])-summary[name]['GT'][metric])<1e-12
 for split in ['fit','holdout']:
  values=[r['arms'][name] for r in fidelity if r['split']==split];assert len(values)==(76 if split=='fit' else 74)
  for key,v in summary[name]['fidelity'][split].items():assert np.isfinite([r[key] for r in values]).all() and abs(np.mean([r[key] for r in values])-v)<1e-12
 assert abs(np.mean([r[name] for r in temporal])-summary[name]['temporal_residual256'])<1e-12
 rows.append({'arm':name,'seed':seed,'capacity':cap,'condition':condition,'gaussians':count,**summary[name]['GT'],'temporal_residual256':summary[name]['temporal_residual256'],'held_native400_mse':summary[name]['fidelity']['holdout']['native400_mse'],'held_area800_mse':summary[name]['fidelity']['holdout']['area800_mse'],'held_GT400_mse':summary[name]['fidelity']['holdout']['GT400_mse'],'train_seconds':info['train_seconds']})
aggregates=[];paired=[];split_effect=[]
for cap in p['capacities']:
 for condition in p['conditions']:
  selected=[r for r in rows if r['capacity']==cap and r['condition']==condition];assert sorted(r['seed'] for r in selected)==[42,43,44]
  aggregates.append({'capacity':cap,'condition':condition,'gaussians':p['capacities'][cap],'n_order_seeds':3,'metrics':{k:{'mean':float(np.mean([r[k] for r in selected])),'sample_std':float(np.std([r[k] for r in selected],ddof=1))} for k in ['PSNR','SSIM','LPIPS_alex_v01','temporal_residual256','held_GT400_mse']}})
 for seed in p['order_seeds']:
  selected={r['condition']:r for r in rows if r['capacity']==cap and r['seed']==seed}
  for a,b in [('teacher800_area','native400'),('both800_area','native400'),('both800_area','teacher800_area')]:paired.append({'capacity':cap,'seed':seed,'a':a,'b':b,**{k+'_a_minus_b':selected[a][k]-selected[b][k] for k in ['PSNR','SSIM','LPIPS_alex_v01','temporal_residual256']}})
for condition in p['conditions']:
 for seed in p['order_seeds']:
  selected={r['capacity']:r for r in rows if r['condition']==condition and r['seed']==seed};split_effect.append({'condition':condition,'seed':seed,'PSNR_ctd128_minus_no_second':selected['ctd128']['PSNR']-selected['no_second']['PSNR'],'LPIPS_ctd128_minus_no_second':selected['ctd128']['LPIPS_alex_v01']-selected['no_second']['LPIPS_alex_v01']})
old=read(source/'validation/summary.json')['official_20_view_gt']['fixed_ctd_after'];repeat=next(r for r in rows if r['capacity']=='ctd128' and r['seed']==42 and r['condition']=='native400');repeat_delta={k:repeat[k]-old[k] for k in ['PSNR','SSIM','LPIPS_alex_v01']}
analysis=out/'analysis';analysis.mkdir()
for filename,data in [('comparison.csv',rows),('paired-effects.csv',paired),('split-effects.csv',split_effect)]:
 with (analysis/filename).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
report={'rows':rows,'aggregates':aggregates,'paired_effects':paired,'split_effects':split_effect,'native400_seed42_E09_repeat_delta':repeat_delta,'verification':{'input_and_source_hashes':True,'shared1140_state_digest':True,'paired_identical_initial_digests':True,'all15_epoch_orders_distinct_across_seeds':True,'both_geometry_arrays_reconstructed':True,'all18_frozen_before_evaluation':True,'all18_finite_parameters_moments_Adam_step1520':True,'all90_epochs_and360_GT_rows':True,'all_fidelity_and_temporal_means':True},'scope':'Primary target-only contrast and secondary student-renderer control, fixedCTD geometry/choices. All380update target-switch pilots from native400-warmed state.','limits':p['limits'],'compute':'18x380=6840new gradient updates; both800_area rasterizes4x pixels per step. GPU shared, walltimes not inference or speed benchmarks.'};dump(analysis/'report.json',report)
with zipfile.ZipFile(analysis/'source.zip','x') as z:z.write(pathlib.Path(__file__).resolve(),'summarize_teacher_rendering_control.py')
dump(root/'projects/dynamic-gaussian-ot-densification-research/experiments/e12-status-20261105.json',{'run':str(out),'report':report,'remote_publish_requested':False,'full4DGS_training_densification_implemented':False})
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(16,4.7),layout='constrained');colors={'no_second':'#2976a9','ctd128':'#d87a2a'};labels=['Native400','Teacher800->400','Both800->400']
for ax,key,title in zip(axes,['PSNR','LPIPS_alex_v01','temporal_residual256'],['20-view GT PSNR (higher)','GT LPIPS (lower)','Temporal residual256 (lower)']):
 for j,cap in enumerate(p['capacities']):
  means=[next(a for a in aggregates if a['capacity']==cap and a['condition']==condition)['metrics'][key]['mean'] for condition in p['conditions']];std=[next(a for a in aggregates if a['capacity']==cap and a['condition']==condition)['metrics'][key]['sample_std'] for condition in p['conditions']];ax.errorbar(np.arange(3)+(j-.5)*.08,means,yerr=std,fmt='o-',color=colors[cap],label=f'{cap}, K={p["capacities"][cap]}',capsize=3)
 ax.set_xticks(range(3),labels,rotation=20);ax.set_title(title);ax.grid(axis='y',alpha=.3)
axes[0].legend(fontsize=8);fig.suptitle('E12: teacher-target primary / matched-renderer secondary; mean +/- SD over3 future-order seeds');fig.savefig(analysis/'teacher-rendering-control.png',dpi=140);plt.close(fig)
print(json.dumps({'aggregates':aggregates,'paired_effects':paired,'split_effects':split_effect,'repeat_delta':repeat_delta,'verification':report['verification']},indent=2))
