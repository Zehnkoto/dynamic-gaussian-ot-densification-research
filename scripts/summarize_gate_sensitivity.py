#!/usr/bin/env python3
"""Audit each E07 gate, accepted geometry, optimizer, input and evaluation mean."""
import argparse,csv,json,os,pathlib,zipfile
import numpy as np
import torch
from run_ctd_comparison import dump,digest
from appearance_model import Appearance
from training_split_core import digest_state,RATES
p=argparse.ArgumentParser();p.add_argument('output',type=pathlib.Path);out=p.parse_args().output.resolve();root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();assert out.is_relative_to(root)
read=lambda p:json.loads(p.read_text())
m=read(out/'manifest.json');protocol=m['protocol'];complete=read(out/'completion.json');assert complete['status']=='completed' and digest(out/'source.zip')==complete['source_sha256']
for path,h in m['inputs'].items():assert digest(pathlib.Path(path))==h
source=root/'outputs/ot/e06-training-event-20261104T113535Z';sm=read(source/'manifest.json');old=read(source/'validation/summary.json');assert sm['orders']==m['orders']
assert set(m['decision_views']).isdisjoint(m['probe_training_views']) and set(m['decision_views'])|set(m['probe_training_views'])==set(m['fit']) and set(m['fit']).isdisjoint(m['holdout'])
for b,order in enumerate(m['probe_orders']):
 rng=np.random.default_rng(np.random.SeedSequence([42,2718,b]));stream=[]
 while len(stream)<80:stream+=rng.permutation(m['probe_training_views']).tolist()
 assert stream[:80]==order and order[:20]==sm['probe_orders'][b]
warm=torch.load(source/'warmup.pt',map_location='cpu');model=Appearance(warm['model']['source_sh'],warm['model']['source_logits'],10000);model.load_state_dict(warm['model']);opt=model.optimizer(RATES);opt.load_state_dict(warm['optimizer']);assert digest_state(model,opt)==warm['digest']
proposal=root/'outputs/ot/e05-split-20261104T103723Z';pm=read(proposal/'manifest.json');geometry=np.load(pm['parent_states'],mmap_mode='r');children=np.load(proposal/'child-states.npy',mmap_mode='r');candidates=read(proposal/'proposal-summary.json')['methods']['window']['selected']
events=read(out/'event-decisions.json');training=read(out/'training.json');s=read(out/'validation/summary.json');gt=read(out/'validation/gt-per-view.json');rows=[]
for name,policy in list(protocol['policies'].items())+[('price1e7_probe20',{'price':1e-7,'probe_steps':20})]:
 reused=name=='price1e7_probe20';ledger=[e for e in read(source/'event-decisions.json') if e['arm']=='gated_window'] if reused else [e for e in events if e['arm']==name];assert len(ledger)==16;accepted=[];count=10000
 for b,e in enumerate(ledger):
  assert e['block']==b and e['parents']==candidates[b*16:(b+1)*16] and e['before_count']==count and e['fit_CTD_gain']>0
  score=e['scores'];gain=np.mean([w['reference']-w['proposal'] for w in score['windows']]);worst=max((w['proposal']-w['reference'])/max(w['reference'],1e-12) for w in score['windows']);accept=bool(gain>16*policy['price'] and worst<=.01)
  assert abs(gain-score['render_gain'])<1e-12 and abs(worst-score['worst_window_relative_worsening'])<1e-12 and accept==score['accept']==e['committed']
  assert e['probe_updates_each_counterfactual']==policy['probe_steps'] and e['live_unchanged_during_probe']
  if accept:accepted+=e['parents'];count+=16;assert e['commit_uses_untrained_proposal']
  else:assert e['rejected_rollback_exact']
  assert e['after_count']==count
 ckpath=(source/'validation/gated_window-appearance.pt') if reused else out/'validation'/(name+'-appearance.pt');ck=torch.load(ckpath,map_location='cpu');assert ck['accepted_parents']==accepted and ck['main_steps']==1520 and len(ck['state_dict']['dc'])==count
 for v in ck['state_dict'].values():assert torch.isfinite(v).all()
 for st in ck['optimizer']['state'].values():assert float(st['step'])==1520 and torch.isfinite(st['exp_avg']).all() and torch.isfinite(st['exp_avg_sq']).all()
 ids=np.array(accepted,dtype=int);expected=np.concatenate([np.array(geometry,copy=True),children[:,ids*2+1]],1);expected[:,ids]=children[:,ids*2];actual=(source/'validation/gated_window-states.npy') if reused else out/'validation'/(name+'-states.npy');assert np.array_equal(expected,np.load(actual,mmap_mode='r'));del expected
 if not reused:
  post=[t for t in training if t['arm']==name];assert len(post)==10 and post[-1]['main_updates']==1520
 row={'arm':name,'price':policy['price'],'probe_steps':policy['probe_steps'],'reused_E06':reused,'gaussians':count,'accepted_batches':len(accepted)//16,'probe_updates':32*policy['probe_steps'],**s['official_20_view_gt'][name+'_after'],'holdout_PSNR_to_full':s['fidelity_to_full_400'][name+'_after']['holdout']['PSNR_to_full'],'temporal_residual':s['fixed_camera_temporal_delta_residual_256'][name+'_after']};rows.append(row)
for name in s['active_gaussian_counts']:
 for phase in ['before','after']:
  views=[r for r in gt if r['arm']==name and r['phase']==phase];assert sorted(r['test_index'] for r in views)==list(range(20))
  for metric in ['PSNR','SSIM','LPIPS_alex_v01']:assert abs(np.mean([r[metric] for r in views])-s['official_20_view_gt'][name+'_'+phase][metric])<1e-12
for name,oldname in [('continue','continue'),('price1e7_probe20','gated_window')]:
 for metric,value in s['official_20_view_gt'][name+'_after'].items():assert abs(value-old['official_20_view_gt'][oldname+'_after'][metric])<1e-5
analysis=out/'analysis';analysis.mkdir()
with (analysis/'comparison.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
report={'rows':rows,'continue':s['official_20_view_gt']['continue_after'],'verification':{'input_hashes':True,'warm_digest':True,'nested_probe_orders':True,'all80_gate_decisions_recomputed':True,'accepted_geometry':True,'Adam_step1520_and_finite':True,'all20GT_means':True,'reused_control_metric_parity':True},'limitations':['One factor at a time; no price/length interaction tested','Single scene/seed/event; no test-selected winner','Fixed teacher/child geometry, appearance-only training','Decision views belong to fit; unequal final counts and probe compute','Reused central cell is not a new independent repeat']};dump(analysis/'report.json',report)
with zipfile.ZipFile(analysis/'source.zip','x') as z:z.write(pathlib.Path(__file__).resolve(),'summarize_gate_sensitivity.py')
dump(root/'projects/dynamic-gaussian-ot-densification-research/experiments/e07-status-20261104.json',{'run':str(out),'report':report,'full4DGS_training_densification_implemented':False})
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
for ax,key,title in zip(axes,['gaussians','PSNR','probe_updates'],['Actual Gaussian count','20-view PSNR (dB)','Discarded probe updates']):
 labels=['price=0 / 20','price=5e-7 / 20','price=1e-7 / 0','price=1e-7 / 80','price=1e-7 / 20*'];ax.bar(range(5),[r[key] for r in rows]);ax.set_xticks(range(5),labels,rotation=35,ha='right');ax.set_title(title)
 if key=='gaussians':ax.set_ylim(10000,10270)
 if key=='PSNR':ax.set_ylim(37.5,38.2)
fig.suptitle('E07: gate sensitivity — fixed warmup, window candidates, one scene/seed; * reused E06');fig.savefig(analysis/'gate-sensitivity.png',dpi=140);plt.close(fig)
print(json.dumps(report,indent=2))
