#!/usr/bin/env python3
"""Independent ledger, schedule, geometry, Adam, exposure and evaluation audit."""
import argparse,csv,json,os,pathlib,zipfile
import numpy as np
import torch
from run_ctd_comparison import dump,digest
from appearance_model import Appearance
from training_split_core import digest_state,RATES
p=argparse.ArgumentParser();p.add_argument('output',type=pathlib.Path);out=p.parse_args().output.resolve();root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();assert out.is_relative_to(root)
read=lambda p:json.loads(p.read_text());m=read(out/'manifest.json');protocol=m['protocol'];complete=read(out/'completion.json');assert complete['status']=='completed' and digest(out/'source.zip')==complete['source_sha256']
for path,h in m['inputs'].items():assert digest(pathlib.Path(path))==h
source=root/'outputs/ot/e06-training-event-20261104T113535Z';e07=root/'outputs/ot/e07-gate-sensitivity-20261104T121416Z';sm=read(source/'manifest.json');assert sm['orders']==m['orders'] and sm['decision_views']==m['decision_views']
assert set(m['decision_views']).isdisjoint(m['probe_training_views']) and set(m['decision_views'])|set(m['probe_training_views'])==set(m['fit']) and set(m['fit']).isdisjoint(m['holdout'])
for b,order in enumerate(m['probe_orders']):assert order[:20]==sm['probe_orders'][b] and set(order)<=set(m['probe_training_views'])
warm=torch.load(source/'warmup.pt',map_location='cpu');model=Appearance(warm['model']['source_sh'],warm['model']['source_logits'],10000);model.load_state_dict(warm['model']);opt=model.optimizer(RATES);opt.load_state_dict(warm['optimizer']);assert digest_state(model,opt)==warm['digest'] and warm['main_steps']==760
proposal=root/'outputs/ot/e05-split-20261104T103723Z';pm=read(proposal/'manifest.json');geometry=np.load(pm['parent_states'],mmap_mode='r');children=np.load(proposal/'child-states.npy',mmap_mode='r');candidates=read(proposal/'proposal-summary.json')['methods']['window']['selected']
events=read(out/'event-decisions.json');training=read(out/'training.json');s=read(out/'validation/summary.json');gt=read(out/'validation/gt-per-view.json');rows=[]
for name,policy in protocol['policies'].items():
 ledger=[e for e in events if e['arm']==name];assert len(ledger)==16;accepted=[];count=10000;expected_schedule={b:step for step,start,end in policy['schedule'] for b in range(start,end)};integral=760*10000
 for b,e in enumerate(ledger):
  assert e['block']==b and e['main_step']==expected_schedule[b] and e['parents']==candidates[b*16:(b+1)*16] and e['before_count']==count and e['fit_CTD_gain']>0
  score=e['scores'];gain=np.mean([w['reference']-w['proposal'] for w in score['windows']]);worst=max((w['proposal']-w['reference'])/max(w['reference'],1e-12) for w in score['windows']);accept=bool(gain>16e-7 and worst<=.01)
  assert abs(gain-score['render_gain'])<1e-12 and abs(worst-score['worst_window_relative_worsening'])<1e-12 and accept==score['accept'] and e['committed']==bool(accept or policy['forced'])
  assert e['probe_updates_each_counterfactual']==policy['probe_steps'] and e['live_unchanged_during_probe']
  if e['committed']:accepted+=e['parents'];count+=16;assert e['commit_uses_untrained_proposal']
  else:assert e['rejected_rollback_exact']
  assert e['after_count']==count
 assert len(set(accepted))==len(accepted)
 ck=torch.load(out/'validation'/(name+'-appearance.pt'),map_location='cpu');assert ck['accepted_parents']==accepted and ck['main_steps']==1520 and len(ck['state_dict']['dc'])==count and complete['counts'][name]==count
 for v in ck['state_dict'].values():assert torch.isfinite(v).all()
 for st in ck['optimizer']['state'].values():assert float(st['step'])==1520 and torch.isfinite(st['exp_avg']).all() and torch.isfinite(st['exp_avg_sq']).all()
 ids=np.array(accepted,dtype=int);expected=np.concatenate([np.array(geometry,copy=True),children[:,ids*2+1]],1);expected[:,ids]=children[:,ids*2];assert np.array_equal(expected,np.load(out/'validation'/(name+'-states.npy'),mmap_mode='r'));del expected
 post=[t for t in training if t['arm']==name];assert len(post)==10
 for e,t in enumerate(post):
  start=760+e*76;active=10000+sum(len(ev['parents']) for ev in ledger if ev['committed'] and ev['main_step']<=start);integral+=76*active;assert t['gaussians']==active and t['main_updates']==start+76 and t['main_gaussian_updates_cumulative']==integral
 rows.append({'arm':name,'reused':False,'gaussians':count,'accepted_batches':len(accepted)//16,'main_gaussian_updates':integral,'discarded_probe_updates':32*policy['probe_steps'],**s['official_20_view_gt'][name+'_after'],'holdout_PSNR_to_full':s['fidelity_to_full_400'][name+'_after']['holdout']['PSNR_to_full'],'temporal_residual':s['fixed_camera_temporal_delta_residual_256'][name+'_after']})
for name,src,oldname,probe in [('once_forced',source,'forced_window',640),('once_immediate',e07,'price1e7_probe0',0),('once_probe20',source,'gated_window',640),('continue',source,'continue',0)]:
 old=read(src/'validation/summary.json');ck=torch.load(src/'validation'/(oldname+'-appearance.pt'),map_location='cpu');n=len(ck['state_dict']['dc'])
 for metric,value in s['official_20_view_gt'][name+'_after'].items():assert abs(value-old['official_20_view_gt'][oldname+'_after'][metric])<1e-5
 rows.append({'arm':name,'reused':True,'gaussians':n,'accepted_batches':(n-10000)//16,'main_gaussian_updates':760*10000+760*n,'discarded_probe_updates':probe,**s['official_20_view_gt'][name+'_after'],'holdout_PSNR_to_full':s['fidelity_to_full_400'][name+'_after']['holdout']['PSNR_to_full'],'temporal_residual':s['fixed_camera_temporal_delta_residual_256'][name+'_after']})
for name in s['active_gaussian_counts']:
 for phase in ['before','after']:
  views=[r for r in gt if r['arm']==name and r['phase']==phase];assert sorted(r['test_index'] for r in views)==list(range(20))
  for metric in ['PSNR','SSIM','LPIPS_alex_v01']:assert abs(np.mean([r[metric] for r in views])-s['official_20_view_gt'][name+'_'+phase][metric])<1e-12
by={r['arm']:r for r in rows};assert by['late_forced']['gaussians']==by['twice_forced']['gaussians']==by['once_forced']['gaussians']==10256
pairs=[]
for a,b in [('twice_forced','once_forced'),('twice_forced','late_forced'),('twice_immediate','once_immediate'),('twice_probe20','once_probe20')]:pairs.append({'a':a,'b':b,'PSNR_a_minus_b':by[a]['PSNR']-by[b]['PSNR'],'count_a_minus_b':by[a]['gaussians']-by[b]['gaussians']})
analysis=out/'analysis';analysis.mkdir()
with (analysis/'comparison.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
report={'rows':rows,'pairs':pairs,'verification':{'input_hashes':True,'warm_digest':True,'same_main_and_probe_orders':True,'all64_new_event_gates_and_schedules_recomputed':True,'accepted_geometry':True,'Adam_step1520_and_finite':True,'main_count_integral':True,'all20GT_means':True,'all4_reused_control_metric_parity':True},'limitations':protocol['limitations'],'scope':'One/two predetermined capacity events in fixed-geometry appearance optimization. Original protocol scope sentence inherited single-event wording; policies/design define the two-event experiment. No numerical protocol change.'};dump(analysis/'report.json',report)
with zipfile.ZipFile(analysis/'source.zip','x') as z:z.write(pathlib.Path(__file__).resolve(),'summarize_two_event_schedule.py')
dump(root/'projects/dynamic-gaussian-ot-densification-research/experiments/e08-status-20261104.json',{'run':str(out),'report':report,'full4DGS_training_densification_implemented':False})
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
names=['once_forced','twice_forced','late_forced','once_immediate','twice_immediate','once_probe20','twice_probe20'];labels=['Forced@760*','Forced@760+1140','Forced@1140','Gate0@760*','Gate0@760+1140','Gate20@760*','Gate20@760+1140'];fig,axes=plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
for ax,key,title in zip(axes,['gaussians','PSNR','main_gaussian_updates'],['Final Gaussian count','20-view PSNR (dB)','Main Gaussian-update integral']):
 values=[by[n][key] for n in names];ax.plot(range(len(names)),values,'o');ax.set_xticks(range(len(names)),labels,rotation=45,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.3)
fig.suptitle('E08: event timing, same ordered256 candidates / 1520 main updates; * reused controls');fig.savefig(analysis/'two-event-schedule.png',dpi=140);plt.close(fig)
print(json.dumps(report,indent=2))
