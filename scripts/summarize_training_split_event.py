#!/usr/bin/env python3
"""Independent E06 ledger, active-state, optimizer-age and evaluation checks."""
import argparse,csv,json,os,pathlib,zipfile
import numpy as np
import torch
from appearance_model import Appearance
from training_split_core import digest_state,RATES
from run_ctd_comparison import dump,digest
ap=argparse.ArgumentParser();ap.add_argument('output',type=pathlib.Path);a=ap.parse_args();out=a.output.resolve();root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();assert out.is_relative_to(root)
completion=json.loads((out/'completion.json').read_text());assert completion['status']=='completed' and digest(out/'source.zip')==completion['source_sha256'];m=json.loads((out/'manifest.json').read_text());p=m['protocol'];events=json.loads((out/'event-decisions.json').read_text());training=json.loads((out/'training.json').read_text());traces=json.loads((out/'count-traces.json').read_text());summary=json.loads((out/'validation/summary.json').read_text());gt=json.loads((out/'validation/gt-per-view.json').read_text())
for path,h in m['inputs'].items():assert digest(pathlib.Path(path))==h
assert set(m['decision_views'])|set(m['probe_training_views'])==set(m['fit']) and not set(m['decision_views'])&set(m['probe_training_views']) and not set(m['fit'])&set(m['holdout'])
assert all(set(order)<=set(m['probe_training_views']) for order in m['probe_orders'])
warm=torch.load(out/'warmup.pt',map_location='cpu');warm_model=Appearance(warm['model']['source_sh'],warm['model']['source_logits'],10000);warm_model.load_state_dict(warm['model']);warm_opt=warm_model.optimizer(RATES);warm_opt.load_state_dict(warm['optimizer']);assert digest_state(warm_model,warm_opt)==warm['digest'] and warm['main_steps']==760
source=root/'outputs/ot/e05-split-20261104T103723Z';source_m=json.loads((source/'manifest.json').read_text());parents=np.load(source_m['parent_states'],mmap_mode='r');child=np.load(source/'child-states.npy',mmap_mode='r');rows=[]
for arm in p['arms']:
 ck=torch.load(out/'validation'/(arm+'-appearance.pt'),map_location='cpu');selected=ck['accepted_parents'];count=10000+len(selected);assert len(set(selected))==len(selected) and ck['main_steps']==1520 and ck['state_dict']['dc'].shape[0]==count and completion['counts'][arm]==count
 assert all(torch.isfinite(t).all() for t in ck['state_dict'].values())
 for state in ck['optimizer']['state'].values():
  assert float(state['step'])==1520
  assert torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()
 ledger=[e for e in events if e['arm']==arm];expected=[];current=10000
 assert len(ledger)==(0 if arm=='continue' else 16)
 for b,e in enumerate(ledger):
  assert e['block']==b and e['main_step']==760 and e['before_count']==current and e['live_unchanged_during_probe']
  scores=e['scores'];gain=np.mean([r['reference']-r['proposal'] for r in scores['windows']]);worst=max((r['proposal']-r['reference'])/max(r['reference'],1e-12) for r in scores['windows']);price=p['batch_size']*p['complexity_price_per_gaussian']
  assert abs(gain-scores['render_gain'])<1e-12 and abs(worst-scores['worst_window_relative_worsening'])<1e-12
  accept=bool(gain>price and worst<=p['max_window_relative_worsening']);assert accept==scores['accept'] and e['committed']==(arm.startswith('forced') or accept)
  if e['committed']:expected+=e['parents'];current+=16;assert e['commit_uses_untrained_proposal']
  else:assert e['rejected_rollback_exact']
  assert e['after_count']==current
 assert selected==expected and current==count
 if arm!='continue':
  expected_geometry=np.concatenate([np.array(parents,copy=True),child[:,np.array(selected,dtype=int)*2+1]],axis=1);expected_geometry[:,np.array(selected,dtype=int)]=child[:,np.array(selected,dtype=int)*2]
  assert np.array_equal(expected_geometry,np.load(out/'validation'/(arm+'-states.npy'),mmap_mode='r'))
 post=[r for r in training if r['arm']==arm];assert len(post)==10 and post[-1]['main_updates']==1520
 for phase in ['before','after']:
  views=[r for r in gt if r['arm']==arm and r['phase']==phase];assert sorted(r['test_index'] for r in views)==list(range(20))
  for metric in ['PSNR','SSIM','LPIPS_alex_v01']:assert abs(np.mean([r[metric] for r in views])-summary['official_20_view_gt'][arm+'_'+phase][metric])<1e-12
 row={'arm':arm,'gaussians':count,'committed_batches':sum(e['committed'] for e in ledger),'gate_passed_batches':sum(e['scores']['accept'] for e in ledger),'PSNR':summary['official_20_view_gt'][arm+'_after']['PSNR'],'LPIPS':summary['official_20_view_gt'][arm+'_after']['LPIPS_alex_v01'],'temporal_residual':summary['fixed_camera_temporal_delta_residual_256'][arm+'_after'],'holdout_PSNR_to_full':summary['fidelity_to_full_400'][arm+'_after']['holdout']['PSNR_to_full'],'probe_updates':len(ledger)*2*p['probe_steps']};rows.append(row)
analysis=out/'analysis';analysis.mkdir(exist_ok=False)
with (analysis/'comparison.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
report={'rows':rows,'verification':{'input_hashes_unchanged':True,'source_hash_verified':True,'warmup_state_digest_verified':True,'main_Adam_steps_each':1520,'all64_gate_scores_recomputed':True,'probe_live_state_and_rejection_rollback_verified':True,'accepted_geometry_reconstructed':True,'finite_parameters_and_moments':True,'20GTmeans_verified':True,'fit_only_decisions':True},'probe_updates_total':sum(r['probe_updates'] for r in rows),'scope':'Single event in fixed-geometry reduced-model appearance optimization. Teacher/child geometry precomputed; not full4DGS adaptive densification. Variable finalcount and unequal compute vs continue; no blanket equal-budget comparison.'};dump(analysis/'report.json',report)
with zipfile.ZipFile(analysis/'source.zip','x') as z:z.write(pathlib.Path(__file__).resolve(),'summarize_training_split_event.py')
dump(root/'projects/dynamic-gaussian-ot-densification-research/experiments/e06-status-20261104.json',{'run':str(out),'report':report,'full4DGS_training_densification_implemented':False});print(json.dumps(report,indent=2))
