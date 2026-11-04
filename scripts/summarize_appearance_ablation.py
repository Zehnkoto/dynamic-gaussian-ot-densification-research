#!/usr/bin/env python3
"""Validate saved sweep outputs; aggregate view-order seeds without selecting weights."""
import argparse,csv,json,os,pathlib,zipfile
import numpy as np
import torch
from run_ctd_comparison import digest,dump
ap=argparse.ArgumentParser();ap.add_argument('output',type=pathlib.Path);a=ap.parse_args();out=a.output.resolve();root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();assert out.is_relative_to(root)
assert json.loads((out/'completion.json').read_text())['conditions']==9
runs=json.loads((out/'runs.json').read_text());assert {(r['mode'],r['seed']) for r in runs}=={(m,s) for m in ['joint','sh','opacity'] for s in [42,43,44]}
rows=[]
for r in runs:
 p=pathlib.Path(r['output']);s=json.loads((p/'summary.json').read_text());m=json.loads((p/'manifest.json').read_text());c=json.loads((p/'completion.json').read_text());views=json.loads((p/'gt-per-view.json').read_text())
 assert c['status']=='completed' and digest(p/'source.zip')==c['source_sha256']
 assert m['protocol']['seed']==r['seed'] and len(m['fit_indices'])==76 and len(m['holdout_indices'])==74
 assert m['trainable_parameters_per_arm']==10000*{'joint':49,'sh':48,'opacity':1}[r['mode']]
 for path,h in m['inputs'].items():assert digest(pathlib.Path(path))==h
 for arm in ['shared','perframe']:
  ck=torch.load(p/(arm+'-appearance.pt'),map_location='cpu');assert ck['steps']==1520
  assert all(torch.isfinite(v).all() for v in ck['state_dict'].values())
  frozen={'joint':[],'sh':['opacity'],'opacity':['dc','rest']}[r['mode']]
  for key in frozen:assert torch.count_nonzero(ck['state_dict'][key])==0
  subset=[v for v in views if v['arm']==arm and v['phase']=='after'];assert sorted(v['test_index'] for v in subset)==list(range(20))
  row={'mode':r['mode'],'seed':r['seed'],'arm':arm,'output':str(p)}
  for metric in ['PSNR','SSIM','LPIPS_alex_v01']:
   value=float(np.mean([v[metric] for v in subset]));assert abs(value-s['official_20_view_gt'][arm+'_after'][metric])<1e-12;row[metric]=value
  row['temporal_residual']=s['fixed_camera_temporal_delta_residual_256'][arm+'_after'];row['holdout_PSNR_to_full']=s['fidelity_to_full_400'][arm+'_after']['holdout']['PSNR_to_full'];rows.append(row)
analysis=out/'analysis';analysis.mkdir(exist_ok=False)
with (analysis/'per-seed.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
metrics=['PSNR','SSIM','LPIPS_alex_v01','temporal_residual','holdout_PSNR_to_full'];groups=[];paired=[]
for mode in ['opacity','sh','joint']:
 for arm in ['shared','perframe']:
  selected=[r for r in rows if r['mode']==mode and r['arm']==arm];assert len(selected)==3
  groups.append({'mode':mode,'arm':arm,'n':3,'metrics':{k:{'mean':float(np.mean([v[k] for v in selected])),'sample_std':float(np.std([v[k] for v in selected],ddof=1))} for k in metrics}})
 for seed in [42,43,44]:
  x={r['arm']:r for r in rows if r['mode']==mode and r['seed']==seed};paired.append({'mode':mode,'seed':seed,'shared_minus_perframe_PSNR':x['shared']['PSNR']-x['perframe']['PSNR'],'shared_temporal_reduction_percent':100*(1-x['shared']['temporal_residual']/x['perframe']['temporal_residual'])})
report={'verification':'All9 completed; input/source hashes, parameter counts, fixed1520steps, frozen residuals, finite checkpoints and20GTmeans checked','seed_scope':'Training-view order only; same base model and CTD geometry','groups':groups,'paired':paired,'limitations':['3 view-order seeds only, not full-pipeline seeds','Single scene and budget','Perframe control has shared slot residuals; not official GHAP fine-tuning','No adaptive split or standalone compressed deformation']};dump(analysis/'report.json',report)
with zipfile.ZipFile(analysis/'source.zip','x') as z:z.write(pathlib.Path(__file__).resolve(),'summarize_appearance_ablation.py')
dump(root/'projects/dynamic-gaussian-ot-densification-research/experiments/e04-ablation-status-20261103.json',{'run':str(out),'report':report,'adaptive_split_completed':False})
print(json.dumps(report,indent=2))
