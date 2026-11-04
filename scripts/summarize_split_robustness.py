#!/usr/bin/env python3
"""Aggregate paired robustness results; preserve budget42 and order-seed distinctions."""
import argparse,csv,json,os,pathlib,zipfile
import numpy as np
from run_ctd_comparison import dump,digest
ap=argparse.ArgumentParser();ap.add_argument('output',type=pathlib.Path);a=ap.parse_args();out=a.output.resolve();root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();assert out.is_relative_to(root);assert json.loads((out/'completion.json').read_text())['conditions']==5
runs=json.loads((out/'runs.json').read_text());assert {(r['splits'],r['seed']) for r in runs}=={(128,42),(256,42),(512,42),(256,43),(256,44)}
rows=[];paired=[];data={}
for run in runs:
 p=pathlib.Path(run['output']);m=json.loads((p/'manifest.json').read_text());assert m['protocol']['splits']==run['splits'] and m['protocol']['seed']==run['seed']
 c=json.loads((p/'validation/completion.json').read_text());assert c['status']=='completed';assert digest(p/'source.zip')==c['source_sha256']
 report=json.loads((p/'validation/analysis/report.json').read_text());assert report['input_hashes_unchanged'] and report['fixed1520steps_each'] and len(report['rows'])==5
 data[(run['splits'],run['seed'])]={'path':p,'rows':{r['arm']:r for r in report['rows']},'proposals':json.loads((p/'proposal-summary.json').read_text())}
 for r in report['rows']:
  assert r['gaussians']==(10000 if r['arm']=='continue' else 10000+run['splits']);rows.append(dict(splits=run['splits'],seed=run['seed'],**r))
 d=data[(run['splits'],run['seed'])]['rows'];paired.append({'splits':run['splits'],'seed':run['seed'],'window_minus_random_PSNR':d['window']['PSNR_after']-d['random']['PSNR_after'],'window_minus_global_PSNR':d['window']['PSNR_after']-d['global']['PSNR_after'],'global_minus_random_PSNR':d['global']['PSNR_after']-d['random']['PSNR_after'],'window_minus_continue_PSNR':d['window']['PSNR_after']-d['continue']['PSNR_after'],'window_temporal_reduction_vs_random_percent':100*(1-d['window']['temporal_after']/d['random']['temporal_after'])})
reference=data[(256,42)]
for seed in [43,44]:
 for name in ['window','global','first']:assert digest(reference['path']/(name+'-states.npy'))==digest(data[(256,seed)]['path']/(name+'-states.npy'))
for name in ['window','global','first']:
 ids=[data[(b,42)]['proposals']['methods'][name]['selected'] for b in [128,256,512]];assert ids[0]==ids[1][:128] and ids[1]==ids[2][:256]
reference_orders=json.loads((reference['path']/'validation/manifest.json').read_text())['orders']
order_sets={seed:set(map(tuple,json.loads((data[(256,seed)]['path']/'validation/manifest.json').read_text())['orders'])) for seed in [42,43,44]}
shared_orders={str(a)+'-'+str(b):len(order_sets[a]&order_sets[b]) for a,b in [(42,43),(42,44),(43,44)]}
assert max(shared_orders.values())==0, 'Order amendment must eliminate repeated epoch permutations across seeds'
for b in [128,512]:assert json.loads((data[(b,42)]['path']/'validation/manifest.json').read_text())['orders']==reference_orders
control_values=[data[(b,42)]['rows']['continue']['PSNR_after'] for b in [128,256,512]]
control_repeat_range=float(max(control_values)-min(control_values))
metrics=['PSNR_after','LPIPS_after','temporal_after','PSNR_delta_vs_continue'];seed_summary=[]
for arm in ['continue','window','global','first','random']:
 selected=[r for r in rows if r['splits']==256 and r['arm']==arm];seed_summary.append({'arm':arm,'n':3,'metrics':{k:{'mean':float(np.mean([r[k] for r in selected])),'sample_std':float(np.std([r[k] for r in selected],ddof=1))} for k in metrics}})
analysis=out/'analysis';analysis.mkdir(exist_ok=False)
with (analysis/'conditions.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
report={'paired':paired,'budget256_order_seed_summary':seed_summary,'verification':{'all5conditions_complete':True,'geometry_identical_across_order_seeds':True,'ranked_budget_sets_nested':True,'seed42_training_orders_identical':True,'seed42_unsplit_PSNR_repeat_range_dB':control_repeat_range,'same1520updates_each':True,'shared_epoch_permutations_across_seeds':shared_orders},'scope':'Base/CTD/local child partitions fixed. Seed changes training order and random proposal sampling only. Budget128/512 seed42 only. Offline fixed extra count, not training-time adaptive split.'};dump(analysis/'report.json',report)
with zipfile.ZipFile(analysis/'source.zip','x') as z:z.write(pathlib.Path(__file__).resolve(),'summarize_split_robustness.py')
dump(root/'projects/dynamic-gaussian-ot-densification-research/experiments/e05-robustness-status-20261104.json',{'run':str(out),'report':report,'adaptive_training_split_implemented':False});print(json.dumps(report,indent=2))
