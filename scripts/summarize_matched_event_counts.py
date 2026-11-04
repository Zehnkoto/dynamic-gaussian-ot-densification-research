"""Check matched-prefix followup and report paired equal-count comparisons."""
import argparse,json,os,pathlib,zipfile
import numpy as np
import torch
from run_ctd_comparison import dump,digest
ap=argparse.ArgumentParser();ap.add_argument('output',type=pathlib.Path);a=ap.parse_args();out=a.output.resolve();root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();assert out.is_relative_to(root)
c=json.loads((out/'completion.json').read_text());m=json.loads((out/'manifest.json').read_text());s=json.loads((out/'validation/summary.json').read_text());assert c['status']=='completed' and digest(out/'source.zip')==c['source_sha256']
source=pathlib.Path(m['source_event']);sm=json.loads((source/'manifest.json').read_text());old=json.loads((source/'validation/summary.json').read_text());assert m['post_orders']==sm['orders'][10:]
for p,h in m['inputs'].items():assert digest(pathlib.Path(p))==h
views=json.loads((out/'validation/gt-per-view.json').read_text())
for arm in m['protocol']['evaluated_arms']:
 for phase in ['before','after']:
  selected=[r for r in views if r['arm']==arm and r['phase']==phase];assert sorted(r['test_index'] for r in selected)==list(range(20))
  for metric in ['PSNR','SSIM','LPIPS_alex_v01']:assert abs(np.mean([r[metric] for r in selected])-s['official_20_view_gt'][arm+'_'+phase][metric])<1e-12
for arm in ['continue','gated_window','gated_random']:
 for metric,v in s['official_20_view_gt'][arm+'_after'].items():assert abs(v-old['official_20_view_gt'][arm+'_after'][metric])<1e-5
pool=json.loads((source/'proposal-summary.json').read_text());rows=[]
for kind in ['window','random']:
 prefix='prefix_'+kind;gated='gated_'+kind;ck=torch.load(out/'validation'/(prefix+'-appearance.pt'),map_location='cpu');count=10000+m['selected_counts'][prefix];assert ck['main_steps']==1520 and ck['selected_parents']==pool['methods'][kind]['selected'][:count-10000] and ck['state_dict']['dc'].shape[0]==count
 assert c['counts'][prefix]==c['counts'][gated]==count and all(torch.isfinite(t).all() for t in ck['state_dict'].values())
 for state in ck['optimizer']['state'].values():assert float(state['step'])==1520
 rows.append({'kind':kind,'gaussians':count,'gated_PSNR':s['official_20_view_gt'][gated+'_after']['PSNR'],'prefix_PSNR':s['official_20_view_gt'][prefix+'_after']['PSNR'],'gated_minus_prefix_PSNR':s['official_20_view_gt'][gated+'_after']['PSNR']-s['official_20_view_gt'][prefix+'_after']['PSNR'],'gated_LPIPS':s['official_20_view_gt'][gated+'_after']['LPIPS_alex_v01'],'prefix_LPIPS':s['official_20_view_gt'][prefix+'_after']['LPIPS_alex_v01'],'gated_temporal':s['fixed_camera_temporal_delta_residual_256'][gated+'_after'],'prefix_temporal':s['fixed_camera_temporal_delta_residual_256'][prefix+'_after']})
analysis=out/'analysis';analysis.mkdir(exist_ok=False);report={'rows':rows,'verification':{'input_hashes_unchanged':True,'warmup_and_post_order_matched':True,'same_finalcounts_each_pair':True,'main_Adam_steps_each':1520,'original_metric_parity':True,'finite_prefix_checkpoints':True,'20GTmeans_verified':True},'scope':'Exploratory matched-count followup. Prefix counts derived from fit-only gate decisions; no thresholds tuned. Different probe overhead, same live-model main schedule.'};dump(analysis/'report.json',report)
with zipfile.ZipFile(analysis/'source.zip','x') as z:z.write(pathlib.Path(__file__).resolve(),'summarize_matched_event_counts.py')
dump(root/'projects/dynamic-gaussian-ot-densification-research/experiments/e06-matched-count-status-20261104.json',{'run':str(out),'report':report});print(json.dumps(report,indent=2))
