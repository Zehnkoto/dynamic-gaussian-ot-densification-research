#!/usr/bin/env python3
"""Posthoc E20 window-boundary and metric-direction diagnostic; no model selection."""
import argparse,json,pathlib
import numpy as np

ARMS=['window_full_geometry','window_position_only']
BASE='fixed_E19_mean_beta0'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('output',type=pathlib.Path);args=ap.parse_args();out=args.output.resolve()
 assert json.loads((out/'audit.json').read_text())['status']=='passed'
 report=json.loads((out/'report.json').read_text());rows={r['arm']:r for r in report['rows']}
 temporal=json.loads((out/'temporal-per-pair.json').read_text());T={(r['arm'],r['time_index']):r['residual'] for r in temporal}
 gt=json.loads((out/'gt-per-view.json').read_text());G={(r['arm'],r['view']):r for r in gt}
 summary={'label':'posthoc diagnostic after E20 training and GT evaluation; no arm or parameter selection','window_boundary_definition':'adjacent time pair i-1 -> i with i divisible by 5','boundary_pairs':39,'within_pairs':160,'arms':{}}
 for arm in ARMS:
  record={}
  for kind,indices in [('boundary',[i for i in range(1,200) if i%5==0]),('within',[i for i in range(1,200) if i%5!=0])]:
   old=float(np.mean([T[(BASE,i)] for i in indices]));new=float(np.mean([T[(arm,i)] for i in indices]))
   record[kind]={'fixed_residual':old,'new_residual':new,'delta':new-old,'relative_delta_percent':100*(new-old)/old,'pairs':len(indices)}
  record['GT_beneficial_views']={metric:sum(direction*(G[(arm,i)][metric]-G[(BASE,i)][metric])>0 for i in range(20)) for metric,direction in [('PSNR',1),('SSIM',1),('LPIPS_alex_v01',-1)]}
  record['full12_cost_reduction_percent']={split:100*(rows[BASE]['distribution'][split]['full12_weighted_cost']-rows[arm]['distribution'][split]['full12_weighted_cost'])/rows[BASE]['distribution'][split]['full12_weighted_cost'] for split in ['fit','holdout']}
  record['switches_total']=rows[arm]['switches_total'];summary['arms'][arm]=record
 path=out/'boundary-analysis.json'
 with path.open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
