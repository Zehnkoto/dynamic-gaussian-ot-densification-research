#!/usr/bin/env python3
"""Audit controlled camera/target/loss metrics and report every prespecified correlation."""
import argparse,csv,json,os,pathlib,zipfile
import numpy as np
from scipy.stats import spearmanr
from run_ctd_comparison import dump,digest
ap=argparse.ArgumentParser();ap.add_argument('output',type=pathlib.Path);out=ap.parse_args().output.resolve();root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();assert out.is_relative_to(root);read=lambda p:json.loads(p.read_text());m=read(out/'manifest.json');done=read(out/'completion.json');data=read(out/'measurements.json');assert done['status']=='completed' and done['training_updates']==0 and done['shared_state_unchanged'] and digest(out/'source.zip')==done['source_sha256']
for p,h in m['inputs'].items():assert digest(pathlib.Path(p))==h
assert len(data)==5000 and done['rows']==5000 and m['GT_native_vs_saved_max_error']<1e-7
conditions=['fit400','train400','test400','train800','test800'];metrics=m['protocol']['metrics'];groups={};max_identity=0.
for row in data:
 key=(row['arm'],row['phase'],row['condition']);groups.setdefault(key,[]).append(row);assert np.isfinite([row[k] for k in metrics]).all();err=abs(row['gt_mse']-row['teacher_mse']-row['teacher_gt_mse']-row['cross_term']);max_identity=max(max_identity,err);assert err<1e-7
for condition in conditions:
 entries=m['conditions'][condition];assert len(entries)==20 and len({e['time_index'] for e in entries})==20
 if condition!='fit400':assert [e['time_index'] for e in entries]==[e['time_index'] for e in m['conditions']['test800']]
 assert all(e['size']==(800 if condition.endswith('800') else 400) for e in entries)
means={}
for phase in [0,380]:
 for b in [-1]+list(range(24)):
  arm='reference' if b<0 else f'batch_{b:02d}'
  for condition in conditions:
   rows=groups[arm,phase,condition];assert [r['view'] for r in rows]==list(range(20)) and [r['time_index'] for r in rows]==[e['time_index'] for e in m['conditions'][condition]];means[arm,phase,condition]={k:float(np.mean([r[k] for r in rows])) for k in metrics}
for condition in conditions:
 for view in range(20):
  baseline=groups['reference',0,condition][view]['teacher_gt_mse']
  for phase in [0,380]:
   for b in [-1]+list(range(24)):
    arm='reference' if b<0 else f'batch_{b:02d}';assert groups[arm,phase,condition][view]['teacher_gt_mse']==baseline
source=root/'outputs/ot/e10-gain-prediction-20261104T135817Z';original=read(source/'measurements.json');oldgt=read(source/'validation/gt-per-view.json');design=read(source/'design.json');assert m['shared_digest']==design['shared_digest'] and [e['time_index'] for e in m['conditions']['fit400']]==design['monitor_views']['fit_decision'];max_monitor=0.;max_gt=0.
for row in original:
 if row['offset'] not in [0,380]:continue
 values=[r['teacher_mix'] for r in groups[row['arm'],row['offset'],'fit400']];max_monitor=max(max_monitor,float(np.max(abs(np.array(values)-row['fit_decision']))))
for row in oldgt:
 new=groups[row['arm'],380,'test800'][row['test_index']];max_gt=max(max_gt,abs(new['gt_psnr_u8']-row['PSNR']))
assert max_monitor<1e-6 and max_gt<1e-5,(max_monitor,max_gt)
gains=[];gainmap={}
for b in range(24):
 arm=f'batch_{b:02d}'
 for phase in [0,380]:
  for condition in conditions:
   ref=means['reference',phase,condition];candidate=means[arm,phase,condition];row={'batch':b,'phase':phase,'condition':condition,**{k:(candidate[k]-ref[k] if k=='gt_psnr_u8' else ref[k]-candidate[k]) for k in metrics}};assert abs(row['gt_mse']-row['teacher_mse']-row['cross_term'])<2e-7;assert abs(row['teacher_gt_mse'])<1e-15;gains.append(row);gainmap[b,phase,condition]=row

def corr(a,b):
 v=float(spearmanr(a,b).statistic);return v if np.isfinite(v) else None
outcomes={f'final_{c}_gt_mse':[gainmap[b,380,c]['gt_mse'] for b in range(24)] for c in conditions};outcomes['final_official_GT_PSNR']=[gainmap[b,380,'test800']['gt_psnr_u8'] for b in range(24)];primary=[]
for predictor in ['teacher_mix','teacher_mse','gt_mix','gt_mse']:
 values=[gainmap[b,0,'fit400'][predictor] for b in range(24)]
 for outcome,target in outcomes.items():primary.append({'predictor':predictor,'outcome':outcome,'spearman':corr(values,target)})
transfer=[]
for a,b in [('train400','test400'),('train800','test800'),('test400','test800'),('train400','train800'),('fit400','train400')]:transfer.append({'a':a,'b':b,'metric':'final_gt_mse_gain','spearman':corr([gainmap[i,380,a]['gt_mse'] for i in range(24)],[gainmap[i,380,b]['gt_mse'] for i in range(24)])})
for a,b in [('gt_mse','gt_mse_u8'),('gt_mse_u8','gt_psnr_u8')]:transfer.append({'a':a,'b':b,'metric':'final_test800_gain','spearman':corr([gainmap[i,380,'test800'][a] for i in range(24)],[gainmap[i,380,'test800'][b] for i in range(24)])})
decomposition=[]
for phase in [0,380]:
 for condition in conditions:
  teacher=np.array([gainmap[b,phase,condition]['teacher_mse'] for b in range(24)]);real=np.array([gainmap[b,phase,condition]['gt_mse'] for b in range(24)]);cross=np.array([gainmap[b,phase,condition]['cross_term'] for b in range(24)]);decomposition.append({'phase':phase,'condition':condition,'teacher_GT_gain_rank_spearman':corr(teacher,real),'mean_abs_cross_gain_over_mean_abs_teacher_gain':float(np.mean(abs(cross))/max(np.mean(abs(teacher)),1e-30)),'positive_teacher_gain_negative_GT_gain':int(((teacher>0)&(real<0)).sum()),'teacher_GT_MSE':means['reference',phase,condition]['teacher_gt_mse']})
analysis=out/'analysis';analysis.mkdir()
for filename,rows in [('metric-gains.csv',gains),('primary-correlations.csv',primary),('transfer-correlations.csv',transfer),('mse-decomposition.csv',decomposition)]:
 with (analysis/filename).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
report={'training_updates':0,'candidate_batches':24,'phases':[0,380],'primary_correlations':primary,'rank_transfer':transfer,'mse_decomposition':decomposition,'verification':{'all_input_and_source_hashes':True,'all5000_rows_finite_and_indexed':True,'all5000_MSE_identities':True,'paired20_times_for_camera_resolution_conditions':True,'fixed_teacher_GT_targets_across_arms_and_phases':True,'E10_fit_teacher_loss_parity':True,'E10_500_GT_PSNR_parity':True,'shared1140_state_unchanged':True},'max_MSE_identity_error':max_identity,'max_E10_monitor_error':max_monitor,'max_E10_test_PSNR_error':max_gt,'limitations':m['protocol']['limitations'],'scope':'Read-only controlled diagnostic of frozen states, not a new trained split policy.'};dump(analysis/'report.json',report)
with zipfile.ZipFile(analysis/'source.zip','x') as z:z.write(pathlib.Path(__file__).resolve(),'summarize_score_target_alignment.py')
dump(root/'projects/dynamic-gaussian-ot-densification-research/experiments/e11-status-20261104.json',{'run':str(out),'report':report,'remote_publish_requested':False,'full4DGS_training_densification_implemented':False})
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained');preds=['teacher_mix','teacher_mse','gt_mix','gt_mse'];target_names=list(outcomes);matrix=np.array([[next(r['spearman'] for r in primary if r['predictor']==p and r['outcome']==o) for o in target_names] for p in preds],dtype=float);axes[0].imshow(matrix,cmap='coolwarm',vmin=-1,vmax=1,aspect='auto');axes[0].set_yticks(range(4),['Teacher mixed','Teacher MSE','GT mixed','GT MSE']);axes[0].set_xticks(range(6),['Fit400','Train400','Test400','Train800','Test800','GT PSNR'],rotation=40,ha='right');axes[0].set_title('Immediate Fit400 score vs final gains')
for i in range(4):
 for j in range(6):axes[0].text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center')
for ax,a,b,title in [(axes[1],'train400','test400','Same20 times,400px: camera transfer'),(axes[2],'test400','test800','Same20 test views: resolution transfer')]:
 x=[gainmap[i,380,a]['gt_mse'] for i in range(24)];y=[gainmap[i,380,b]['gt_mse'] for i in range(24)];ax.scatter(x,y);ax.set_xlabel(a+' final GT MSE gain');ax.set_ylabel(b+' final GT MSE gain');ax.set_title(title+f'\nrho={corr(x,y):.3f}');ax.grid(alpha=.3)
fig.suptitle('E11: frozen models only; same-time camera/resolution controls; exploratory24-batch ranks');fig.savefig(analysis/'score-target-alignment.png',dpi=140);plt.close(fig)
print(json.dumps(report,indent=2))
