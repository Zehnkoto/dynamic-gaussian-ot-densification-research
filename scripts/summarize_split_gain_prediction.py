#!/usr/bin/env python3
"""Independently audit matched-horizon marginal gains and exploratory rank correlations."""
import argparse,csv,hashlib,json,os,pathlib,zipfile
import numpy as np
import torch
from scipy.stats import spearmanr
from appearance_model import Appearance
from training_split_core import digest_state,RATES
from run_ctd_comparison import dump,digest
p=argparse.ArgumentParser();p.add_argument('output',type=pathlib.Path);out=p.parse_args().output.resolve();root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();assert out.is_relative_to(root)
read=lambda p:json.loads(p.read_text());m=read(out/'manifest.json');protocol=m['protocol'];done=read(out/'completion.json');assert done['status']=='completed' and digest(out/'source.zip')==done['source_sha256']
for path,h in m['inputs'].items():assert digest(pathlib.Path(path))==h
source=root/'outputs/ot/e09-current-state-reranking-20261104T133944Z';source_m=read(source/'manifest.json');selected=read(source/'selection.json');design=read(out/'design.json');freeze=read(out/'training-complete.json');measurements=read(out/'measurements.json');training=read(out/'training.json');gt=read(out/'validation/gt-per-view.json');first=selected['first_parents'];pool=selected['candidate_batches'];assert design['first_parents']==first and design['candidate_batches']==pool and design['future_orders']==source_m['orders'][15:] and m['orders']==source_m['orders']
assert design['monitor_views']=={'fit_decision':source_m['decision_views'],'holdout':source_m['holdout']} and set(m['fit']).isdisjoint(m['holdout']) and set(source_m['decision_views'])<=set(m['fit'])
shared=torch.load(source/'before-second.pt',map_location='cpu');live=Appearance(shared['model']['source_sh'],shared['model']['source_logits'],10128);live.load_state_dict(shared['model']);opt=live.optimizer(RATES);opt.load_state_dict(shared['optimizer']);assert digest_state(live,opt)==shared['digest']==design['shared_digest'] and shared['main_steps']==1140
assert freeze['all25_frozen_before_GT'] and done['all25_frozen_before_GT'] and freeze['new_gradient_updates']==done['new_gradient_updates']==9500
assert len(freeze['checkpoint_hashes'])==25 and (out/'training-complete.json').stat().st_mtime<=(out/'validation/gt-per-view.json').stat().st_mtime
proposal=root/'outputs/ot/e05-split-20261104T103723Z';pm=read(proposal/'manifest.json');gains=np.load(proposal/'children.npz')['gains'];times=np.load(pathlib.Path(pm['source_cache'])/'times.npy');bins=np.minimum((times*5).astype(int),4);window=np.stack([gains[[i for i in m['fit'] if bins[i]==b]].mean(0) for b in range(5)]).min(0);parents=np.load(pm['parent_states'],mmap_mode='r');children=np.load(proposal/'child-states.npy',mmap_mode='r')
assert len(measurements)==100 and len(training)==125 and len(gt)==500
means={};gtmeans={}
for batch in [-1]+list(range(24)):
 name='reference' if batch<0 else f'batch_{batch:02d}';added=[] if batch<0 else pool[batch];ids=first+added;count=10128+len(added);path=out/'validation'/(name+'-appearance.pt');assert digest(path)==freeze['checkpoint_hashes'][path.name] and path.stat().st_mtime<=(out/'training-complete.json').stat().st_mtime
 ck=torch.load(path,map_location='cpu');assert ck['start_digest']==shared['digest'] and ck['main_steps']==1520 and ck['added_parents']==added and len(ck['state_dict']['dc'])==count and len(set(ids))==len(ids)
 for v in ck['state_dict'].values():assert torch.isfinite(v).all()
 for state in ck['optimizer']['state'].values():assert float(state['step'])==1520 and torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()
 geo=np.concatenate([np.array(parents,copy=True),children[:,np.array(ids)*2+1]],1);geo[:,ids]=children[:,np.array(ids)*2];assert hashlib.sha256(np.ascontiguousarray(geo).tobytes()).hexdigest()==ck['geometry_sha256'];del ck,geo
 arm_measure=[r for r in measurements if r['arm']==name];assert sorted(r['offset'] for r in arm_measure)==protocol['horizons'];means[name]={}
 for r in arm_measure:
  assert len(r['fit_decision'])==20 and len(r['holdout'])==74 and np.isfinite(r['fit_decision']+r['holdout']).all();means[name][r['offset']]={s:float(np.mean(r[s])) for s in ['fit_decision','holdout']}
 post=[r for r in training if r['arm']==name];assert len(post)==5
 for e,r in enumerate(post):assert r['offset']==(e+1)*76 and r['main_steps']==1140+(e+1)*76 and r['gaussians']==count and np.isfinite(r['mean_epoch_loss'])
 views=[r for r in gt if r['arm']==name];assert sorted(r['test_index'] for r in views)==list(range(20));gtmeans[name]={metric:float(np.mean([r[metric] for r in views])) for metric in ['PSNR','SSIM','LPIPS_alex_v01']};assert np.isfinite(list(gtmeans[name].values())).all()
rows=[]
for b,ids in enumerate(pool):
 name=f'batch_{b:02d}';row={'batch':b,'window_CTD_gain':float(window[ids].sum()),'mean_CTD_gain':float(gains[m['fit']][:,ids].sum(1).mean())}
 for h in protocol['horizons']:
  for s,key in [('fit_decision','fit_loss_gain'),('holdout','holdout_loss_gain')]:row[f'{key}_{h}']=means['reference'][h][s]-means[name][h][s]
 row.update({'GT_PSNR_gain_380':gtmeans[name]['PSNR']-gtmeans['reference']['PSNR'],'GT_SSIM_gain_380':gtmeans[name]['SSIM']-gtmeans['reference']['SSIM'],'GT_LPIPS_decrease_380':gtmeans['reference']['LPIPS_alex_v01']-gtmeans[name]['LPIPS_alex_v01'],'final_GT_PSNR':gtmeans[name]['PSNR'],'final_GT_LPIPS':gtmeans[name]['LPIPS_alex_v01'],'E09_immediate_score_difference':row['fit_loss_gain_0']-selected['current_scores'][b]});rows.append(row)
predictors=['fit_loss_gain_0','fit_loss_gain_20','fit_loss_gain_76','window_CTD_gain','mean_CTD_gain'];outcomes=['fit_loss_gain_380','holdout_loss_gain_380','GT_PSNR_gain_380','GT_LPIPS_decrease_380'];correlations=[]
for predictor in predictors:
 for outcome in outcomes:
  value=float(spearmanr([r[predictor] for r in rows],[r[outcome] for r in rows]).statistic);correlations.append({'predictor':predictor,'outcome':outcome,'spearman':value if np.isfinite(value) else None})
maxdiff=max(abs(r['E09_immediate_score_difference']) for r in rows);assert maxdiff<1e-6,'Immediate-state metric mismatch beyond declared numerical tolerance'
analysis=out/'analysis';analysis.mkdir()
for filename,data in [('candidate-outcomes.csv',rows),('correlations.csv',correlations)]:
 with (analysis/filename).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
report={'reference':{'gaussians':10128,'GT':gtmeans['reference'],'horizon_mean_losses':means['reference']},'candidate_count':24,'candidate_gaussians':10144,'new_gradient_updates':9500,'correlations':correlations,'max_E09_immediate_score_difference':maxdiff,'positive_immediate_but_negative_final_fit':sum(r['fit_loss_gain_0']>0 and r['fit_loss_gain_380']<0 for r in rows),'positive_immediate_but_negative_GT_PSNR':sum(r['fit_loss_gain_0']>0 and r['GT_PSNR_gain_380']<0 for r in rows),'outcome_ranges':{key:{'min':min(r[key] for r in rows),'max':max(r[key] for r in rows)} for key in outcomes},'verification':{'inputs_and_source_hashes':True,'shared1140_checkpoint_digest':True,'all25_geometry_recipes_reconstructed':True,'all25_finite_parameters_moments_and_Adam_step1520':True,'all100_horizon_records_and125_epochs_checked':True,'all500_GT_rows_averaged':True,'all25_final_model_hashes_frozen_before_GT':True,'same380_future_updates_each':True,'E09_immediate_score_parity_within1e_minus6':True},'scope':protocol['scope'],'limitations':protocol['limitations'],'interpretation':'Exploratory correlations, not a new selected policy or causal guarantee for joint128-parent selection. No p-values or significance claim.'};dump(analysis/'report.json',report)
with zipfile.ZipFile(analysis/'source.zip','x') as z:z.write(pathlib.Path(__file__).resolve(),'summarize_split_gain_prediction.py')
dump(root/'projects/dynamic-gaussian-ot-densification-research/experiments/e10-status-20261104.json',{'run':str(out),'report':report,'full4DGS_training_densification_implemented':False,'remote_publish_requested':False})
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(17,5.2),layout='constrained')
for ax,target,title in zip(axes[:2],['fit_loss_gain_380','GT_PSNR_gain_380'],['Fit loss gain after380 matched updates','GT PSNR gain after380 matched updates']):
 ax.scatter([r['fit_loss_gain_0'] for r in rows],[r[target] for r in rows]);ax.axhline(0,color='gray',lw=.7);ax.axvline(0,color='gray',lw=.7);ax.set_xlabel('Immediate fit loss gain');ax.set_ylabel(title);value=next(r['spearman'] for r in correlations if r['predictor']=='fit_loss_gain_0' and r['outcome']==target);ax.set_title(f'Spearman={value:.3f}' if value is not None else 'Constant data');ax.grid(alpha=.3)
matrix=np.array([[next(r['spearman'] for r in correlations if r['predictor']==pred and r['outcome']==outcome) for outcome in outcomes] for pred in predictors],dtype=float);axes[2].imshow(matrix,cmap='coolwarm',vmin=-1,vmax=1,aspect='auto');axes[2].set_xticks(range(4),['Final fit','Final heldout','GT PSNR','GT LPIPS'],rotation=25,ha='right');axes[2].set_yticks(range(5),['Immediate','After20','After76','CTD window','CTD mean']);axes[2].set_title('Exploratory Spearman correlations')
for i in range(5):
 for j in range(4):axes[2].text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center')
fig.suptitle('E10: all24 independent16-parent batches; same shared state and380-update reference; one scene/order');fig.savefig(analysis/'split-gain-prediction.png',dpi=140);plt.close(fig)
print(json.dumps(report,indent=2))
