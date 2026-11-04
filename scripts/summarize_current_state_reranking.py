#!/usr/bin/env python3
"""Audit fit-only batch reranking, common states, final models and eval means."""
import argparse,csv,json,os,pathlib,zipfile
import numpy as np
import torch
from appearance_model import Appearance
from training_split_core import digest_state,RATES
from run_ctd_comparison import dump,digest
p=argparse.ArgumentParser();p.add_argument('output',type=pathlib.Path);out=p.parse_args().output.resolve();root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();assert out.is_relative_to(root)
read=lambda p:json.loads(p.read_text());m=read(out/'manifest.json');protocol=m['protocol'];done=read(out/'completion.json');assert done['status']=='completed' and done['source_sha256']==digest(out/'source.zip')
for path,h in m['inputs'].items():assert digest(pathlib.Path(path))==h
source=root/'outputs/ot/e06-training-event-20261104T113535Z';sm=read(source/'manifest.json');assert sm['orders']==m['orders'] and sm['decision_views']==m['decision_views'];assert set(m['fit']).isdisjoint(m['holdout']) and set(m['decision_views'])<=set(m['fit'])
selection=read(out/'selection.json');ledger=read(out/'scoring.json');events=read(out/'event-decisions.json');training=read(out/'training.json');summary=read(out/'validation/summary.json');gt=read(out/'validation/gt-per-view.json')
proposal=root/'outputs/ot/e05-split-20261104T103723Z';pm=read(proposal/'manifest.json');child=np.load(proposal/'children.npz');gains=child['gains'];times=np.load(pathlib.Path(pm['source_cache'])/'times.npy');bins=np.minimum((times*5).astype(int),4);scores=np.stack([gains[[i for i in m['fit'] if bins[i]==b]].mean(0) for b in range(5)]).min(0);eligible=child['eligible'];ranked=eligible[np.lexsort((np.arange(len(eligible)),-scores[eligible]))[:512]];first=ranked[:128].tolist();pool=ranked[128:].reshape(24,16);assert first==selection['first_parents'] and pool.tolist()==selection['candidate_batches'] and len(set(ranked))==512
for filename,step in [('after-first.pt',760),('before-second.pt',1140)]:
 ck=torch.load(out/filename,map_location='cpu');model=Appearance(ck['model']['source_sh'],ck['model']['source_logits'],10128);model.load_state_dict(ck['model']);opt=model.optimizer(RATES);opt.load_state_dict(ck['optimizer']);assert digest_state(model,opt)==ck['digest'] and ck['main_steps']==step and ck['selected_parents']==first
 assert selection['cost']['stale' if step==760 else 'current']['live_digest_before_after']==ck['digest']
 for v in ck['model'].values():assert torch.isfinite(v).all()
 for st in ck['optimizer']['state'].values():assert float(st['step'])==step and torch.isfinite(st['exp_avg']).all() and torch.isfinite(st['exp_avg_sq']).all()
 if step==1140:assert ck['digest']==selection['shared_second_digest']
assert selection['scoring_live_unchanged'] and len(ledger)==48
for step,key in [(760,'stale'),(1140,'current')]:
 rows=[r for r in ledger if r['step']==step];assert len(rows)==24 and [r['batch'] for r in rows]==list(range(24));calculated=[]
 for b,r in enumerate(rows):
  assert r['parents']==pool[b].tolist() and len(r['reference'])==len(r['proposal'])==20 and r['reference']==rows[0]['reference']
  a=np.array(r['reference']);c=np.array(r['proposal']);assert np.isfinite(a).all() and np.isfinite(c).all();gain=float((a-c).mean());assert abs(gain-r['mean_render_gain'])<1e-12 and abs(gains[m['fit']][:,pool[b]].sum(1).mean()-r['fit_CTD_gain'])<1e-12;calculated.append(gain)
 assert np.array_equal(calculated,selection[key+'_scores'])
 cost=selection['cost'][key];assert cost['main_step']==step and cost['seconds']>=0 and cost['reference_renders']==20 and cost['proposal_renders']==480
expected={'fixed_ctd':list(range(8)),'current_render':np.lexsort((np.arange(24),-np.array(selection['current_scores'])))[:8].tolist(),'stale_render':np.lexsort((np.arange(24),-np.array(selection['stale_scores'])))[:8].tolist(),'random':np.random.default_rng(np.random.SeedSequence([42,909])).choice(24,8,replace=False).tolist()};assert expected==selection['chosen_rank_order']
geometry=np.load(pm['parent_states'],mmap_mode='r');children=np.load(proposal/'child-states.npy',mmap_mode='r');rows=[];sets={}
common=[r for r in training if r['arm']=='shared_middle'];assert len(common)==5
for e,t in enumerate(common):assert t['main_updates']==760+(e+1)*76 and t['gaussians']==10128
for name in protocol['arms']:
 ids=first+pool[sorted(expected[name])].reshape(-1).tolist();assert selection['chosen_commit_order'][name]==sorted(expected[name]) and len(ids)==len(set(ids))==256;sets[name]=set(ids[128:]);ev=[r for r in events if r['arm']==name];assert len(ev)==8
 for b,r in zip(sorted(expected[name]),ev):assert r['batch']==b and r['step']==1140 and r['parents']==pool[b].tolist() and r['after_count']-r['before_count']==16 and r['optical_depth_error']<1e-5
 assert ev[0]['before_count']==10128 and ev[-1]['after_count']==10256
 ck=torch.load(out/'validation'/(name+'-appearance.pt'),map_location='cpu');assert ck['accepted_parents']==ids and ck['start_digest']==selection['shared_second_digest'] and ck['main_steps']==1520 and len(ck['state_dict']['dc'])==10256
 for v in ck['state_dict'].values():assert torch.isfinite(v).all()
 for st in ck['optimizer']['state'].values():assert float(st['step'])==1520 and torch.isfinite(st['exp_avg']).all() and torch.isfinite(st['exp_avg_sq']).all()
 expected_geo=np.concatenate([np.array(geometry,copy=True),children[:,np.array(ids)*2+1]],1);expected_geo[:,ids]=children[:,np.array(ids)*2];assert np.array_equal(expected_geo,np.load(out/'validation'/(name+'-states.npy'),mmap_mode='r'));del expected_geo
 post=[r for r in training if r['arm']==name];assert len(post)==5
 for e,t in enumerate(post):assert t['main_updates']==1140+(e+1)*76 and t['gaussians']==10256
 integral=760*10000+sum(76*t['gaussians'] for t in common+post);assert integral==done['main_gaussian_updates_each_split_arm']==15345920
 rows.append({'arm':name,'gaussians':10256,'second_selected_batches':len(expected[name]),'main_gaussian_updates':integral,**summary['official_20_view_gt'][name+'_after'],'holdout_PSNR_to_full':summary['fidelity_to_full_400'][name+'_after']['holdout']['PSNR_to_full'],'temporal_residual':summary['fixed_camera_temporal_delta_residual_256'][name+'_after'],'new_parents_vs_fixed_ctd':len(sets[name]-sets['fixed_ctd'])})
for name in summary['active_gaussian_counts']:
 for phase in ['before','after']:
  views=[r for r in gt if r['arm']==name and r['phase']==phase];assert sorted(r['test_index'] for r in views)==list(range(20))
  for metric in ['PSNR','SSIM','LPIPS_alex_v01']:assert abs(np.mean([r[metric] for r in views])-summary['official_20_view_gt'][name+'_'+phase][metric])<1e-12
old=read(source/'validation/summary.json')
for metric,v in summary['official_20_view_gt']['continue_after'].items():assert abs(v-old['official_20_view_gt']['continue_after'][metric])<1e-5
previous=root/'outputs/ot/e08-two-event-20261104T123454Z';old_ck=torch.load(previous/'validation/twice_forced-appearance.pt',map_location='cpu');new_ck=torch.load(out/'validation/fixed_ctd-appearance.pt',map_location='cpu');assert old_ck['accepted_parents']==new_ck['accepted_parents'];old_metrics=read(previous/'validation/summary.json')['official_20_view_gt']['twice_forced_after'];repeat_delta={k:rows[0][k]-v for k,v in old_metrics.items()}
from scipy.stats import spearmanr
correlation=float(spearmanr(selection['current_scores'],selection['stale_scores']).statistic);by={r['arm']:r for r in rows};pairs=[{'arm':n,'minus_fixed_ctd_PSNR':by[n]['PSNR']-by['fixed_ctd']['PSNR'],'minus_stale_render_PSNR':by[n]['PSNR']-by['stale_render']['PSNR']} for n in protocol['arms']]
analysis=out/'analysis';analysis.mkdir()
with (analysis/'comparison.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
report={'rows':rows,'pairs':pairs,'scoring_cost':selection['cost'],'current_vs_stale_score_spearman':correlation,'current_stale_selected_parent_overlap':len(sets['current_render']&sets['stale_render']),'fixed_ctd_E08_repeat_delta':repeat_delta,'verification':{'input_and_source_hashes':True,'shared760_and1140_digests':True,'pool_reconstructed_fit_only':True,'all48_candidate_batch_scores_recomputed':True,'all4_rankings_reconstructed':True,'shared_main_orders_and_exact10256_finalcounts':True,'equal_main_gaussian_update_integral':True,'accepted_geometry':True,'finite_Adam_step1520':True,'all20GT_means':True,'continue_metric_parity':True,'fixed_ctd_same_parents_as_E08':True},'limitations':protocol['limitations']};dump(analysis/'report.json',report)
with zipfile.ZipFile(analysis/'source.zip','x') as z:z.write(pathlib.Path(__file__).resolve(),'summarize_current_state_reranking.py')
# Public report omits state digests, timings retained for honest scoring-cost reporting.
public=json.loads(json.dumps(report))
for cost in public['scoring_cost'].values():cost.pop('live_digest_before_after')
dump(root/'projects/dynamic-gaussian-ot-densification-research/experiments/e09-status-20261104.json',{'run':str(out),'report':public,'full4DGS_training_densification_implemented':False})
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained');labels=['Fixed CTD','Current render','Stale render','Random']
for ax,key,title in zip(axes[:2],['PSNR','temporal_residual'],['20-view PSNR (dB), higher is better','Temporal RGB delta residual, lower is better']):
 values=[r[key] for r in rows];ax.plot(range(4),values,'o');ax.set_xticks(range(4),labels,rotation=20);ax.set_title(title);ax.grid(axis='y',alpha=.3)
axes[2].scatter(selection['stale_scores'],selection['current_scores']);axes[2].set_xlabel('Fit gain at step760');axes[2].set_ylabel('Fit gain at step1140');axes[2].set_title(f'24 batch scores; Spearman={correlation:.3f}');axes[2].grid(alpha=.3)
fig.suptitle('E09: second-event reranking; same shared state, final K=10256 and main-update schedule');fig.savefig(analysis/'current-state-reranking.png',dpi=140);plt.close(fig)
print(json.dumps(public,indent=2))
