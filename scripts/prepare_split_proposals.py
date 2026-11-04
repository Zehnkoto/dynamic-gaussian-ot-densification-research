#!/usr/bin/env python3
"""Fit local child identities only on76times and materialize equal-budget proposals."""
import argparse,json,os,pathlib,zipfile
import numpy as np
from scipy.stats import spearmanr
from split_proposal_core import bipartitions,child_trajectories,selections
from run_ctd_comparison import digest,dump
ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path,required=True);ap.add_argument('--protocol',default='split-proposal-v1.json');ap.add_argument('--splits',type=int);ap.add_argument('--seed',type=int);args=ap.parse_args();root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();out=args.output.resolve();repo=root/'projects/dynamic-gaussian-ot-densification-research'
if not out.is_relative_to(root) or out.exists():raise ValueError('Unsafe or existing output')
protocol=json.loads((repo/'experiments'/args.protocol).read_text());cache=root/'outputs/ot/e02-ctd-20261103T082303Z/trajectory-cache';prior=root/'outputs/ot/e02-ctd-refined-20261103T082856Z/budget-10000';appearance=root/'outputs/ot/e04-appearance-20261103T095355Z'
if args.splits is not None:protocol['splits']=args.splits;protocol['final_budget']=protocol['base_budget']+args.splits
if args.seed is not None:protocol['seed']=args.seed
assert 0<protocol['splits']<=protocol['base_budget']
meta=json.loads((cache/'manifest.json').read_text());inputs={cache/n:r['sha256'] for n,r in meta['files'].items()}
base=root/'outputs/dnerf/bouncingballs/baseline-numpy1235-20261030T174401Z'
for name,record in meta['baseline_model_hashes'].items():inputs[base/name]=record['sha256']
for name in ['shared-grouping.npz','shared-states.npy']:inputs[prior/name]=digest(prior/name)
inputs[appearance/'shared-appearance.pt']=digest(appearance/'shared-appearance.pt')
for p,h in inputs.items():assert digest(p)==h
states=np.load(cache/'states.npy',mmap_mode='r');times=np.load(cache/'times.npy');weights=np.load(cache/'opacity.npy').astype(np.float64).reshape(-1);weights/=weights.sum();group=np.load(prior/'shared-grouping.npz');parents=group['labels'];parent_states=np.load(prior/'shared-states.npy',mmap_mode='r')
split=json.loads((repo/'experiments/reduction-ctd-v2.json').read_text());fit=[int(np.argmin(abs(times-r['time']))) for r in split['fit']];held=[int(np.argmin(abs(times-r['time']))) for r in split['reduction_holdout']];assert len(fit)==76 and len(held)==74 and not set(fit)&set(held)
out.mkdir(parents=True);dump(out/'manifest.json',{'protocol':protocol,'inputs':{str(p):h for p,h in inputs.items()},'fit':fit,'holdout':held,'parent_states':str(prior/'shared-states.npy'),'appearance_checkpoint':str(appearance/'shared-appearance.pt'),'parent_representatives':group['representatives'].tolist(),'source_cache':str(cache)})
print('Fitting local two-way trajectory partitions',flush=True)
child,eligible,iterations=bipartitions(states,parents,weights,fit);centers,mass,gains=child_trajectories(states,parents,child,weights)
assert np.isfinite(centers).all() and np.isclose(mass.sum(),1)
np.savez(out/'children.npz',source_child=child,child_mass=mass,eligible=eligible,gains=gains);np.save(out/'child-states.npy',centers)
scores,selected,windows=selections(gains,eligible,fit,times,protocol['splits'],protocol['seed'])
base_ctd=[]
for t in range(150):base_ctd.append(float((weights[:,None]*(states[t].astype(np.float64)-parent_states[t,parents])**2).sum()))
base_ctd=np.array(base_ctd);summary={'eligible_parents':len(eligible),'max_local_iterations':max(iterations),'candidate_iterations_at_cap':sum(i==30 for i in iterations),'base_fit_CTD':float(base_ctd[fit].mean()),'base_holdout_CTD':float(base_ctd[held].mean()),'methods':{}}
for method,ids in selected.items():
 assert len(np.unique(ids))==protocol['splits'] and np.all(mass[ids*2]>0) and np.all(mass[ids*2+1]>0)
 geometry=np.concatenate([np.array(parent_states,copy=True),centers[:,ids*2+1]],axis=1);geometry[:,ids]=centers[:,ids*2]
 assert geometry.shape==(150,protocol['final_budget'],12)
 eigen=np.linalg.eigvalsh(geometry[:,:,3:].astype(np.float64).reshape(-1,3,3))
 # Source covariance is float32 R diag(scale^2) R^T; tiny negative eigenvalues
 # already occur upstream. Reject beyond two float32 eps relative to max eigenvalue.
 tolerance=2*np.finfo(np.float32).eps*np.maximum(eigen[:,-1],1e-30)+1e-12
 assert np.all(eigen[:,0]>=-tolerance), 'Covariance indefinite beyond float32 rounding'
 np.save(out/(method+'-states.npy'),geometry)
 decrease=gains[:,ids].sum(1)
 assert np.all(base_ctd-decrease>=-1e-8)
 summary['methods'][method]={'covariance_min_eigenvalue':float(eigen[:,0].min()),'covariance_min_relative_eigenvalue':float((eigen[:,0]/np.maximum(eigen[:,-1],1e-30)).min()),'selected':ids.tolist(),'fit_CTD_gain':float(decrease[fit].mean()),'holdout_CTD_gain':float(decrease[held].mean()),'holdout_CTD_reduction_percent':float(100*decrease[held].mean()/base_ctd[held].mean())}
 if method in scores:summary['methods'][method]['score_vs_holdout_gain_spearman']=float(spearmanr(scores[method][eligible],gains[held].mean(0)[eligible]).statistic)
 print('PROPOSAL',method,'holdout CTD reduction',summary['methods'][method]['holdout_CTD_reduction_percent'],flush=True)
np.savez(out/'scores.npz',**scores,window_means=windows)
dump(out/'proposal-summary.json',summary)
with zipfile.ZipFile(out/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
 for name in ['split_proposal_core.py','test_split_proposal_core.py','prepare_split_proposals.py','run_split_validation.py','evaluate_split_validation.py','appearance_model.py','reduction_scene.py','run_ctd_comparison.py','reduction_core.py','inspect_baseline.py','common.sh']:z.write(repo/'scripts'/name,'scripts/'+name)
 z.write(repo/'experiments'/args.protocol,'experiments/'+args.protocol)
dump(out/'completion.json',{'status':'completed','source_sha256':digest(out/'source.zip'),'artifacts':{p.name:digest(p) for p in list(out.glob('*.npy'))+list(out.glob('*.npz'))+[out/'proposal-summary.json']}});print('PROPOSALS COMPLETE',flush=True)
