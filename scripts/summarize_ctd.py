#!/usr/bin/env python3
"""Validate complete E02 outputs and export a comparison CSV/figure/report."""
import argparse,csv,hashlib,json,os,pathlib
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=pathlib.Path,required=True);args=ap.parse_args()
    root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();run=args.run.resolve()
    if not run.is_relative_to(root):raise ValueError('Unsafe run path')
    done=json.loads((run/'completion.json').read_text());assert done['status']=='completed'
    assert hashlib.sha256((run/'source.zip').read_bytes()).hexdigest()==done['source_snapshot_sha256']
    cache=pathlib.Path(done['trajectory_cache']);source_states=np.load(cache/'states.npy',mmap_mode='r');source_times=np.load(cache/'times.npy');weights=np.load(cache/'opacity.npy').astype(np.float64).ravel();weights/=weights.sum()
    summaries=[];table=[];verification={}
    for budget in done['budgets']:
        folder=run/f'budget-{budget}';s=json.loads((folder/'summary.json').read_text());time_rows=json.loads((folder/'time-results.json').read_text())
        assert s['budget']==budget and len(time_rows)==150
        assert np.array_equal(np.array([x['time'] for x in time_rows]),source_times)
        assert sum(x['split']=='fit' for x in time_rows)==76 and sum(x['split']=='shared-fit-holdout' for x in time_rows)==74
        grouping=np.load(folder/'shared-grouping.npz');labels=grouping['labels'];assert len(np.unique(labels))==budget
        mass=np.bincount(labels,weights=weights,minlength=budget);assert np.allclose(mass,grouping['normalized_mass'],rtol=1e-10,atol=1e-12)
        assert np.isclose(mass.sum(),1) and (mass>0).all()
        per_labels=np.load(folder/'perframe-labels.npy',mmap_mode='r');reps=np.load(folder/'perframe-representatives.npy',mmap_mode='r')
        assert per_labels.shape==(150,len(weights)) and reps.shape==(150,budget)
        assert reps.min()>=0 and reps.max()<len(weights)
        for row in per_labels:assert len(np.unique(row))==budget
        for method in ['shared','perframe']:
            states=np.load(folder/(method+'-states.npy'),mmap_mode='r');assert states.shape==(150,budget,12) and np.isfinite(states).all()
            for ti in [0,75,149]:
                lab=labels if method=='shared' else np.asarray(per_labels[ti])
                m=np.bincount(lab,weights=weights,minlength=budget);expected=np.zeros((budget,12),dtype=np.float64)
                np.add.at(expected,lab,np.asarray(source_states[ti],dtype=np.float64)*weights[:,None]);expected/=m[:,None]
                assert np.allclose(expected,states[ti],rtol=3e-6,atol=1e-7)
        perview=json.loads((folder/'gt-per-view.json').read_text())
        for method in ['shared','perframe']:
            assert len(perview[method])==20
            for key in ['PSNR','SSIM','LPIPS_alex_v01']:assert np.isclose(np.mean([x[key] for x in perview[method]]),s['official_20_view_gt'][method][key],atol=1e-12)
            row={'budget':budget,'method':method,**s['official_20_view_gt'][method],
                 'temporal_delta_residual':s['fixed_camera_temporal_delta_residual'][method],
                 'fit_CTD':s['fit_and_holdout']['fit'][method+'_ctd'],'holdout_CTD':s['fit_and_holdout']['shared-fit-holdout'][method+'_ctd'],
                 'fitting_seconds':s['shared_fit']['seconds'] if method=='shared' else s['perframe_fitting_150_times_seconds'],
                 'converged_fits':int(s['shared_fit']['converged']) if method=='shared' else s['perframe_converged'],
                 'total_fits':1 if method=='shared' else 150}
            table.append(row)
        delta=s['fixed_camera_temporal_delta_residual'];s['temporal_residual_reduction_percent_shared_vs_oracle']=100*(1-delta['shared']/delta['perframe'])
        verification[str(budget)]={'exact_counts_all_150_times':True,'shared_mass_conserved':True,'finite_states':True,'barycenters_recomputed_3_times_both_methods':True,'time_order_matches':True,'GT_means_match_per_view':True,'shared_converged':s['shared_fit']['converged'],'perframe_converged':s['perframe_converged'],'perframe_total':150}
        summaries.append(s)
    destination=run/'analysis';destination.mkdir(exist_ok=False)
    with (destination/'comparison.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for method,label,color in [('perframe','Per-frame CTD oracle','#2767aa'),('shared','Shared CTD grouping','#d36b22')]:
        rows=sorted([x for x in table if x['method']==method],key=lambda x:x['budget']);x=[r['budget'] for r in rows]
        for ax,key in zip(axes,['PSNR','LPIPS_alex_v01','temporal_delta_residual']):ax.plot(x,[r[key] for r in rows],'-o',label=label,color=color)
    base=json.loads((root/'outputs/dnerf/bouncingballs/baseline-numpy1235-20261030T174401Z/baseline-validation.json').read_text())
    axes[0].axhline(base['PSNR'],ls='--',color='gray',label='Full 4DGS baseline')
    for ax,label in zip(axes,['Test PSNR (dB, higher better)','Test LPIPS-Alex (lower better)','Temporal delta residual (lower better)']):
        ax.set_xlabel('Gaussian budget');ax.set_ylabel(label);ax.set_xticks(sorted(done['budgets']));ax.grid(alpha=.25)
    axes[0].legend(fontsize=8);axes[2].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    fig.suptitle('CTD adaptation: single seed, no appearance fine-tuning')
    fig.savefig(destination/'budget-tradeoff.png',dpi=220);fig.savefig(destination/'budget-tradeoff.svg');plt.close(fig)
    report={'run':str(run),'source_snapshot_sha256':done['source_snapshot_sha256'],'verification':verification,'budgets':summaries,
      'limitations':['Single seed and scene; exploratory results, not statistical evidence.','GHAP CTD core adaptation with global tiled assignment and exact-K repair, not the official full GHAP pipeline.','Per-frame oracle refits at every evaluation time; shared grouping is fitted at 76 times.','Temporal metric compares RGB time differences against full baseline; it can trade off with image fidelity.','Timing is reduction fitting only on a shared GPU, not standalone compressed-model inference speed.','No appearance fine-tuning, velocity term or adaptive split.']}
    with (destination/'report.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({'analysis':str(destination),'verification':verification,'temporal_improvement_percent':{s['budget']:s['temporal_residual_reduction_percent_shared_vs_oracle'] for s in summaries}},indent=2))
if __name__=='__main__':main()
