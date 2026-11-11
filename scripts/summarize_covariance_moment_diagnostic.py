#!/usr/bin/env python3
"""Independent E16 artifact audit and compact paired report; never select weights."""
import argparse,json,os,pathlib
import numpy as np
import torch
from run_ctd_comparison import digest,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=pathlib.Path);a=ap.parse_args()
    root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();out=a.run.resolve();assert out.is_relative_to(root)
    read=lambda n:json.loads((out/n).read_text())
    manifest=read('manifest.json');freeze=read('training-complete.json');completion=read('completion.json');report=read('report.json')
    assert completion['status']=='completed'
    for p,h in manifest['inputs'].items():assert digest(pathlib.Path(p))==h,p
    assert digest(out/'source.zip')==completion['source_sha256']
    assert len(freeze['checkpoint_hashes'])==6 and freeze['gradient_updates']==9120
    assert (out/'training-complete.json').stat().st_mtime_ns<(out/'gt-per-view.json').stat().st_mtime_ns
    fit=manifest['fit'];held=manifest['holdout'];assert len(fit)==76 and len(held)==74 and set(fit)|set(held)==set(range(150))
    orders=manifest['orders']
    for seed in [42,43,44]:
        assert len(orders[str(seed)])==20 and all(sorted(o)==sorted(fit) for o in orders[str(seed)])
    assert len({tuple(o) for v in orders.values() for o in v})==60
    gt=read('gt-per-view.json');fid=read('fidelity-per-view.json');temp=read('temporal-per-pair.json');training=read('training.json')
    assert len(gt)==160 and len(fid)==1200 and len(temp)==1192 and len(training)==120
    byname={r['arm']:r for r in report['rows']};assert len(byname)==8
    for name,r in byname.items():
        vals=[v for v in gt if v['arm']==name];assert sorted(v['view'] for v in vals)==list(range(20))
        for metric in ['PSNR','SSIM','LPIPS_alex_v01']:
            assert abs(r[metric]-np.mean([v[metric] for v in vals]))<1e-12
        vals=[v for v in temp if v['arm']==name];assert sorted(v['time_index'] for v in vals)==list(range(1,150))
        assert abs(r['temporal_residual256']-np.mean([v['residual'] for v in vals]))<1e-12
        for split in ['fit','holdout']:
            vals=[v for v in fid if v['arm']==name and v['split']==split];assert len(vals)==(76 if split=='fit' else 74)
            for metric in ['PSNR_to_teacher','L1_to_teacher','mixed_loss_to_teacher']:
                assert abs(r[split][metric]-np.mean([v[metric] for v in vals]))<1e-12
    cache=root/'outputs/ot/e02-ctd-20261103T082303Z/trajectory-cache';prior=root/'outputs/ot/e02-ctd-refined-20261103T082856Z/budget-10000'
    source_sh=torch.from_numpy(np.load(cache/'sh.npy'));source_logits=None
    for name,h in freeze['checkpoint_hashes'].items():
        assert digest(out/name)==h
        ck=torch.load(out/name,map_location='cpu');assert ck['steps']==1520
        assert all(int(s['step'])==1520 for s in ck['optimizer']['state'].values())
        assert all(torch.isfinite(v).all() for v in ck['state_dict'].values())
        assert torch.equal(ck['state_dict']['source_sh'],source_sh)
        if source_logits is None:source_logits=ck['state_dict']['source_logits']
        else:assert torch.equal(source_logits,ck['state_dict']['source_logits'])
        rows=[r for r in training if r['arm']==name.removesuffix('-appearance.pt')]
        assert [r['epoch'] for r in rows]==list(range(1,21)) and rows[-1]['updates']==1520
    # Rebuild scatter through raw E[xx^T]-E[x]E[x]^T, independently of centered algorithm.
    full=np.load(cache/'states.npy',mmap_mode='r');labels=np.load(prior/'shared-grouping.npz')['labels'];w=np.load(cache/'opacity.npy').reshape(-1).astype(np.float64);w/=w.sum()
    mass=np.bincount(labels,weights=w,minlength=10000);stored=np.load(out/'center-scatter.npy',mmap_mode='r');maxerr=0.;max_rounding_ratio=0.
    for t in range(150):
        xyz=full[t,:,:3].astype(np.float64)
        mu=np.stack([np.bincount(labels,weights=w*xyz[:,j],minlength=10000)/mass for j in range(3)],1)
        for j in range(3):
            for k in range(3):
                raw=np.bincount(labels,weights=w*xyz[:,j]*xyz[:,k],minlength=10000)/mass
                product=mu[:,j]*mu[:,k];direct=raw-product
                error=abs(direct-stored[t,:,j,k]);maxerr=max(maxerr,float(error.max()))
                # Float32 cache permits half an ULP; raw moments add float64 cancellation.
                allowance=.51*abs(np.spacing(stored[t,:,j,k]))+1e-12*(abs(raw)+abs(product)+1)
                max_rounding_ratio=max(max_rounding_ratio,float(np.max(error/allowance)))
    assert max_rounding_ratio<=1
    paired=[]
    metrics=['PSNR','SSIM','LPIPS_alex_v01','temporal_residual256']
    for seed in [42,43,44]:
        old=byname[f'mean_covariance__seed{seed}'];new=byname[f'moment_covariance__seed{seed}']
        paired.append({'seed':seed,**{m:new[m]-old[m] for m in metrics}})
    aggregate={}
    for c in ['mean_covariance','moment_covariance']:
        rows=[byname[f'{c}__seed{s}'] for s in [42,43,44]]
        aggregate[c]={m:{'mean':float(np.mean([r[m] for r in rows])),'sample_sd':float(np.std([r[m] for r in rows],ddof=1))} for m in metrics}
    audit={'verification':{'all_source_and_checkpoint_hashes':True,'all6_frozen_before_evaluation':True,'all160_GT1200_fidelity1192_temporal_rows':True,'all9120_updates_and_Adam1520':True,'paired_orders_source_appearance_identical':True,'independent_raw_second_moment_scatter_reconstruction':True},'max_independent_scatter_error':maxerr,'max_error_to_float32_rounding_bound':max_rounding_ratio,'aggregate':aggregate,'paired':paired,'E04_reproduction_PSNR_delta':report['E04_reproduction_PSNR_delta'],'scope':manifest['protocol']['scope']}
    dump(out/'audit.json',audit)
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
