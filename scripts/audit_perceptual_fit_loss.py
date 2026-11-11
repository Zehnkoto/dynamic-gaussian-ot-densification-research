#!/usr/bin/env python3
"""Independent bookkeeping and result audit for the fixed E18 beta comparison."""
import argparse, hashlib, json, math, pathlib
import numpy as np
import torch


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def check_close(a,b,tol=1e-8):
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tol,(a,b)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=pathlib.Path);args=ap.parse_args()
    out=args.output.resolve();assert out.is_dir()
    manifest=json.loads((out/'manifest.json').read_text());protocol=manifest['protocol']
    assert protocol['perceptual_beta']==0.1 and protocol['order_seeds']==[42,43,44]
    assert len(manifest['fit'])==76 and len(manifest['holdout'])==74
    assert set(manifest['fit']).isdisjoint(manifest['holdout'])
    assert set(manifest['fit'])|set(manifest['holdout'])==set(range(150))
    for p,h in manifest['inputs'].items():assert digest(pathlib.Path(p))==h,p
    for seed,orders in manifest['orders'].items():
        rng=np.random.default_rng(int(seed));assert len(orders)==20
        assert all(o==rng.permutation(manifest['fit']).tolist() for o in orders)
    frozen=json.loads((out/'training-complete.json').read_text())
    assert frozen['all6_frozen_before_evaluation'] and frozen['gradient_updates']==9120
    assert len(frozen['checkpoint_hashes'])==6
    for n,h in frozen['checkpoint_hashes'].items():
        p=out/n;assert digest(p)==h
        ckpt=torch.load(p,map_location='cpu')
        assert ckpt['steps']==1520 and ckpt['seed'] in [42,43,44]
        assert ckpt['condition'] in protocol['conditions']
        assert all(int(s['step'])==1520 for s in ckpt['optimizer']['state'].values())
        assert all(torch.isfinite(v).all() for v in ckpt['state_dict'].values())
    training=json.loads((out/'training.json').read_text());assert len(training)==120
    names={f'{condition}__seed{seed}' for seed in [42,43,44] for condition in protocol['conditions']}
    assert {r['arm'] for r in training}==names
    assert all([r['updates'] for r in training if r['arm']==n]==list(range(76,1521,76)) for n in names)
    gt=json.loads((out/'gt-per-view.json').read_text())
    fidelity=json.loads((out/'fidelity-per-view.json').read_text())
    temporal=json.loads((out/'temporal-per-pair.json').read_text())
    allnames=names|{'mean_covariance__before','moment_covariance__before'}
    assert len(gt)==160 and len(fidelity)==1200 and len(temporal)==1192
    for n in allnames:
        assert {r['view'] for r in gt if r['arm']==n}==set(range(20))
        assert {r['time_index'] for r in fidelity if r['arm']==n}==set(range(150))
        assert {r['time_index'] for r in temporal if r['arm']==n}==set(range(1,150))
        assert sum(r['split']=='fit' for r in fidelity if r['arm']==n)==76
    report=json.loads((out/'report.json').read_text());rows={r['arm']:r for r in report['rows']}
    assert set(rows)==allnames
    for n,row in rows.items():
        for metric in ['PSNR','SSIM','LPIPS_alex_v01']:
            check_close(row[metric],float(np.mean([r[metric] for r in gt if r['arm']==n])))
        check_close(row['temporal_residual256'],float(np.mean([r['residual'] for r in temporal if r['arm']==n])))
        for split in ['fit','holdout']:
            records=[r for r in fidelity if r['arm']==n and r['split']==split]
            for metric in ['PSNR_to_teacher','L1_to_teacher','mixed_loss_to_teacher']:
                check_close(row[split][metric],float(np.mean([r[metric] for r in records])))
    assert len(report['paired_beta_minus_E16'])==6
    controls={r['arm']:r for r in report['control_rows']}
    for pair in report['paired_beta_minus_E16']:
        n=f"{pair['condition']}__seed{pair['seed']}"
        for metric in ['PSNR','SSIM','LPIPS_alex_v01','temporal_residual256']:
            check_close(pair[metric],rows[n][metric]-controls[n][metric])
        check_close(pair['holdout_mixed_loss_delta'],rows[n]['holdout']['mixed_loss_to_teacher']-controls[n]['holdout']['mixed_loss_to_teacher'])
    for condition,mean in report['paired_beta_mean'].items():
        pairs=[r for r in report['paired_beta_minus_E16'] if r['condition']==condition]
        assert len(pairs)==3
        for metric,value in mean.items():check_close(value,float(np.mean([r[metric] for r in pairs])))
    completion=json.loads((out/'completion.json').read_text())
    assert completion['status']=='completed' and completion['inputs_unchanged'] and completion['checkpoint_hashes_verified']
    assert completion['source_sha256']==digest(out/'source.zip')
    result={'status':'passed','input_hashes':len(manifest['inputs']),'checkpoints':6,'fit_epoch_rows':120,'gt_rows':160,'fidelity_rows':1200,'temporal_rows':1192,'paired_rows':6,'paired_beta_mean':report['paired_beta_mean']}
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
