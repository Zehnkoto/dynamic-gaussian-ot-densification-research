#!/usr/bin/env python3
"""E17 independent consistency audit and fixed-mask LPIPS contribution accounting."""
import argparse,json,os,pathlib,zipfile
import numpy as np
from run_ctd_comparison import digest,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=pathlib.Path);a=ap.parse_args()
    root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();out=a.run.resolve();assert out.is_relative_to(root)
    read=lambda n:json.loads((out/n).read_text())
    man=read('manifest.json');report=read('report.json');rows=read('per-view.json');paired=read('paired-per-view.json')
    assert read('completion.json')['status']=='completed'
    assert digest(out/'source.zip')==read('completion.json')['source_sha256']
    for path,h in man['inputs'].items():assert digest(pathlib.Path(path))==h,path
    for path,h in man['source_E16_inputs'].items():assert digest(pathlib.Path(path))==h,path
    assert len(rows)==60 and len(paired)==20
    indexed={(r['view'],r['arm']):r for r in rows}
    assert len(indexed)==60 and all((j,n) in indexed for j in range(20) for n in ['teacher','mean','moment'])
    max_spatial_scalar_error=max(abs(r['spatial_map_mean']-r['LPIPS']) for r in rows)
    assert max_spatial_scalar_error<1e-5
    e16=root/'outputs/ot/e16-covariance-moment-diagnostic-20261110-v1'
    orig=json.loads((e16/'gt-per-view.json').read_text())
    original={(r['view'],r['arm']):r for r in orig}
    max_metric_error=0.
    for j in range(20):
        for n in ['mean','moment']:
            r=indexed[j,n];p=original[j,f'{n}_covariance__seed42']
            for col,base in [('PSNR','PSNR'),('SSIM','SSIM'),('LPIPS','LPIPS_alex_v01')]:
                max_metric_error=max(max_metric_error,abs(r[col]-p[base]))
        a=indexed[j,'mean'];b=indexed[j,'moment'];q=paired[j];assert q['view']==j
        for col,target in [('PSNR','PSNR_delta'),('LPIPS','LPIPS_delta')]:
            assert abs(q[target]-(b[col]-a[col]))<1e-10
        assert sum(a[k+'_n'] for k in ['edge','other_foreground','background'])==800*800
        assert all(a[k+'_n']==b[k+'_n']==indexed[j,'teacher'][k+'_n'] for k in ['edge','other_foreground','background'])
    assert max_metric_error<2e-4
    assert sum(q['LPIPS_delta']>0 for q in paired)==20 and sum(q['PSNR_delta']>0 for q in paired)==19
    assert abs(np.mean([q['PSNR_delta'] for q in paired])-report['paired']['mean_PSNR_delta'])<1e-10
    assert abs(np.mean([q['LPIPS_delta'] for q in paired])-report['paired']['mean_LPIPS_delta'])<1e-10
    regions=['edge','other_foreground','background'];npx=20*800*800;summary={}
    for label in regions:
        count=sum(indexed[j,'mean'][label+'_n'] for j in range(20))
        a=sum(indexed[j,'mean'][label+'_spatial_mean']*indexed[j,'mean'][label+'_n'] for j in range(20))/count
        b=sum(indexed[j,'moment'][label+'_spatial_mean']*indexed[j,'moment'][label+'_n'] for j in range(20))/count
        assert abs(a-report['regions']['mean'][label]['spatial_mean'])<1e-8
        assert abs(b-report['regions']['moment'][label]['spatial_mean'])<1e-8
        summary[label]={'pixels':count,'pixel_fraction':count/npx,'mean_spatial_LPIPS_delta':b-a,'contribution_to_global_delta':(b-a)*count/npx,
                        'worse_views':sum(q[label+'_spatial_delta']>0 for q in paired),
                        'pixel_MSE_better_views':sum(q[label+'_MSE_delta']<0 for q in paired)}
    assert sum(v['pixels'] for v in summary.values())==npx
    global_delta=sum(v['contribution_to_global_delta'] for v in summary.values())
    assert abs(global_delta-report['paired']['mean_LPIPS_delta'])<1e-5
    base=np.load(root/'outputs/ot/e02-ctd-refined-20261103T082856Z/budget-10000/shared-states.npy',mmap_mode='r')
    scatter=np.load(e16/'center-scatter.npy',mmap_mode='r')
    fraction=[];quantile=[]
    for t in range(150):
        trace=np.asarray(base[t,:,3]+base[t,:,7]+base[t,:,11],dtype=np.float64)
        extra=np.asarray(scatter[t,:,0,0]+scatter[t,:,1,1]+scatter[t,:,2,2],dtype=np.float64)
        ratio=extra/np.maximum(trace,1e-12)
        fraction.append(float(np.mean(ratio>1)))
        quantile.append(float(np.quantile(ratio,.9)))
    edge_grad_mean=report['regions']['mean']['edge']['gradient_mean']
    edge_grad_moment=report['regions']['moment']['edge']['gradient_mean']
    audit={'verification':{'all_E16_and_E17_inputs_hash_matched':True,'all20_views_and_masks_recomputed':True,'E16_metric_max_difference':max_metric_error,
                           'spatial_map_mean_vs_scalar_LPIPS_max_difference':max_spatial_scalar_error,'region_contributions_match_global_delta':True},
           'spatial_LPIPS_delta_contributions':summary,'global_spatial_LPIPS_delta':global_delta,
           'edge_gradient_moment_vs_mean_ratio':edge_grad_moment/edge_grad_mean,
           'edge_MSE_moment_vs_mean_ratio':report['regions']['moment']['edge']['MSE']/report['regions']['mean']['edge']['MSE'],
           'source_covariance_expansion':{'fraction_trace_more_than_doubled_time_mean':float(np.mean(fraction)),
                                          'fraction_trace_more_than_doubled_range':[float(min(fraction)),float(max(fraction))],
                                          'ninetieth_percentile_extra_over_base_trace_time_mean':float(np.mean(quantile))},
           'interpretation':'Worse LPIPS spans edge, non-edge foreground and background. Lower edge gradient and larger covariance are consistent with smoothing; map is observational, not a causal attribution of every pixel.'}
    dump(out/'audit.json',audit)
    repo=root/'projects/dynamic-gaussian-ot-densification-research'
    with zipfile.ZipFile(out/'analysis-source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        z.write(repo/'scripts/summarize_perceptual_diagnostic.py','scripts/summarize_perceptual_diagnostic.py')
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
