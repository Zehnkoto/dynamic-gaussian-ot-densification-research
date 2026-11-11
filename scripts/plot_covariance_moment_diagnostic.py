#!/usr/bin/env python3
"""Plot every predeclared E16 arm; no post-hoc selection."""
import argparse,json,os,pathlib,csv,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_ctd_comparison import digest,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=pathlib.Path);a=ap.parse_args()
    root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();out=a.run.resolve();assert out.is_relative_to(root)
    audit=json.loads((out/'audit.json').read_text());report=json.loads((out/'report.json').read_text());dest=out/'paired-diagnostic-v2.png';assert not dest.exists()
    conditions=['mean_covariance','moment_covariance'];colors=['#4976a0','#d28b37'];fig,axes=plt.subplots(1,3,figsize=(12,3.7))
    for ax,metric,title in zip(axes,['PSNR','LPIPS_alex_v01','temporal_residual256'],['GT PSNR (higher is better)','GT LPIPS (lower is better)','Temporal residual (lower is better)']):
        means=[audit['aggregate'][c][metric]['mean'] for c in conditions];sd=[audit['aggregate'][c][metric]['sample_sd'] for c in conditions]
        ax.bar([0,1],means,yerr=sd,color=colors,width=.6,capsize=4,alpha=.85)
        for seed in [42,43,44]:
            vals=[next(r[metric] for r in report['rows'] if r['arm']==f'{c}__seed{seed}') for c in conditions]
            ax.plot([0,1],vals,'k.-',alpha=.45,lw=.7,ms=4)
        ax.set_xticks([0,1],['Mean covariance','Moment matching']);ax.set_title(title,fontsize=11);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        if metric=='PSNR':ax.set_ylim(36.8,38.8)
        for i,y in enumerate(means):ax.text(i,y+(.08 if metric=='PSNR' else max(means)*.025),f'{y:.4f}' if metric!='temporal_residual256' else f'{y:.6f}',ha='center',fontsize=10)
        if metric!='PSNR':ax.set_ylim(0,max(means)*1.20)
    fig.suptitle('E16: covariance-only diagnostic | fixed 10,000 groups | paired training orders 42/43/44',fontsize=12)
    fig.text(.5,.01,'PSNR and temporal residual improve, but LPIPS worsens. One scene; no overall superiority claim.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.06,1,.92]);fig.savefig(dest,dpi=170);plt.close(fig)
    with (out/'metrics-v2.csv').open('x',newline='') as f:
        fields=['arm','PSNR','SSIM','LPIPS_alex_v01','temporal_residual256'];writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(report['rows'])
    repo=root/'projects/dynamic-gaussian-ot-densification-research'
    with zipfile.ZipFile(out/'analysis-source-v2.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ['summarize_covariance_moment_diagnostic.py','plot_covariance_moment_diagnostic.py']:
            z.write(repo/'scripts'/name,'scripts/'+name)
    dump(out/'analysis-artifacts-v2.json',{'plot_sha256':digest(dest),'analysis_source_sha256':digest(out/'analysis-source-v2.zip'),'audit_sha256':digest(out/'audit.json'),'report_sha256':digest(out/'report.json')})
    print(dest)
if __name__=='__main__':main()
