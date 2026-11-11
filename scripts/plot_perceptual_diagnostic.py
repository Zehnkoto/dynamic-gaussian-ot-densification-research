#!/usr/bin/env python3
"""E17 region summary figure from frozen diagnostic report."""
import argparse,json,os,pathlib,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_ctd_comparison import digest,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=pathlib.Path);a=ap.parse_args()
    root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();out=a.run.resolve();assert out.is_relative_to(root)
    r=json.loads((out/'report.json').read_text());audit=json.loads((out/'audit.json').read_text())
    labels=['GT edge','Other foreground','Background'];regions=['edge','other_foreground','background']
    fig,axes=plt.subplots(1,3,figsize=(13,4.3))
    colors=['#4976a0','#d28b37'];x=np.arange(3);width=.35
    for ax,metric,title in [(axes[0],'spatial_mean','Spatial LPIPS to GT'),(axes[1],'MSE','Pixel MSE to GT')]:
        for j,arm in enumerate(['mean','moment']):
            values=[r['regions'][arm][region][metric] for region in regions]
            ax.bar(x+(j-.5)*width,values,width,label='Mean covariance' if j==0 else 'Moment matching',color=colors[j])
        ax.set_xticks(x,labels);ax.set_title(title);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        if metric=='MSE':ax.set_yscale('log')
    values=[audit['spatial_LPIPS_delta_contributions'][region]['contribution_to_global_delta'] for region in regions]
    axes[2].bar(x,values,color='#bb5c53',width=.55)
    axes[2].set_xticks(x,labels);axes[2].set_title('Contribution to total LPIPS increase');axes[2].grid(axis='y',alpha=.15);axes[2].set_axisbelow(True)
    for i,v in enumerate(values):axes[2].text(i,v+.00007,f'{v/sum(values)*100:.1f}%',ha='center')
    axes[2].set_ylim(0,max(values)*1.25)
    axes[0].legend(frameon=False,loc='upper right')
    fig.suptitle('E17 | 20 paired GT views | fixed region masks | frozen seed 42 models')
    fig.text(.5,.01,'Spatial LPIPS map locates changes; its receptive field does not prove pixelwise causation.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.045,1,.92]);path=out/'region-summary.png';fig.savefig(path,dpi=170);plt.close(fig)
    repo=root/'projects/dynamic-gaussian-ot-densification-research'
    with zipfile.ZipFile(out/'plot-source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        z.write(repo/'scripts/plot_perceptual_diagnostic.py','scripts/plot_perceptual_diagnostic.py')
    dump(out/'plot-artifacts.json',{'plot_sha256':digest(path),'source_sha256':digest(out/'plot-source.zip'),'audit_sha256':digest(out/'audit.json')})
    print(path)
if __name__=='__main__':main()
