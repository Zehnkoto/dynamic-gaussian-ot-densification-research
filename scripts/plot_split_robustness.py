"""Saved budget sweep and three-order-seed aggregates, with scope in the figure."""
import csv,json,pathlib,os,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=pathlib.Path(sys.argv[1]).resolve();root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();assert out.is_relative_to(root);analysis=out/'analysis';report=json.loads((analysis/'report.json').read_text())
with (analysis/'conditions.csv').open() as f:rows=list(csv.DictReader(f))
fig,axes=plt.subplots(1,3,figsize=(12,3.8));colors={'continue':'#7b8586','window':'#007ca8','global':'#44958c','first':'#af9a42','random':'#cc7925'}
for arm in ['window','global','first','random']:
 values=[next(float(r['PSNR_after']) for r in rows if int(r['splits'])==b and int(r['seed'])==42 and r['arm']==arm) for b in [128,256,512]];axes[0].plot([128,256,512],values,'o-',color=colors[arm],label=arm)
base=next(float(r['PSNR_after']) for r in rows if int(r['seed'])==42 and r['arm']=='continue');axes[0].axhline(base,color=colors['continue'],linestyle='--',label='unsplit');axes[0].set_xticks([128,256,512]);axes[0].set_xlabel('Added Gaussians, seed42 only');axes[0].set_ylabel('GT PSNR (dB)');axes[0].legend(fontsize=7);axes[0].set_title('Budget sensitivity')
groups=report['budget256_order_seed_summary'];names=[g['arm'] for g in groups]
for ax,key,title,scale in zip(axes[1:],['PSNR_after','temporal_after'],['Extra256: GT PSNR','Extra256: temporal residual'],[1,1000]):
 means=[g['metrics'][key]['mean']*scale for g in groups];std=[g['metrics'][key]['sample_std']*scale for g in groups];ax.bar(names,means,yerr=std,capsize=3,color=[colors[n] for n in names]);ax.tick_params(axis='x',labelsize=7);ax.set_title(title);ax.set_ylim(0,max(means)*1.15)
 for i,v in enumerate(means):ax.text(i,v,f'{v:.3f}',ha='center',va='bottom',fontsize=7)
axes[2].set_ylabel('RGB delta residual x1000');fig.suptitle('Offline CTD split robustness | fixed geometry and base model',fontsize=12);fig.text(.5,.01,'Extra256: mean +/- sample SD over 3 view-order/random-selection seeds. Other budgets: seed42 only. Same1520updates.',ha='center',fontsize=8);fig.tight_layout(rect=(0,.05,1,.94));target=analysis/'robustness-summary.png';assert not target.exists();fig.savefig(target,dpi=160);plt.close(fig)
