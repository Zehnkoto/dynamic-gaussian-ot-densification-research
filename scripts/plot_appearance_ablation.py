"""Plot saved E04 ablation aggregates, including three order-seed sample SDs."""
import pathlib,json,sys,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=pathlib.Path(sys.argv[1]).resolve();root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();assert out.is_relative_to(root)
r=json.loads((out/'analysis/report.json').read_text());g={(v['mode'],v['arm']):v['metrics'] for v in r['groups']};modes=['sh','opacity','joint'];x=np.arange(3);fig,axes=plt.subplots(1,3,figsize=(11,3.5))
for ax,key,title,scale in zip(axes,['PSNR','LPIPS_alex_v01','temporal_residual'],['GT PSNR (higher better)','GT LPIPS (lower better)','Temporal residual (lower better)'],[1,1,1000]):
 for arm,offset,color,label in [('shared',-.18,'#007ca8','Shared geometry'),('perframe',.18,'#cc7925','Per-frame geometry')]:
  means=[g[(m,arm)][key]['mean']*scale for m in modes];std=[g[(m,arm)][key]['sample_std']*scale for m in modes]
  ax.bar(x+offset,means,width=.35,yerr=std,capsize=3,label=label,color=color)
  for at,v in zip(x+offset,means):ax.text(at,v,f'{v:.3f}',ha='center',va='bottom',fontsize=7)
 ax.set_xticks(x,['SH only','Opacity only','Both']);ax.set_title(title,fontsize=10);ax.set_ylim(0,max(g[(m,a)][key]['mean']*scale for m in modes for a in ['shared','perframe'])*1.2)
axes[0].legend(fontsize=8);axes[2].set_ylabel('RGB delta residual x 1000');fig.suptitle('K=10000 | mean +/- sample SD over 3 training-order seeds',fontsize=12);fig.text(.5,.005,'Geometry fixed. Equal parameter counts across geometry arms within each mode; 480k / 10k / 490k across modes.',ha='center',fontsize=8);fig.tight_layout(rect=(0,.04,1,.94))
p=out/'analysis/ablation-summary.png';assert not p.exists();fig.savefig(p,dpi=170);plt.close(fig)
