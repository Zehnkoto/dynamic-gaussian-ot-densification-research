"""Scientific event-count, cost/quality and equal-count followup review artifacts."""
import pathlib,json,sys,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image
out=pathlib.Path(sys.argv[1]).resolve();matched=pathlib.Path(sys.argv[2]).resolve();root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();assert out.is_relative_to(root) and matched.is_relative_to(root)
report=json.loads((out/'analysis/report.json').read_text());events=json.loads((out/'event-decisions.json').read_text());follow=json.loads((matched/'analysis/report.json').read_text());colors={'continue':'#7b8586','forced_window':'#44958c','gated_window':'#007ca8','forced_random':'#d6a36e','gated_random':'#cc7925'}
fig,axes=plt.subplots(1,3,figsize=(12,3.8))
for arm in ['forced_window','gated_window','gated_random']:
 e=[r for r in events if r['arm']==arm];axes[0].step(range(17),[10000]+[r['after_count'] for r in e],where='post',label=('both forced arms' if arm=='forced_window' else arm),color=colors[arm],linestyle='--' if arm.startswith('forced') else '-')
axes[0].set_xlabel('Proposal batches at live step760');axes[0].set_ylabel('Active Gaussians');axes[0].set_title('Accepted capacity change');axes[0].legend(fontsize=6)
for i,r in enumerate(report['rows']):
 axes[1].scatter(r['gaussians'],r['PSNR'],color=colors[r['arm']],marker='^' if r['arm'].startswith('forced') else 'o');axes[1].annotate(r['arm'],(r['gaussians'],r['PSNR']),xytext={'continue':(0,10),'forced_window':(0,10),'gated_window':(-90,-16),'forced_random':(10,0),'gated_random':(-50,-18)}[r['arm']],textcoords='offset points',fontsize=7)
axes[1].set_xlim(9970,10410);axes[1].set_ylim(37.65,38.15);axes[1].set_xlabel('Final active Gaussians');axes[1].set_ylabel('GT PSNR (dB)');axes[1].set_title('Counts differ: no equal-budget claim')
x=np.arange(2);labels=['Window candidates','Random candidates']
for offset,key,label,color in [(-.18,'gated_PSNR','Render gate','#007ca8'),(.18,'prefix_PSNR','Fixed prefix','#cc7925')]:
 v=[r[key] for r in follow['rows']];axes[2].bar(x+offset,v,width=.35,label=label,color=color)
 for i,value in enumerate(v):axes[2].text(i+offset,value,f'{value:.3f}',ha='center',va='bottom',fontsize=7)
axes[2].set_xticks(x,labels,fontsize=8);axes[2].set_ylim(0,43);axes[2].set_title('Matched finalcount followup');axes[2].legend(fontsize=7,loc='lower left')
fig.suptitle('One split event inside reduced-model appearance training | single scene/seed',fontsize=12);fig.text(.5,.01,'All live models:760warm+760post updates. Probes discarded. Fixed teacher/child geometry; not full4DGS adaptive densification.',ha='center',fontsize=8);fig.tight_layout(rect=(0,.05,1,.94));target=out/'analysis/training-event-summary-v2.png';assert not target.exists();fig.savefig(target,dpi=160);plt.close(fig)
for p in [out,matched]:
 if (p/'analysis/video-verification.json').exists():continue
 reader=imageio.get_reader(p/'validation/fixed-camera-comparison.mp4');count=0
 for frame in reader:
  if count==75:Image.fromarray(frame).save(p/'analysis/fixed-camera-middle.png')
  count+=1
 reader.close();assert count==150;(p/'analysis/video-verification.json').write_text(json.dumps({'decoded_frames':count,'expected_frames':150},indent=2))
print('Event figure saved and both150frame videos decoded')
