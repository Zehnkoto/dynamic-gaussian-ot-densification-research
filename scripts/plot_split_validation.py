"""Plot E05 fixed-budget proposal comparison and verify the saved inspection video."""
import pathlib,json,sys,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image
inp=pathlib.Path(sys.argv[1]).resolve();root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();assert inp.is_relative_to(root);out=inp/'validation';analysis=out/'analysis';report=json.loads((analysis/'report.json').read_text());rows=report['rows'];labels=['Continue\nK10000','Window\nK10256','Global\nK10256','First-frame\nK10256','Random\nK10256'];colors=['#7b8586','#007ca8','#44958c','#af9a42','#cc7925'];fig,axes=plt.subplots(1,3,figsize=(12,3.8))
values=[[r['holdout_CTD_reduction_percent'] for r in rows],[r['PSNR_delta_vs_continue'] for r in rows],[100*(1-r['temporal_after']/rows[0]['temporal_after']) for r in rows]]
for ax,v,title in zip(axes,values,['Held-out CTD reduction (%)','GT PSNR gain vs continue (dB)','Temporal residual reduction (%)']):
 ax.bar(labels,v,color=colors);ax.set_title(title,fontsize=10);ax.tick_params(axis='x',labelsize=8);ax.set_ylim(0,max(v)*1.25)
 for i,value in enumerate(v):ax.text(i,value,f'{value:.3f}',ha='center',va='bottom',fontsize=8)
fig.suptitle('Offline split pilot | same extra 1520 appearance updates | single seed',fontsize=12);fig.text(.5,.01,'Four split arms each add 256 Gaussians. Windows rank fit76 shared partitions; not training-time adaptive densification.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.05,1,.94));p=analysis/'split-comparison.png';assert not p.exists();fig.savefig(p,dpi=160);plt.close(fig)
reader=imageio.get_reader(out/'fixed-camera-comparison.mp4');count=0
for frame in reader:
 if count==75:Image.fromarray(frame).save(analysis/'fixed-camera-middle.png')
 count+=1
reader.close();assert count==150;(analysis/'video-verification.json').write_text(json.dumps({'decoded_frames':count,'expected_frames':150},indent=2))
print('Plot saved; 150 video frames decoded')
