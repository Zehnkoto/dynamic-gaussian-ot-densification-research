"""Evaluate final fixed-step offline split proposals without selecting on evaluation."""
import json
import numpy as np
import torch
from PIL import Image,ImageDraw
import imageio.v2 as imageio
from reduction_scene import render_state,small_camera
from run_ctd_comparison import dump

@torch.no_grad()
def evaluate(out,base,inp,protocol,times,fit,full,sh,opacity,cfg,scene,cameras,models,render,tensor):
 from utils.loss_utils import ssim
 from utils.image_utils import psnr
 fidelity=[]
 for i in range(150):
  target=render_state(cameras[i],tensor(full[i]),sh,opacity,cfg.sh_degree);row={'time':float(times[i]),'split':'fit' if i in fit else 'holdout'}
  for name in models:
   for phase in ['before','after']:
    pred=render(name,i,cameras[i],phase=='after');row[name+'_'+phase]={'PSNR_to_full':float(psnr(pred.unsqueeze(0),target.unsqueeze(0))),'L1_to_full':float((pred-target).abs().mean())}
  fidelity.append(row)
 dump(out/'fidelity-per-view.json',fidelity);print('FIDELITY EVALUATED',flush=True)
 import lpips
 criterion=lpips.LPIPS(net='alex',version='0.1').cuda().eval();gtrows=[]
 test=json.loads((base.parents[3]/'datasets/bouncingballs/transforms_test.json').read_text())['frames']
 for name in models:(out/(name+'-test-after')).mkdir()
 for j,r in enumerate(test):
  i=int(np.argmin(abs(times-r['time'])));assert abs(times[i]-r['time'])<1e-7
  cam=scene.getTestCameras()[j];gt=tensor(np.asarray(Image.open(base/f'test/ours_20000/gt/{j:05d}.png').convert('RGB'),dtype=np.float32)/255).permute(2,0,1).unsqueeze(0)
  for name in models:
   for phase in ['before','after']:
    pred=(render(name,i,cam,phase=='after')*255).round()/255;p=pred.unsqueeze(0)
    gtrows.append({'arm':name,'phase':phase,'test_index':j,'PSNR':float(psnr(p,gt)),'SSIM':float(ssim(p,gt)),'LPIPS_alex_v01':float(criterion(p,gt,normalize=True))})
    if phase=='after':Image.fromarray((pred.permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)).save(out/(name+'-test-after')/f'{j:05d}.png')
 dump(out/'gt-per-view.json',gtrows);del criterion;print('GT EVALUATED',flush=True)
 fixed=small_camera(scene.getTestCameras()[10],256);previous=None;residual={name+'_'+p:[] for name in models for p in ['before','after']};temporal=[]
 with imageio.get_writer(out/'fixed-camera-comparison.mp4',fps=15,codec='libx264',quality=8,macro_block_size=1) as video:
  for i,t in enumerate(times):
   imgs={'full':render_state(fixed,tensor(full[i]),sh,opacity,cfg.sh_degree)}
   for name in models:
    for p in ['before','after']:imgs[name+'_'+p]=render(name,i,fixed,p=='after')
   if previous is not None:
    row={'time':float(t)}
    for key in residual:
     v=float(((imgs[key]-previous[key])-(imgs['full']-previous['full'])).abs().mean());residual[key].append(v);row[key]=v
    temporal.append(row)
   panel=Image.new('RGB',(1536,296),'white');draw=ImageDraw.Draw(panel);keys=['full']+[name+'_after' for name in models]
   for j,key in enumerate(keys):
    draw.text((j*256+4,4),key,fill='black');draw.text((j*256+4,20),f't={t:.3f}',fill='black');panel.paste(Image.fromarray((imgs[key].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)),(j*256,40))
   video.append_data(np.asarray(panel));previous=imgs
 dump(out/'temporal-per-pair.json',temporal)
 summary={'protocol':protocol,'official_20_view_gt':{},'fidelity_to_full_400':{},'fixed_camera_temporal_delta_residual_256':{key:float(np.mean(v)) for key,v in residual.items()},('candidate_pool' if protocol.get('event_step') else 'split_proposals'):json.loads((inp/'proposal-summary.json').read_text()),'active_gaussian_counts':{name:len(model.dc) for name,model in models.items()}}
 for name in models:
  for phase in ['before','after']:
   key=name+'_'+phase;selected=[r for r in gtrows if r['arm']==name and r['phase']==phase]
   summary['official_20_view_gt'][key]={metric:float(np.mean([r[metric] for r in selected])) for metric in ['PSNR','SSIM','LPIPS_alex_v01']}
   summary['fidelity_to_full_400'][key]={s:{metric:float(np.mean([r[key][metric] for r in fidelity if r['split']==s])) for metric in ['PSNR_to_full','L1_to_full']} for s in ['fit','holdout']}
 e04=base.parents[2]/'ot/e04-appearance-20261103T095355Z/summary.json';reference=json.loads(e04.read_text())['official_20_view_gt']['shared_after']
 for metric,v in summary['official_20_view_gt']['continue_before'].items():assert abs(v-reference[metric])<1e-5,(metric,v,reference[metric])
 dump(out/'summary.json',summary);print('RESULT',json.dumps(summary),flush=True)
