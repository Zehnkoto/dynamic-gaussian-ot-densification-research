"""Evaluate every E13 branch with shared position corrections."""
import json
import numpy as np
import torch
from PIL import Image
from reduction_scene import render_state,small_camera
from run_ctd_comparison import dump
from position_ctd_core import apply_translation

@torch.no_grad()
def evaluate(out,base,protocol,times,fit,held,full,sh,opacity,cfg,scene,cameras,models,geometries,teachers,real400,tensor,offsets):
 from utils.loss_utils import ssim
 from utils.image_utils import psnr
 def pred(name,i,cam):
  model=models[name];colors,alpha=model(torch.arange(len(model.dc),device='cuda'));return render_state(cam,apply_translation(tensor(geometries[name][i]),offsets[name]),colors,alpha,cfg.sh_degree)
 def mix(p,t):return float(.8*(p-t).abs().mean()+.2*(1-ssim(p.unsqueeze(0),t.unsqueeze(0))))
 fidelity=[]
 for i in range(150):
  row={'time_index':i,'split':'fit' if i in fit else 'holdout','arms':{}}
  for name in models:
   p=pred(name,i,cameras[i]);values={}
   for target in ['native400','area800']:
    t=teachers[target][i];values[target+'_mse']=float((p-t).square().mean());values[target+'_mix']=mix(p,t)
   values['GT400_mse']=float((p-real400[i]).square().mean());row['arms'][name]=values
  fidelity.append(row)
 dump(out/'fidelity-per-view.json',fidelity);print('FIDELITY COMPLETE',flush=True)
 import lpips
 criterion=lpips.LPIPS(net='alex',version='0.1').cuda().eval();gtrows=[];test=json.loads((base.parents[3]/'datasets/bouncingballs/transforms_test.json').read_text())['frames']
 for j,r in enumerate(test):
  i=int(np.argmin(abs(times-r['time'])));assert abs(times[i]-r['time'])<1e-7;cam=scene.getTestCameras()[j];gt=tensor(np.asarray(Image.open(base/f'test/ours_20000/gt/{j:05d}.png').convert('RGB'),dtype=np.float32)/255).permute(2,0,1).unsqueeze(0)
  for name in models:
   p=((pred(name,i,cam)*255).round()/255).unsqueeze(0);gtrows.append({'arm':name,'test_index':j,'PSNR':float(psnr(p,gt)),'SSIM':float(ssim(p,gt)),'LPIPS_alex_v01':float(criterion(p,gt,normalize=True))})
 dump(out/'gt-per-view.json',gtrows);del criterion;print('GT COMPLETE',flush=True)
 fixed=small_camera(scene.getTestCameras()[10],256);previous=None;temporal=[]
 for i in range(150):
  images={'teacher':render_state(fixed,tensor(full[i]),sh,opacity,cfg.sh_degree)}
  for name in models:images[name]=pred(name,i,fixed)
  if previous is not None:temporal.append({'time_index':i,**{name:float(((images[name]-previous[name])-(images['teacher']-previous['teacher'])).abs().mean()) for name in models}})
  previous=images
 dump(out/'temporal-per-pair.json',temporal)
 summary={}
 for name in models:
  views=[r for r in gtrows if r['arm']==name];summary[name]={'gaussians':len(models[name].dc),'GT':{k:float(np.mean([r[k] for r in views])) for k in ['PSNR','SSIM','LPIPS_alex_v01']},'temporal_residual256':float(np.mean([r[name] for r in temporal])),'fidelity':{}}
  for split in ['fit','holdout']:
   values=[r['arms'][name] for r in fidelity if r['split']==split];summary[name]['fidelity'][split]={k:float(np.mean([r[k] for r in values])) for k in values[0]}
 dump(out/'summary.json',summary);print('EVALUATION COMPLETE',flush=True)
