#!/usr/bin/env python3
"""Read-only paired target/loss/camera/resolution diagnosis of E10 frozen models."""
import argparse,json,os,pathlib,time,zipfile
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from appearance_model import Appearance
from training_split_core import clone_adam,split_adam,digest_state,RATES
from reduction_scene import load_scene,small_camera,render_state
from alignment_metrics import mse_decomposition
from run_ctd_comparison import dump,digest

@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args();root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();repo=root/'projects/dynamic-gaussian-ot-densification-research';out=a.output.resolve();assert out.is_relative_to(root) and not out.exists();protocol=json.loads((repo/'experiments/score-target-alignment-v1.json').read_text());e10=root/'outputs/ot/e10-gain-prediction-20261104T135817Z';e09=root/'outputs/ot/e09-current-state-reranking-20261104T133944Z';proposal=root/'outputs/ot/e05-split-20261104T103723Z'
 read=lambda p:json.loads(p.read_text());assert read(e10/'completion.json')['status']=='completed';em=read(e10/'manifest.json');design=read(e10/'design.json');freeze=read(e10/'training-complete.json');pm=read(proposal/'manifest.json');inputs={pathlib.Path(p):h for p,h in em['inputs'].items()}
 for name in ['manifest.json','design.json','measurements.json','validation/gt-per-view.json','training-complete.json','completion.json']:inputs[e10/name]=digest(e10/name)
 for name,h in freeze['checkpoint_hashes'].items():inputs[e10/'validation'/name]=h
 for p,h in inputs.items():assert p.is_relative_to(root) and digest(p)==h
 base=root/'outputs/dnerf/bouncingballs/baseline-numpy1235-20261030T174401Z';cfg,pc,scene,pipe=load_scene(base);assert cfg.white_background
 def tensor(x):return torch.tensor(np.array(x,copy=True),device='cuda')
 cache=pathlib.Path(pm['source_cache']);full=np.load(cache/'states.npy',mmap_mode='r');times=np.load(cache/'times.npy');sh=tensor(np.load(cache/'sh.npy'));opacity=tensor(np.load(cache/'opacity.npy'));geometry=np.load(pm['parent_states'],mmap_mode='r');children=np.load(proposal/'child-states.npy',mmap_mode='r');mass=np.load(proposal/'children.npz')['child_mass'];first=design['first_parents'];pool=design['candidate_batches'];geometry=np.concatenate([np.array(geometry,copy=True),children[:,np.array(first)*2+1]],1);geometry[:,first]=children[:,np.array(first)*2]
 shared=torch.load(e09/'before-second.pt',map_location='cuda');live=Appearance(shared['model']['source_sh'],shared['model']['source_logits'],10128);live.load_state_dict(shared['model']);opt=live.optimizer(RATES);opt.load_state_dict(shared['optimizer']);assert digest_state(live,opt)==shared['digest']==design['shared_digest']
 dataset=root/'datasets/bouncingballs';train_json=read(dataset/'transforms_train.json');test_json=read(dataset/'transforms_test.json');inputs[dataset/'transforms_train.json']=digest(dataset/'transforms_train.json');inputs[dataset/'transforms_test.json']=digest(dataset/'transforms_test.json');train=scene.getTrainCameras();test=scene.getTestCameras();train_by_time={}
 for j,r in enumerate(train_json['frames']):
  i=int(np.argmin(abs(times-r['time'])));assert abs(times[i]-r['time'])<1e-7;train_by_time[i]=j
 definitions={'fit400':[(i,train_by_time[i],'train',400) for i in design['monitor_views']['fit_decision']]}
 paired=[]
 for j,r in enumerate(test_json['frames']):
  i=int(np.argmin(abs(times-r['time'])));assert abs(times[i]-r['time'])<1e-7;paired.append((i,j))
 for split in ['train','test']:
  for size in [400,800]:definitions[f'{split}{size}']=[(i,train_by_time[i] if split=='train' else j,split,size) for i,j in paired]
 conditions={};metadata={};gt_parity=0.
 for name,entries in definitions.items():
  conditions[name]=[];metadata[name]=[]
  for i,j,split,size in entries:
   raw=(train if split=='train' else test)[j];assert abs(float(raw.time)-times[i])<1e-7;cam=small_camera(raw,size);gt=raw.original_image.cuda();assert tuple(gt.shape)==(3,800,800)
   frame=(train_json if split=='train' else test_json)['frames'][j];png=dataset/(frame['file_path']+'.png');inputs[png]=digest(png)
   if split=='test':
    saved=base/f'test/ours_20000/gt/{j:05d}.png';inputs[saved]=digest(saved);saved_gt=tensor(np.asarray(Image.open(saved).convert('RGB'),dtype=np.float32)/255).permute(2,0,1);error=float((gt-saved_gt).abs().max());gt_parity=max(gt_parity,error);assert error<1e-7;gt=saved_gt
   if size==400:gt=F.interpolate(gt.unsqueeze(0),size=(400,400),mode='area').squeeze(0)
   teacher=render_state(cam,tensor(full[i]),sh,opacity,cfg.sh_degree);conditions[name].append((i,j,cam,gt,teacher));metadata[name].append({'time_index':i,'time':float(times[i]),'frame_index':j,'camera_split':split,'size':size})
 out.mkdir(parents=True);dump(out/'manifest.json',{'protocol':protocol,'inputs':{str(p):h for p,h in inputs.items()},'conditions':metadata,'first_parents':first,'candidate_batches':pool,'shared_digest':shared['digest'],'GT_native_vs_saved_max_error':gt_parity,'GPU':os.environ.get('CUDA_VISIBLE_DEVICES')})
 with zipfile.ZipFile(out/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
  for name in ['run_score_target_alignment.py','alignment_metrics.py','test_alignment_metrics.py','appearance_model.py','training_split_core.py','reduction_scene.py','inspect_baseline.py','run_ctd_comparison.py','common.sh']:z.write(repo/'scripts'/name,'scripts/'+name)
  z.write(repo/'experiments/score-target-alignment-v1.json','experiments/score-target-alignment-v1.json')
 from utils.loss_utils import ssim
 def mix(p,t):return float(.8*(p-t).abs().mean()+.2*(1-ssim(p.unsqueeze(0),t.unsqueeze(0))))
 rows=[];started=time.monotonic()
 for phase in protocol['phases']:
  for batch in [-1]+list(range(24)):
   name='reference' if batch<0 else f'batch_{batch:02d}';geo=geometry
   if phase==0:
    model,unused=clone_adam(live,opt)
    if batch>=0:
     ids=np.array(pool[batch]);model,unused,error=split_adam(model,unused,ids,mass[ids*2]/(mass[ids*2]+mass[ids*2+1]))
   else:
    ck=torch.load(e10/'validation'/(name+'-appearance.pt'),map_location='cuda');state=ck['state_dict'];model=Appearance(state['source_sh'],state['source_logits'],len(state['dc']));model.load_state_dict(state);del ck
   if batch>=0:
    ids=np.array(pool[batch]);geo=np.concatenate([np.array(geometry,copy=True),children[:,ids*2+1]],1);geo[:,ids]=children[:,ids*2]
   colors,alpha=model(torch.arange(len(model.dc),device='cuda'))
   for condition,entries in conditions.items():
    for view,(i,j,cam,gt,teacher) in enumerate(entries):
     pred=render_state(cam,tensor(geo[i]),colors,alpha,cfg.sh_degree);metrics=mse_decomposition(pred,teacher,gt);q=(pred*255).round()/255;mse=float((q-gt).square().mean());metrics.update({'teacher_mix':mix(pred,teacher),'gt_mix':mix(pred,gt),'gt_mse_u8':mse,'gt_psnr_u8':float(-10*torch.log10((q-gt).square().mean()))});assert np.isfinite(list(metrics.values())).all();assert abs(metrics['gt_mse']-metrics['teacher_mse']-metrics['teacher_gt_mse']-metrics['cross_term'])<1e-7
     rows.append({'arm':name,'phase':phase,'condition':condition,'view':view,'time_index':i,**metrics})
   row={'phase':phase,'arm':name,'rows':len(rows)};print('MEASURE',json.dumps(row),flush=True)
   with (out/'progress.json').open('w') as f:json.dump(row,f)
   del model,geo
 assert len(rows)==5000 and digest_state(live,opt)==shared['digest'];dump(out/'measurements.json',rows)
 for p,h in inputs.items():assert digest(p)==h
 dump(out/'completion.json',{'status':'completed','training_updates':0,'input_hashes_unchanged':True,'shared_state_unchanged':True,'source_sha256':digest(out/'source.zip'),'rows':len(rows),'seconds':time.monotonic()-started});print('SCORE TARGET ALIGNMENT COMPLETE',flush=True)
if __name__=='__main__':main()
