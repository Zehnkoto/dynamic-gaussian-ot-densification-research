#!/usr/bin/env python3
"""E20: fit-only five-time-window reassignment at fixed E19 support size."""
import argparse,hashlib,json,math,os,pathlib,shutil,sys,time,zipfile
import numpy as np
import torch
from PIL import Image
from appearance_model import Appearance
from reduction_core import nearest,barycenters
from reduction_scene import state,small_camera,render_state
from inspect_baseline import configuration

SCENE='jumpingjacks'
E19='e19-second-scene-jumpingjacks-20261110-v1'
BASE='baseline-e19-jumpingjacks-20261110-v1'

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def dump(p,obj):
 with p.open('x') as f:json.dump(obj,f,indent=2);f.write('\n')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path,required=True);args=ap.parse_args()
 root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();repo=root/'projects/dynamic-gaussian-ot-densification-research';e19=root/'outputs/ot'/E19;base=root/'outputs/dnerf'/SCENE/BASE;data=root/'datasets'/SCENE;out=args.output.resolve()
 assert out.is_relative_to(root/'outputs/ot') and not out.exists() and shutil.disk_usage(root).free>5*1024**3
 protocol_path=repo/'experiments/windowed-reassignment-v1.json';protocol=json.loads(protocol_path.read_text())
 prior=json.loads((e19/'manifest.json').read_text());control_report=json.loads((e19/'report.json').read_text());complete=json.loads((e19/'completion.json').read_text());control_ckpt=e19/'mean_covariance__beta0-appearance.pt'
 assert complete['status']=='completed' and json.loads((e19/'audit.json').read_text())['status']=='passed'
 n=prior['n'];k=prior['k'];fit=prior['fit_indices'];held=prior['holdout_indices'];orders=prior['orders'];cap=round(protocol['max_switch_fraction_per_window']*n)
 assert n==24985 and k==9276 and len(fit)==101 and len(held)==99 and cap==500
 assert fit==list(range(0,200,2))+[199] and held==list(range(1,199,2))
 assert protocol['window_size']==5 and protocol['window_count']==40 and len(orders)==20
 cfg=configuration(base/'cfg_args');assert pathlib.Path(cfg.source_path).resolve()==data and pathlib.Path(cfg.model_path).resolve()==base
 assert cfg.no_do and cfg.no_dshs
 model_files=sorted((base/'point_cloud/iteration_20000').glob('*'))
 assert {p.name for p in model_files}=={'point_cloud.ply','deformation.pth','deformation_table.pth','deformation_accum.pth'}
 files=[protocol_path,e19/'manifest.json',e19/'report.json',e19/'audit.json',e19/'completion.json',e19/'training-complete.json',control_ckpt,e19/'teacher-states.npy',e19/'mean-states.npy',e19/'shared-grouping.npz',data/'transforms_train.json',data/'transforms_test.json',base/'cfg_args',*model_files]
 files.extend(base/f'test/ours_20000/gt/{j:05d}.png' for j in range(20))
 inputs={p:digest(p) for p in files}
 assert digest(control_ckpt)==json.loads((e19/'training-complete.json').read_text())['checkpoint_hashes'][control_ckpt.name]
 full=np.load(e19/'teacher-states.npy',mmap_mode='r');fixed_states=np.load(e19/'mean-states.npy',mmap_mode='r');group=np.load(e19/'shared-grouping.npz');fixed_labels=np.array(group['labels'],dtype=np.int64);ids_np=np.array(group['representatives'],dtype=np.int64)
 assert full.shape==(200,n,12) and fixed_states.shape==(200,k,12) and fixed_labels.shape==(n,)
 sys.path.insert(0,str(root/'projects/4DGaussians'))
 from arguments import ModelParams,ModelHiddenParams,PipelineParams
 from scene import Scene,GaussianModel
 ap2=argparse.ArgumentParser();mp=ModelParams(ap2);hp=ModelHiddenParams(ap2);pp=PipelineParams(ap2)
 pc=GaussianModel(cfg.sh_degree,hp.extract(cfg));scene=Scene(mp.extract(cfg),pc,load_iteration=20000,shuffle=False);pipe=pp.extract(cfg)
 assert len(pc.get_xyz)==n
 for v in vars(pc).values():
  if isinstance(v,torch.nn.Parameter):v.requires_grad_(False)
  elif isinstance(v,torch.nn.Module):v.requires_grad_(False);v.eval()
 def tensor(x):return torch.tensor(np.array(x,copy=True),device='cuda')
 sh=pc.get_features.detach();opacity=pc.get_opacity.detach();logits=pc._opacity.detach();weights=opacity.flatten().double();weights/=weights.sum();w=weights.cpu().numpy();ids=tensor(ids_np).long()
 assert abs(float(w.sum())-1)<1e-12
 initial=Appearance(sh,logits,k);saved=torch.load(control_ckpt,map_location='cpu');assert saved['steps']==2020 and saved['condition']=='mean_covariance' and saved['beta']==0.
 initial.load_state_dict(saved['state_dict']);initial.eval()
 assert torch.equal(initial.source_sh,sh) and torch.equal(initial.source_logits,logits)
 for p in initial.parameters():p.requires_grad_(False)
 out.mkdir(parents=True);start=time.monotonic()
 dump(out/'manifest.json',{'protocol':protocol,'E19_control':str(control_ckpt),'n':n,'k':k,'cap_per_window':cap,'fit_indices':fit,'holdout_indices':held,'orders':orders,'inputs':{str(p):h for p,h in inputs.items()},'GPU':os.environ.get('CUDA_VISIBLE_DEVICES')})
 with zipfile.ZipFile(out/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
  for name in ['run_windowed_reassignment.py','appearance_model.py','reduction_core.py','reduction_scene.py','inspect_baseline.py','common.sh']:z.write(repo/'scripts'/name,'scripts/'+name)
  z.write(protocol_path,'experiments/'+protocol_path.name)
 arms=['window_full_geometry','window_position_only'];geometries={'fixed_E19_mean_beta0':fixed_states};switch_logs={};label_paths={};distortion=[]
 for arm in arms:
  metric='full' if arm=='window_full_geometry' else 'position'
  previous=fixed_labels.copy();all_labels=np.lib.format.open_memmap(out/(arm+'-labels.npy'),mode='w+',dtype=np.int32,shape=(200,n));events=[]
  for window in range(40):
   block=list(range(5*window,5*window+5));seen=[i for i in block if i in fit];assert len(seen) in [2,3]
   dimension=12 if metric=='full' else 3
   x=np.transpose(full[seen,:,:dimension],(1,0,2)).copy().reshape(n,-1)
   c=np.transpose(fixed_states[seen,:,:dimension],(1,0,2)).copy().reshape(k,-1)
   xt=tensor(x).double()/math.sqrt(len(seen));ct=tensor(c).double()/math.sqrt(len(seen))
   target,best=nearest(xt,ct)
   current=((xt-ct[tensor(previous).long()])**2).sum(1)
   gain=(current-best).clamp_min(0).cpu().numpy();proposed=target.cpu().numpy()
   score=w*gain;ranking=np.argsort(-score,kind='stable');counts=np.bincount(previous,minlength=k);updated=previous.copy();accepted=0;raw_positive=int(np.count_nonzero((gain>1e-12)&(proposed!=previous)))
   for idx in ranking:
    if accepted>=cap or score[idx]<=1e-12:break
    old=updated[idx];new=int(proposed[idx])
    if new==old or counts[old]<=1:continue
    updated[idx]=new;counts[old]-=1;counts[new]+=1;accepted+=1
   assert counts.min()>=1 and accepted<=cap
   all_labels[block]=updated.astype(np.int32)
   events.append({'window':window,'fit_indices':seen,'switches':accepted,'positive_candidates':raw_positive,'disagreement_with_fixed':int(np.count_nonzero(updated!=fixed_labels)),'minimum_group_size':int(counts.min()),'weighted_fixed_center_gain':float(np.sum(w*(current.cpu().numpy()-((xt-ct[tensor(updated).long()])**2).sum(1).cpu().numpy())))})
   previous=updated
   del xt,ct,target,best,current
   if window%10==0:print('ASSIGN',arm,window,'switches',accepted,flush=True)
  all_labels.flush();del all_labels
  labels=np.load(out/(arm+'-labels.npy'),mmap_mode='r');states=np.lib.format.open_memmap(out/(arm+'-states.npy'),mode='w+',dtype=np.float32,shape=(200,k,12))
  with torch.no_grad():
   for i in range(200):
    centers,_=barycenters(tensor(full[i]).double(),tensor(labels[i]).long(),weights,k)
    states[i]=centers.float().cpu().numpy()
    if i%50==0:print('BARYCENTERS',arm,i,flush=True)
  states.flush();del states
  geometries[arm]=np.load(out/(arm+'-states.npy'),mmap_mode='r');switch_logs[arm]=events;label_paths[arm]=labels
 dump(out/'switch-events.json',switch_logs)
 # The full 12D weighted proxy is evaluated on every time for all three arms.
 with torch.no_grad():
  for i in range(200):
   source=tensor(full[i]).double()
   for arm in ['fixed_E19_mean_beta0',*arms]:
    labels=tensor(fixed_labels if arm=='fixed_E19_mean_beta0' else label_paths[arm][i]).long();centers=tensor(geometries[arm][i]).double()
    delta=source-centers[labels]
    distortion.append({'arm':arm,'time_index':i,'split':'fit' if i in fit else 'holdout','full12_weighted_cost':float((weights*(delta*delta).sum(1)).sum()),'position_weighted_cost':float((weights*(delta[:,:3]**2).sum(1)).sum())})
 dump(out/'distribution-per-time.json',distortion)
 frames=json.loads((data/'transforms_train.json').read_text())['frames'];times=np.array(sorted(r['time'] for r in frames),dtype=np.float64);assert len(times)==200
 train=scene.getTrainCameras();cameras={}
 for j,r in enumerate(frames):
  i=int(np.argmin(abs(times-r['time'])));assert abs(times[i]-r['time'])<1e-7 and abs(float(train[j].time)-r['time'])<1e-7
  cameras[i]=small_camera(train[j],400)
 assert len(cameras)==200
 from utils.loss_utils import ssim
 from utils.image_utils import psnr
 import lpips
 def loss_fn(p,t):return .8*(p-t).abs().mean()+.2*(1-ssim(p.unsqueeze(0),t.unsqueeze(0)))
 def render(arm,i,cam,model):
  color,alpha=model(ids)
  return render_state(cam,tensor(geometries[arm][i]),color,alpha,cfg.sh_degree)
 teacher={}
 with torch.no_grad():
  for i in fit:teacher[i]=render_state(cameras[i],tensor(full[i]),sh,opacity,cfg.sh_degree).detach()
 models={'fixed_E19_mean_beta0':initial};logs=[];hashes={}
 for arm in arms:
  model=Appearance(sh,logits,k);optimizer=model.optimizer({'dc':0.0025,'rest':0.000125,'opacity':0.01})
  for epoch,order in enumerate(orders):
   losses=[]
   for i in order:
    optimizer.zero_grad(set_to_none=True);pred=render(arm,i,cameras[i],model);loss=loss_fn(pred,teacher[i]);assert torch.isfinite(loss)
    loss.backward();optimizer.step();losses.append(float(loss.detach()))
   row={'arm':arm,'epoch':epoch+1,'updates':(epoch+1)*101,'mean_fit_loss':float(np.mean(losses)),'elapsed_seconds':time.monotonic()-start};logs.append(row)
   (out/'progress.json').write_text(json.dumps(row,indent=2)+'\n')
   if (epoch+1)%5==0:print('TRAIN',json.dumps(row),flush=True)
  assert all(int(s['step'])==2020 for s in optimizer.state.values())
  assert all(torch.isfinite(p).all() for p in model.parameters())
  path=out/(arm+'-appearance.pt');torch.save({'state_dict':{q:v.detach().cpu() for q,v in model.state_dict().items()},'optimizer':optimizer.state_dict(),'steps':2020,'arm':arm},path)
  hashes[path.name]=digest(path);models[arm]=model
 dump(out/'training.json',logs);dump(out/'training-complete.json',{'two_new_models_frozen_before_GT':True,'control_reused_and_hash_verified':True,'checkpoint_hashes':hashes,'new_gradient_updates':4040})
 print('BOTH NEW MODELS FROZEN; EVALUATING',flush=True);teacher.clear();criterion=lpips.LPIPS(net='alex',version='0.1').cuda().eval()
 gtrows=[];fidelity=[];temporal=[];previous=None
 with torch.no_grad():
  tests=json.loads((data/'transforms_test.json').read_text())['frames']
  for j,r in enumerate(tests):
   i=int(np.argmin(abs(times-r['time'])));assert abs(times[i]-r['time'])<1e-7
   gt=tensor(np.asarray(Image.open(base/f'test/ours_20000/gt/{j:05d}.png').convert('RGB'),dtype=np.float32)/255).permute(2,0,1).unsqueeze(0)
   for arm,model in models.items():
    pred=(render(arm,i,scene.getTestCameras()[j],model)*255).round()/255;p=pred.unsqueeze(0)
    gtrows.append({'arm':arm,'view':j,'PSNR':float(psnr(p,gt)),'SSIM':float(ssim(p,gt)),'LPIPS_alex_v01':float(criterion(p,gt,normalize=True))})
  print('GT COMPLETE',flush=True)
  fixed_cam=small_camera(scene.getTestCameras()[10],256)
  for i in range(200):
   target=render_state(cameras[i],tensor(full[i]),sh,opacity,cfg.sh_degree)
   images={'teacher':render_state(fixed_cam,tensor(full[i]),sh,opacity,cfg.sh_degree)}
   for arm,model in models.items():
    p=render(arm,i,cameras[i],model)
    fidelity.append({'arm':arm,'time_index':i,'split':'fit' if i in fit else 'holdout','PSNR_to_teacher':float(psnr(p.unsqueeze(0),target.unsqueeze(0))),'L1_to_teacher':float((p-target).abs().mean()),'mixed_loss_to_teacher':float(loss_fn(p,target))})
    images[arm]=render(arm,i,fixed_cam,model)
   if previous is not None:
    for arm in models:temporal.append({'arm':arm,'time_index':i,'residual':float(((images[arm]-previous[arm])-(images['teacher']-previous['teacher'])).abs().mean())})
   previous=images
   if i%50==0:print('FIDELITY/TEMPORAL',i,flush=True)
 dump(out/'gt-per-view.json',gtrows);dump(out/'fidelity-per-view.json',fidelity);dump(out/'temporal-per-pair.json',temporal)
 rows=[]
 for arm in models:
  gt=[r for r in gtrows if r['arm']==arm];tmp=[r['residual'] for r in temporal if r['arm']==arm]
  row={'arm':arm,**{m:float(np.mean([r[m] for r in gt])) for m in ['PSNR','SSIM','LPIPS_alex_v01']},'temporal_residual256':float(np.mean(tmp))}
  for split in ['fit','holdout']:
   vals=[r for r in fidelity if r['arm']==arm and r['split']==split]
   row[split]={m:float(np.mean([r[m] for r in vals])) for m in ['PSNR_to_teacher','L1_to_teacher','mixed_loss_to_teacher']}
  row['distribution']={split:{m:float(np.mean([r[m] for r in distortion if r['arm']==arm and r['split']==split])) for m in ['full12_weighted_cost','position_weighted_cost']} for split in ['fit','holdout']}
  if arm in arms:
   row['switches_total']=sum(e['switches'] for e in switch_logs[arm]);row['mean_disagreement_with_fixed']=float(np.mean([e['disagreement_with_fixed']/n for e in switch_logs[arm]]))
  rows.append(row)
 control=next(r for r in control_report['rows'] if r['arm']=='mean_covariance__beta0')
 reproduced=next(r for r in rows if r['arm']=='fixed_E19_mean_beta0')
 assert abs(reproduced['PSNR']-control['PSNR'])<1e-4 and abs(reproduced['LPIPS_alex_v01']-control['LPIPS_alex_v01'])<1e-6
 by={r['arm']:r for r in rows};metrics=['PSNR','SSIM','LPIPS_alex_v01','temporal_residual256']
 paired={arm:{m:by[arm][m]-by['fixed_E19_mean_beta0'][m] for m in metrics} for arm in arms}
 report={'rows':rows,'paired_new_minus_fixed':paired,'control_E19_reproduction':{m:reproduced[m]-control[m] for m in metrics},'switch_summary':{arm:{'total':sum(e['switches'] for e in switch_logs[arm]),'mean_per_window':float(np.mean([e['switches'] for e in switch_logs[arm]])),'max_per_window':max(e['switches'] for e in switch_logs[arm]),'mean_disagreement_with_fixed':by[arm]['mean_disagreement_with_fixed']} for arm in arms},'scope':protocol['scope']}
 dump(out/'report.json',report)
 for p,h in inputs.items():assert digest(p)==h,str(p)
 for name,h in hashes.items():assert digest(out/name)==h,name
 dump(out/'completion.json',{'status':'completed','inputs_unchanged':True,'checkpoints_verified':True,'source_sha256':digest(out/'source.zip'),'seconds':time.monotonic()-start})
 print('RESULT',json.dumps(report),flush=True)

if __name__=='__main__':main()
