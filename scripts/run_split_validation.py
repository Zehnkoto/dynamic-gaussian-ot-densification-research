#!/usr/bin/env python3
"""Paired offline split validation; all arms continue appearance for fixed1520steps."""
import argparse,json,os,pathlib,time
import numpy as np
import torch
from appearance_model import Appearance
from reduction_scene import load_scene,small_camera,render_state
from run_ctd_comparison import digest,dump
ap=argparse.ArgumentParser();ap.add_argument('--input',type=pathlib.Path,required=True);args=ap.parse_args();root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();inp=args.input.resolve();out=inp/'validation'
assert inp.is_relative_to(root) and not out.exists();manifest=json.loads((inp/'manifest.json').read_text());protocol=manifest['protocol'];completion=json.loads((inp/'completion.json').read_text());assert completion['status']=='completed'
for p,h in manifest['inputs'].items():assert digest(pathlib.Path(p))==h
for n,h in completion['artifacts'].items():assert digest(inp/n)==h
base=root/'outputs/dnerf/bouncingballs/baseline-numpy1235-20261030T174401Z';cfg,pc,scene,pipe=load_scene(base)
for a in vars(pc).values():
 if isinstance(a,torch.nn.Parameter):a.requires_grad_(False)
 elif isinstance(a,torch.nn.Module):a.requires_grad_(False);a.eval()
cache=pathlib.Path(manifest['source_cache']);full=np.load(cache/'states.npy',mmap_mode='r');times=np.load(cache/'times.npy')
def tensor(a):return torch.tensor(np.array(a,copy=True),device='cuda')
source_sh=tensor(np.load(cache/'sh.npy'));source_op=tensor(np.load(cache/'opacity.npy'));assert torch.equal(pc.get_features,source_sh) and torch.equal(pc.get_opacity,source_op)
ck=torch.load(manifest['appearance_checkpoint'],map_location='cuda')['state_dict'];reps=torch.tensor(manifest['parent_representatives'],device='cuda',dtype=torch.long)
parent_sh=(ck['source_sh'][reps]+torch.cat([ck['dc'],ck['rest']],1)).detach();parent_logits=(ck['source_logits'][reps]+ck['opacity']).detach();children=np.load(inp/'children.npz');mass=children['child_mass'];proposals=json.loads((inp/'proposal-summary.json').read_text())
features={'continue':np.load(manifest['parent_states'],mmap_mode='r')};models={'continue':Appearance(parent_sh,parent_logits,protocol['base_budget'])};optical_checks={}
for name,r in proposals['methods'].items():
 ids=np.array(r['selected']);features[name]=np.load(inp/(name+'-states.npy'),mmap_mode='r');sh=torch.cat([parent_sh,parent_sh[ids]],0).clone();logits=torch.cat([parent_logits,parent_logits[ids]],0).clone()
 fractions=tensor(mass[ids*2]/(mass[ids*2]+mass[ids*2+1])).double().reshape(-1,1);tau=torch.nn.functional.softplus(parent_logits[ids].double())
 logits[ids]=torch.log(torch.expm1(tau*fractions)).float();logits[protocol['base_budget']:]=torch.log(torch.expm1(tau*(1-fractions))).float()
 error=float((torch.nn.functional.softplus(logits[ids].double())+torch.nn.functional.softplus(logits[protocol['base_budget']:].double())-tau).abs().max());assert error<1e-5;optical_checks[name]=error
 models[name]=Appearance(sh,logits,protocol['final_budget'])
indices={name:torch.arange(len(model.dc),device='cuda') for name,model in models.items()}
def render(name,i,cam,after=True):
 m=models[name];sh,op=m(indices[name]) if after else (m.source_sh,torch.sigmoid(m.source_logits))
 return render_state(cam,tensor(features[name][i]),sh,op,cfg.sh_degree)
frames=json.loads((root/'datasets/bouncingballs/transforms_train.json').read_text())['frames'];cameras={};train=scene.getTrainCameras()
for j,r in enumerate(frames):
 i=int(np.argmin(abs(times-r['time'])));cam=train[j];assert abs(float(cam.time)-r['time'])<1e-7;cameras[i]=small_camera(cam,protocol['training_size'])
fit=manifest['fit'];held=manifest['holdout'];assert len(cameras)==150 and not set(fit)&set(held)
orders=[np.random.default_rng(protocol['seed']+e).permutation(fit).tolist() for e in range(protocol['epochs'])]
out.mkdir();dump(out/'manifest.json',{'protocol':protocol,'input_source_sha256':completion['source_sha256'],'orders':orders,'fit':fit,'holdout':held,'GPU':os.environ.get('CUDA_VISIBLE_DEVICES'),'optical_depth_sum_max_error':optical_checks,'counts':{n:len(m.dc) for n,m in models.items()},'trainable_parameters':{n:sum(p.numel() for p in m.parameters()) for n,m in models.items()}})
from utils.loss_utils import ssim
teacher={}
with torch.no_grad():
 for i in fit:teacher[i]=render_state(cameras[i],tensor(full[i]),source_sh,source_op,cfg.sh_degree).detach()
def loss_fn(pred,target):return .8*(pred-target).abs().mean()+.2*(1-ssim(pred.unsqueeze(0),target.unsqueeze(0)))
log=[];start=time.monotonic()
for name,model in models.items():
 opt=model.optimizer({'dc':.0025,'rest':.000125,'opacity':.01})
 pred=render(name,fit[0],cameras[fit[0]])
 with torch.no_grad():assert torch.equal(pred,render(name,fit[0],cameras[fit[0]],False))
 loss_fn(pred,teacher[fit[0]]).backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.parameters());opt.zero_grad(set_to_none=True)
 for epoch,order in enumerate(orders):
  losses=[]
  for i in order:
   opt.zero_grad(set_to_none=True);loss=loss_fn(render(name,i,cameras[i]),teacher[i]);assert torch.isfinite(loss);loss.backward();opt.step();losses.append(float(loss.detach()))
  row={'arm':name,'epoch':epoch+1,'updates':(epoch+1)*76,'mean_loss':float(np.mean(losses)),'seconds':time.monotonic()-start};log.append(row);print('TRAIN',json.dumps(row),flush=True)
  with (out/'progress.json').open('w') as f:json.dump(row,f)
 assert all(torch.isfinite(p).all() for p in model.parameters())
 torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'steps':1520,'protocol':protocol},out/(name+'-appearance.pt'))
dump(out/'training.json',log);teacher.clear();print('TRAINING COMPLETE',flush=True)
from evaluate_split_validation import evaluate
evaluate(out,base,inp,protocol,times,fit,full,source_sh,source_op,cfg,scene,cameras,models,render,tensor)
for p,h in manifest['inputs'].items():assert digest(pathlib.Path(p))==h
for n,h in completion['artifacts'].items():assert digest(inp/n)==h
dump(out/'completion.json',{'status':'completed','inputs_unchanged':True,'arms':list(models),'source_sha256':completion['source_sha256']});print('SPLIT VALIDATION COMPLETE',flush=True)
