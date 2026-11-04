#!/usr/bin/env python3
"""Matched-count prefix controls, derived only from fit-gate accepted counts."""
import argparse,json,os,pathlib,zipfile
import numpy as np
import torch
from appearance_model import Appearance
from training_split_core import split_adam,digest_state,RATES
from reduction_scene import load_scene,small_camera,render_state
from run_ctd_comparison import dump,digest

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args();root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();repo=root/'projects/dynamic-gaussian-ot-densification-research';out=a.output.resolve();assert out.is_relative_to(root) and not out.exists()
 source=root/'outputs/ot/e06-training-event-20261104T113535Z';manifest=json.loads((source/'manifest.json').read_text());complete=json.loads((source/'completion.json').read_text());assert complete['status']=='completed';protocol=json.loads((repo/'experiments/training-split-matched-count-v1.json').read_text());proposal_root=root/'outputs/ot/e05-split-20261104T103723Z';pm=json.loads((proposal_root/'manifest.json').read_text());proposals=json.loads((proposal_root/'proposal-summary.json').read_text())
 inputs={pathlib.Path(p):h for p,h in manifest['inputs'].items()};inputs[source/'warmup.pt']=digest(source/'warmup.pt')
 for name in ['continue','gated_window','gated_random']:
  inputs[source/'validation'/(name+'-appearance.pt')]=digest(source/'validation'/(name+'-appearance.pt'))
  if name!='continue':inputs[source/'validation'/(name+'-states.npy')]=digest(source/'validation'/(name+'-states.npy'))
 for p,h in inputs.items():assert digest(p)==h
 base=root/'outputs/dnerf/bouncingballs/baseline-numpy1235-20261030T174401Z';cfg,pc,scene,pipe=load_scene(base)
 for v in vars(pc).values():
  if isinstance(v,torch.nn.Parameter):v.requires_grad_(False)
  elif isinstance(v,torch.nn.Module):v.requires_grad_(False);v.eval()
 def tensor(x):return torch.tensor(np.array(x,copy=True),device='cuda')
 cache=pathlib.Path(pm['source_cache']);full=np.load(cache/'states.npy',mmap_mode='r');times=np.load(cache/'times.npy');sh=tensor(np.load(cache/'sh.npy'));opacity=tensor(np.load(cache/'opacity.npy'));geometry=np.load(pm['parent_states'],mmap_mode='r');child_geometry=np.load(proposal_root/'child-states.npy',mmap_mode='r');mass=np.load(proposal_root/'children.npz')['child_mass'];fit=manifest['fit']
 models={};geometries={}
 for name in ['continue','gated_window','gated_random']:
  state=torch.load(source/'validation'/(name+'-appearance.pt'),map_location='cuda')['state_dict'];m=Appearance(state['source_sh'],state['source_logits'],len(state['dc']));m.load_state_dict(state);models[name]=m;geometries[name]=geometry if name=='continue' else np.load(source/'validation'/(name+'-states.npy'),mmap_mode='r')
 initial_ck=torch.load(pm['appearance_checkpoint'],map_location='cuda')['state_dict'];reps=torch.tensor(pm['parent_representatives'],device='cuda',dtype=torch.long);initial_sh=(initial_ck['source_sh'][reps]+torch.cat([initial_ck['dc'],initial_ck['rest']],1)).detach();initial_logits=(initial_ck['source_logits'][reps]+initial_ck['opacity']).detach()
 cameras={};frames=json.loads((root/'datasets/bouncingballs/transforms_train.json').read_text())['frames'];train=scene.getTrainCameras()
 for j,r in enumerate(frames):
  i=int(np.argmin(abs(times-r['time'])));cam=train[j];assert abs(float(cam.time)-r['time'])<1e-7;cameras[i]=small_camera(cam,400)
 from utils.loss_utils import ssim
 def image(model,geo,i,cam=None):
  colors,alpha=model(torch.arange(len(model.dc),device='cuda'));return render_state(cameras[i] if cam is None else cam,tensor(geo[i]),colors,alpha,cfg.sh_degree)
 def loss(pred,target):return .8*(pred-target).abs().mean()+.2*(1-ssim(pred.unsqueeze(0),target.unsqueeze(0)))
 teacher={}
 with torch.no_grad():
  for i in fit:teacher[i]=render_state(cameras[i],tensor(full[i]),sh,opacity,cfg.sh_degree).detach()
 out.mkdir(parents=True);dest=out/'validation';dest.mkdir();selected_counts={};training=[]
 for name,source_name,method in [('prefix_window','gated_window','window'),('prefix_random','gated_random','random')]:
  count=complete['counts'][source_name]-10000;assert count>0;parents=np.array(proposals['methods'][method]['selected'][:count]);selected_counts[name]=count
  warm=torch.load(source/'warmup.pt',map_location='cuda');m=Appearance(warm['model']['source_sh'],warm['model']['source_logits'],10000);m.load_state_dict(warm['model']);opt=m.optimizer(RATES);opt.load_state_dict(warm['optimizer']);assert digest_state(m,opt)==warm['digest']
  m,opt,_=split_adam(m,opt,parents,mass[parents*2]/(mass[parents*2]+mass[parents*2+1]));geo=np.concatenate([np.array(geometry,copy=True),child_geometry[:,parents*2+1]],axis=1);geo[:,parents]=child_geometry[:,parents*2]
  for e,order in enumerate(manifest['orders'][10:]):
   values=[]
   for i in order:
    opt.zero_grad(set_to_none=True);value=loss(image(m,geo,i),teacher[i]);assert torch.isfinite(value);value.backward();opt.step();values.append(float(value.detach()))
   row={'arm':name,'epoch':e+1,'main_steps':760+(e+1)*76,'loss':float(np.mean(values))};training.append(row);print('TRAIN',json.dumps(row),flush=True)
  assert all(torch.isfinite(p).all() for p in m.parameters());torch.save({'state_dict':{k:v.detach().cpu() for k,v in m.state_dict().items()},'optimizer':opt.state_dict(),'main_steps':1520,'selected_parents':parents.tolist()},dest/(name+'-appearance.pt'));np.save(dest/(name+'-states.npy'),geo);models[name]=m;geometries[name]=geo
 teacher.clear();models={name:models[name] for name in protocol['evaluated_arms']};dump(out/'manifest.json',{'protocol':protocol,'inputs':{str(p):h for p,h in inputs.items()},'source_event':str(source),'selected_counts':selected_counts,'post_orders':manifest['orders'][10:],'GPU':os.environ.get('CUDA_VISIBLE_DEVICES')});dump(out/'training.json',training);dump(out/'proposal-summary.json',proposals)
 with zipfile.ZipFile(out/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
  for name in ['run_matched_event_counts.py','training_split_core.py','appearance_model.py','reduction_scene.py','evaluate_split_validation.py','run_ctd_comparison.py','reduction_core.py','inspect_baseline.py','common.sh']:z.write(repo/'scripts'/name,'scripts/'+name)
  z.write(repo/'experiments/training-split-matched-count-v1.json','experiments/training-split-matched-count-v1.json')
 def render(name,i,cam,after=True):
  if after:return image(models[name],geometries[name],i,cam)
  return render_state(cam,tensor(geometry[i]),initial_sh,torch.sigmoid(initial_logits),cfg.sh_degree)
 from evaluate_split_validation import evaluate
 evaluate(dest,base,out,protocol,times,fit,full,sh,opacity,cfg,scene,cameras,models,render,tensor)
 for p,h in inputs.items():assert digest(p)==h
 dump(out/'completion.json',{'status':'completed','source_sha256':digest(out/'source.zip'),'counts':{n:len(m.dc) for n,m in models.items()},'inputs_unchanged':True});print('MATCHED COUNTS COMPLETE',flush=True)
if __name__=='__main__':main()
