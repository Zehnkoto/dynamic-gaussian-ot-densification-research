#!/usr/bin/env python3
"""E07: one-factor gate sensitivity, with exact E06 warmup reuse."""
import argparse,json,os,pathlib,time,zipfile
import numpy as np
import torch
from appearance_model import Appearance
from training_split_core import clone_adam,split_adam,digest_state,render_gate,RATES
from reduction_scene import load_scene,small_camera,render_state
from run_ctd_comparison import digest,dump

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args();root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();repo=root/'projects/dynamic-gaussian-ot-densification-research';out=a.output.resolve()
 assert out.is_relative_to(root) and not out.exists();protocol=json.loads((repo/'experiments/gate-sensitivity-v1.json').read_text());inp=root/'outputs/ot/e05-split-20261104T103723Z';original=json.loads((inp/'manifest.json').read_text());completed=json.loads((inp/'completion.json').read_text());assert completed['status']=='completed'
 assert (protocol['warmup_epochs']+protocol['post_epochs'])*76==protocol['main_updates_per_arm']
 inputs={pathlib.Path(p):h for p,h in original['inputs'].items()}
 for name in ['child-states.npy','children.npz','proposal-summary.json']:inputs[inp/name]=digest(inp/name)
 source=root/'outputs/ot/e06-training-event-20261104T113535Z'; source_manifest=json.loads((source/'manifest.json').read_text()); assert json.loads((source/'completion.json').read_text())['status']=='completed'
 for name in ['warmup.pt','manifest.json','validation/gated_window-appearance.pt','validation/gated_window-states.npy','validation/continue-appearance.pt']:inputs[source/name]=digest(source/name)
 for p,h in inputs.items():assert p.is_relative_to(root) and digest(p)==h
 base=root/'outputs/dnerf/bouncingballs/baseline-numpy1235-20261030T174401Z';cfg,pc,scene,pipe=load_scene(base)
 for item in vars(pc).values():
  if isinstance(item,torch.nn.Parameter):item.requires_grad_(False)
  elif isinstance(item,torch.nn.Module):item.requires_grad_(False);item.eval()
 def tensor(x):return torch.tensor(np.array(x,copy=True),device='cuda')
 cache=pathlib.Path(original['source_cache']);full=np.load(cache/'states.npy',mmap_mode='r');times=np.load(cache/'times.npy');sh=tensor(np.load(cache/'sh.npy'));opacity=tensor(np.load(cache/'opacity.npy'));assert torch.equal(pc.get_features,sh) and torch.equal(pc.get_opacity,opacity)
 geometry=np.load(original['parent_states'],mmap_mode='r');child_geometry=np.load(inp/'child-states.npy',mmap_mode='r');child=np.load(inp/'children.npz');mass=child['child_mass'];gains=child['gains'];proposals=json.loads((inp/'proposal-summary.json').read_text())
 ck=torch.load(original['appearance_checkpoint'],map_location='cuda')['state_dict'];reps=torch.tensor(original['parent_representatives'],device='cuda',dtype=torch.long);initial_sh=(ck['source_sh'][reps]+torch.cat([ck['dc'],ck['rest']],1)).detach();initial_logits=(ck['source_logits'][reps]+ck['opacity']).detach()
 fit=original['fit'];held=original['holdout'];assert len(fit)==76 and len(held)==74 and not set(fit)&set(held)
 bins=np.minimum((times*5).astype(int),4);decision=[];decision_bins=[]
 for b in range(5):
  pool=[i for i in fit if bins[i]==b];chosen=np.array(pool)[np.linspace(0,len(pool)-1,4,dtype=int)].tolist();decision+=chosen;decision_bins+=[b]*4
 probe_fit=[i for i in fit if i not in decision];assert len(set(decision))==20 and len(probe_fit)==56
 orders=[np.random.default_rng(seq).permutation(fit).tolist() for seq in np.random.SeedSequence(protocol['seed']).spawn(protocol['epochs'])]
 batch_size=protocol['batch_size'];blocks=protocol['max_splits']//batch_size;assert protocol['max_splits']%batch_size==0
 probe_orders=[]
 for b in range(blocks):
  rng=np.random.default_rng(np.random.SeedSequence([protocol['seed'],2718,b])); stream=[]
  while len(stream)<80:stream+=rng.permutation(probe_fit).tolist()
  probe_orders.append(stream[:80]); assert stream[:20]==source_manifest['probe_orders'][b]
 assert orders==source_manifest['orders'] and decision==source_manifest['decision_views']
 assert all(set(order)<=set(probe_fit) for order in probe_orders) and not set(decision)&set(held)
 cameras={};frames=json.loads((root/'datasets/bouncingballs/transforms_train.json').read_text())['frames'];train=scene.getTrainCameras()
 for j,r in enumerate(frames):
  i=int(np.argmin(abs(times-r['time'])));cam=train[j];assert abs(float(cam.time)-r['time'])<1e-7;cameras[i]=small_camera(cam,protocol['training_size'])
 out.mkdir(parents=True);dest=out/'validation';dest.mkdir();dump(out/'manifest.json',{'protocol':protocol,'inputs':{str(p):h for p,h in inputs.items()},'fit':fit,'holdout':held,'decision_views':decision,'decision_window_ids':decision_bins,'probe_training_views':probe_fit,'orders':orders,'probe_orders':probe_orders,'GPU':os.environ.get('CUDA_VISIBLE_DEVICES')});dump(out/'proposal-summary.json',proposals)
 with zipfile.ZipFile(out/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
  for name in ['run_gate_sensitivity.py','training_split_core.py','test_training_split_core.py','appearance_model.py','reduction_scene.py','evaluate_split_validation.py','run_ctd_comparison.py','reduction_core.py','inspect_baseline.py','common.sh']:z.write(repo/'scripts'/name,'scripts/'+name)
  z.write(repo/'experiments/gate-sensitivity-v1.json','experiments/gate-sensitivity-v1.json')
 from utils.loss_utils import ssim
 def image(model,geo,i,cam=None):
  colors,alpha=model(torch.arange(len(model.dc),device='cuda'));return render_state(cameras[i] if cam is None else cam,tensor(geo[i]),colors,alpha,cfg.sh_degree)
 teacher={}
 with torch.no_grad():
  for i in fit:teacher[i]=render_state(cameras[i],tensor(full[i]),sh,opacity,cfg.sh_degree).detach()
 def loss(pred,target):return .8*(pred-target).abs().mean()+.2*(1-ssim(pred.unsqueeze(0),target.unsqueeze(0)))
 def update(model,opt,geo,i):
  opt.zero_grad(set_to_none=True);value=loss(image(model,geo,i),teacher[i]);assert torch.isfinite(value);value.backward();opt.step();return float(value.detach())
 @torch.no_grad()
 def decision_losses(model,geo):return [float(loss(image(model,geo,i),teacher[i])) for i in decision]
 warm=torch.load(source/'warmup.pt',map_location='cuda');live=Appearance(warm['model']['source_sh'],warm['model']['source_logits'],10000);live.load_state_dict(warm['model']);optimizer=live.optimizer(RATES);optimizer.load_state_dict(warm['optimizer']);warm_digest=digest_state(live,optimizer);assert warm_digest==warm['digest'] and warm['main_steps']==760
 training=[];started=time.monotonic()
 models={};geometries={};events=[];traces=[]
 for name in protocol['arms']:
  policy=protocol['policies'][name]; steps=policy['probe_steps']; price=policy['price']
  assert __import__('shutil').disk_usage(root).free>3*1024**3
  model,opt=clone_adam(live,optimizer);geo=geometry;accepted=[];assert digest_state(model,opt)==warm_digest
  traces.append({'arm':name,'main_step':protocol['event_step'],'proposal_block':-1,'gaussians':len(model.dc)})
  if name!='continue':
   method='window';candidates=proposals['methods'][method]['selected'];assert len(candidates)==protocol['max_splits']
   for block in range(blocks):
    parents=np.array(candidates[block*batch_size:(block+1)*batch_size]);assert not set(parents)&set(accepted)
    fractions=mass[parents*2]/(mass[parents*2]+mass[parents*2+1]);ctd_gain=float(gains[fit][:,parents].sum(1).mean());assert ctd_gain>0
    prior=digest_state(model,opt);ref,refopt=clone_adam(model,opt);proposed,propopt,optical_error=split_adam(model,opt,parents,fractions)
    proposed_geo=np.concatenate([np.array(geo,copy=True),child_geometry[:,parents*2+1]],axis=1);proposed_geo[:,parents]=child_geometry[:,parents*2];proposal_digest=digest_state(proposed,propopt)
    immediate_ref=decision_losses(model,geo);immediate_prop=decision_losses(proposed,proposed_geo)
    for i in probe_orders[block][:steps]:update(ref,refopt,geo,i);update(proposed,propopt,proposed_geo,i)
    scores=render_gate(decision_losses(ref,geo),decision_losses(proposed,proposed_geo),decision_bins,batch_size,price,protocol['max_window_relative_worsening'])
    assert digest_state(model,opt)==prior
    committed=name.startswith('forced') or scores['accept'];before_count=len(model.dc)
    if committed:
     model,opt,_=split_adam(model,opt,parents,fractions);geo=proposed_geo;accepted+=parents.tolist();assert digest_state(model,opt)==proposal_digest
    else:assert digest_state(model,opt)==prior
    row={'arm':name,'block':block,'main_step':protocol['event_step'],'parents':parents.tolist(),'fit_CTD_gain':ctd_gain,'scores':scores,'committed':committed,'before_count':before_count,'after_count':len(model.dc),'immediate_render_gain':float(np.mean(immediate_ref)-np.mean(immediate_prop)),'optical_depth_max_error':optical_error,'live_unchanged_during_probe':True,'commit_uses_untrained_proposal':bool(committed),'rejected_rollback_exact':bool(not committed),'probe_updates_each_counterfactual':steps,'complexity_price_per_gaussian':price};events.append(row)
    traces.append({'arm':name,'main_step':protocol['event_step'],'proposal_block':block,'gaussians':len(model.dc)});print('EVENT',json.dumps({k:row[k] for k in ['arm','block','committed','before_count','after_count']}),flush=True)
    del ref,refopt,proposed,propopt
  for e,order in enumerate(orders[protocol['warmup_epochs']:]):
   values=[update(model,opt,geo,i) for i in order];row={'arm':name,'epoch':e+1,'main_updates':protocol['event_step']+(e+1)*76,'loss':float(np.mean(values)),'gaussians':len(model.dc)};training.append(row);print('POST',json.dumps(row),flush=True)
   with (out/'progress.json').open('w') as f:json.dump(row,f)
  assert all(torch.isfinite(p).all() for p in model.parameters())
  torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':opt.state_dict(),'main_steps':protocol['main_updates_per_arm'],'accepted_parents':accepted,'protocol':protocol},dest/(name+'-appearance.pt'))
  if name!='continue':np.save(dest/(name+'-states.npy'),geo)
  models[name]=model;geometries[name]=geo;traces.append({'arm':name,'main_step':1520,'proposal_block':blocks,'gaussians':len(model.dc)})
 assert digest_state(live,optimizer)==warm_digest
 dump(out/'event-decisions.json',events);dump(out/'count-traces.json',traces);dump(out/'training.json',training);teacher.clear()
 for name,source_name in [('continue','continue'),('price1e7_probe20','gated_window')]:
  ck=torch.load(source/'validation'/(source_name+'-appearance.pt'),map_location='cuda')['state_dict'];m=Appearance(ck['source_sh'],ck['source_logits'],len(ck['dc']));m.load_state_dict(ck);models[name]=m;geometries[name]=geometry if name=='continue' else np.load(source/'validation'/(source_name+'-states.npy'),mmap_mode='r')
 def render(name,i,cam,after=True):
  if after:return image(models[name],geometries[name],i,cam)
  return render_state(cam,tensor(geometry[i]),initial_sh,torch.sigmoid(initial_logits),cfg.sh_degree)
 from evaluate_split_validation import evaluate
 evaluate(dest,base,out,protocol,times,fit,full,sh,opacity,cfg,scene,cameras,models,render,tensor)
 for p,h in inputs.items():assert digest(p)==h
 dump(out/'completion.json',{'status':'completed','source_sha256':digest(out/'source.zip'),'inputs_unchanged':True,'counts':{n:len(m.dc) for n,m in models.items()},'main_updates_each':1520,'probe_updates_by_arm':{n:blocks*2*v['probe_steps'] for n,v in protocol['policies'].items()},'seconds':time.monotonic()-started});print('GATE SENSITIVITY COMPLETE',flush=True)
if __name__=='__main__':main()
