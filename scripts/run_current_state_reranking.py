#!/usr/bin/env python3
"""E09: current-state batch reranking with a shared second-event checkpoint."""
import argparse,json,os,pathlib,time,zipfile
import numpy as np
import torch
from appearance_model import Appearance
from training_split_core import clone_adam,split_adam,digest_state,render_gate,RATES
from reduction_scene import load_scene,small_camera,render_state
from run_ctd_comparison import digest,dump

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args();root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();repo=root/'projects/dynamic-gaussian-ot-densification-research';out=a.output.resolve()
 assert out.is_relative_to(root) and not out.exists();protocol=json.loads((repo/'experiments/current-state-reranking-v1.json').read_text());inp=root/'outputs/ot/e05-split-20261104T103723Z';original=json.loads((inp/'manifest.json').read_text());completed=json.loads((inp/'completion.json').read_text());assert completed['status']=='completed'
 assert (protocol['warmup_epochs']+protocol['post_epochs'])*76==protocol['main_updates_per_arm']
 inputs={pathlib.Path(p):h for p,h in original['inputs'].items()}
 for name in ['child-states.npy','children.npz','proposal-summary.json','scores.npz']:inputs[inp/name]=digest(inp/name)
 source=root/'outputs/ot/e06-training-event-20261104T113535Z'; source_manifest=json.loads((source/'manifest.json').read_text()); assert json.loads((source/'completion.json').read_text())['status']=='completed'
 for name in ['warmup.pt','manifest.json','validation/gated_window-appearance.pt','validation/gated_window-states.npy','validation/continue-appearance.pt']:inputs[source/name]=digest(source/name)
 e08=root/'outputs/ot/e08-two-event-20261104T123454Z'
 inputs[e08/'validation/summary.json']=digest(e08/'validation/summary.json');inputs[e08/'validation/twice_forced-appearance.pt']=digest(e08/'validation/twice_forced-appearance.pt')
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
  for name in ['run_current_state_reranking.py','reranking_core.py','split_proposal_core.py','test_reranking_core.py','test_repeated_split_adam.py','training_split_core.py','test_training_split_core.py','appearance_model.py','reduction_scene.py','evaluate_split_validation.py','run_ctd_comparison.py','reduction_core.py','inspect_baseline.py','common.sh']:z.write(repo/'scripts'/name,'scripts/'+name)
  z.write(repo/'experiments/current-state-reranking-v1.json','experiments/current-state-reranking-v1.json')
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
 training=[];started=time.monotonic();events=[];scoring=[]
 from split_proposal_core import selections
 from reranking_core import choose_batches
 _,ranked,_=selections(gains,child['eligible'],fit,times,protocol['pool_size'],42)
 ranked=ranked['window'];assert ranked[:256].tolist()==proposals['methods']['window']['selected']
 first=ranked[:128];pool=ranked[128:].reshape(24,16);assert len(set(ranked))==512
 def apply(model,opt,geo,parents):
  fractions=mass[parents*2]/(mass[parents*2]+mass[parents*2+1]);new,nopt,error=split_adam(model,opt,parents,fractions)
  g=np.concatenate([np.array(geo,copy=True),child_geometry[:,parents*2+1]],1);g[:,parents]=child_geometry[:,parents*2]
  return new,nopt,g,error
 for b in range(8):live,optimizer,geometry_live,error=apply(live,optimizer,geometry if b==0 else geometry_live,first[b*16:(b+1)*16])
 first_digest=digest_state(live,optimizer);torch.save({'model':live.state_dict(),'optimizer':optimizer.state_dict(),'main_steps':760,'digest':first_digest,'selected_parents':first.tolist()},out/'after-first.pt')
 @torch.no_grad()
 def score_pool(model,opt,geo,step):
  start=time.monotonic();before=digest_state(model,opt);reference=decision_losses(model,geo);rows=[]
  for b,parents in enumerate(pool):
   proposed,popt,g,error=apply(model,opt,geo,parents);values=decision_losses(proposed,g);row={'step':step,'batch':b,'parents':parents.tolist(),'reference':reference,'proposal':values,'mean_render_gain':float(np.mean(np.array(reference)-values)),'fit_CTD_gain':float(gains[fit][:,parents].sum(1).mean())};assert row['fit_CTD_gain']>0;rows.append(row);print('SCORE',json.dumps({k:row[k] for k in ['step','batch','mean_render_gain']}),flush=True);del proposed,popt,g
  assert digest_state(model,opt)==before
  scoring.extend(rows);return [r['mean_render_gain'] for r in rows],{'seconds':time.monotonic()-start,'reference_renders':20,'proposal_renders':480,'live_digest_before_after':before,'main_step':step}
 stale_scores,stale_cost=score_pool(live,optimizer,geometry_live,760)
 for e,order in enumerate(orders[10:15]):
  values=[update(live,optimizer,geometry_live,i) for i in order];row={'arm':'shared_middle','epoch':e+1,'main_updates':760+(e+1)*76,'gaussians':10128,'loss':float(np.mean(values))};training.append(row);print('MIDDLE',json.dumps(row),flush=True)
 shared_digest=digest_state(live,optimizer);torch.save({'model':live.state_dict(),'optimizer':optimizer.state_dict(),'main_steps':1140,'digest':shared_digest,'selected_parents':first.tolist()},out/'before-second.pt')
 current_scores,current_cost=score_pool(live,optimizer,geometry_live,1140)
 chosen={'fixed_ctd':list(range(8)),'current_render':choose_batches(current_scores,8),'stale_render':choose_batches(stale_scores,8),'random':np.random.default_rng(np.random.SeedSequence(protocol['random_seed_sequence'])).choice(24,8,replace=False).tolist()}
 dump(out/'selection.json',{'first_parents':first.tolist(),'candidate_batches':pool.tolist(),'current_scores':current_scores,'stale_scores':stale_scores,'chosen_rank_order':chosen,'chosen_commit_order':{n:sorted(v) for n,v in chosen.items()},'shared_second_digest':shared_digest,'scoring_live_unchanged':True,'cost':{'stale':stale_cost,'current':current_cost}});dump(out/'scoring.json',scoring)
 models={};geometries={}
 for name in protocol['arms']:
  assert __import__('shutil').disk_usage(root).free>3*1024**3
  model,opt=clone_adam(live,optimizer);geo=geometry_live;accepted=first.tolist();assert digest_state(model,opt)==shared_digest
  for b in sorted(chosen[name]):
   before=len(model.dc);model,opt,geo,error=apply(model,opt,geo,pool[b]);accepted+=pool[b].tolist();events.append({'arm':name,'step':1140,'batch':b,'parents':pool[b].tolist(),'before_count':before,'after_count':len(model.dc),'optical_depth_error':error})
  assert len(accepted)==256 and len(set(accepted))==256 and len(model.dc)==10256
  for e,order in enumerate(orders[15:]):
   values=[update(model,opt,geo,i) for i in order];row={'arm':name,'epoch':e+1,'main_updates':1140+(e+1)*76,'gaussians':len(model.dc),'loss':float(np.mean(values))};training.append(row);print('POST',json.dumps(row),flush=True)
   with (out/'progress.json').open('w') as f:json.dump(row,f)
  assert all(torch.isfinite(p).all() for p in model.parameters())
  torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':opt.state_dict(),'main_steps':1520,'accepted_parents':accepted,'start_digest':shared_digest,'protocol':protocol},dest/(name+'-appearance.pt'));np.save(dest/(name+'-states.npy'),geo);models[name]=model;geometries[name]=geo
 assert digest_state(live,optimizer)==shared_digest;dump(out/'training.json',training);dump(out/'event-decisions.json',events);teacher.clear()
 ck=torch.load(source/'validation/continue-appearance.pt',map_location='cuda')['state_dict'];m=Appearance(ck['source_sh'],ck['source_logits'],len(ck['dc']));m.load_state_dict(ck);models['continue']=m;geometries['continue']=geometry
 def render(name,i,cam,after=True):
  if after:return image(models[name],geometries[name],i,cam)
  return render_state(cam,tensor(geometry[i]),initial_sh,torch.sigmoid(initial_logits),cfg.sh_degree)
 from evaluate_split_validation import evaluate
 evaluate(dest,base,out,protocol,times,fit,full,sh,opacity,cfg,scene,cameras,models,render,tensor)
 for p,h in inputs.items():assert digest(p)==h
 dump(out/'completion.json',{'status':'completed','source_sha256':digest(out/'source.zip'),'inputs_unchanged':True,'counts':{n:len(m.dc) for n,m in models.items()},'main_updates_each':1520,'main_gaussian_updates_each_split_arm':760*10000+380*10128+380*10256,'shared_scoring_cost':{'stale':stale_cost,'current':current_cost},'seconds':time.monotonic()-started});print('CURRENT STATE RERANKING COMPLETE',flush=True)
if __name__=='__main__':main()
