#!/usr/bin/env python3
"""E10: immediate/short-horizon gain versus matched long-horizon outcomes."""
import argparse,json,os,pathlib,time,zipfile
import numpy as np
import torch
from appearance_model import Appearance
from training_split_core import clone_adam,split_adam,digest_state,render_gate,RATES
from reduction_scene import load_scene,small_camera,render_state
from run_ctd_comparison import digest,dump

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args();root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();repo=root/'projects/dynamic-gaussian-ot-densification-research';out=a.output.resolve()
 assert out.is_relative_to(root) and not out.exists();protocol=json.loads((repo/'experiments/split-gain-prediction-v1.json').read_text());inp=root/'outputs/ot/e05-split-20261104T103723Z';original=json.loads((inp/'manifest.json').read_text());completed=json.loads((inp/'completion.json').read_text());assert completed['status']=='completed'
 assert (protocol['warmup_epochs']+protocol['post_epochs'])*76==protocol['main_updates_per_arm']
 inputs={pathlib.Path(p):h for p,h in original['inputs'].items()}
 for name in ['child-states.npy','children.npz','proposal-summary.json','scores.npz']:inputs[inp/name]=digest(inp/name)
 source=root/'outputs/ot/e06-training-event-20261104T113535Z'; source_manifest=json.loads((source/'manifest.json').read_text()); assert json.loads((source/'completion.json').read_text())['status']=='completed'
 for name in ['warmup.pt','manifest.json','validation/gated_window-appearance.pt','validation/gated_window-states.npy','validation/continue-appearance.pt']:inputs[source/name]=digest(source/name)
 e09=root/'outputs/ot/e09-current-state-reranking-20261104T133944Z'; assert json.loads((e09/'completion.json').read_text())['status']=='completed'
 for name in ['before-second.pt','selection.json','manifest.json']:inputs[e09/name]=digest(e09/name)
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
  for name in ['run_split_gain_prediction.py','reranking_core.py','split_proposal_core.py','test_reranking_core.py','test_repeated_split_adam.py','training_split_core.py','test_training_split_core.py','appearance_model.py','reduction_scene.py','evaluate_split_validation.py','run_ctd_comparison.py','reduction_core.py','inspect_baseline.py','common.sh']:z.write(repo/'scripts'/name,'scripts/'+name)
  z.write(repo/'experiments/split-gain-prediction-v1.json','experiments/split-gain-prediction-v1.json')
 from utils.loss_utils import ssim
 def image(model,geo,i,cam=None):
  colors,alpha=model(torch.arange(len(model.dc),device='cuda'));return render_state(cameras[i] if cam is None else cam,tensor(geo[i]),colors,alpha,cfg.sh_degree)
 teacher={}
 with torch.no_grad():
  for i in range(150):teacher[i]=render_state(cameras[i],tensor(full[i]),sh,opacity,cfg.sh_degree).detach()
 def loss(pred,target):return .8*(pred-target).abs().mean()+.2*(1-ssim(pred.unsqueeze(0),target.unsqueeze(0)))
 def update(model,opt,geo,i):
  opt.zero_grad(set_to_none=True);value=loss(image(model,geo,i),teacher[i]);assert torch.isfinite(value);value.backward();opt.step();return float(value.detach())
 @torch.no_grad()
 def decision_losses(model,geo):return [float(loss(image(model,geo,i),teacher[i])) for i in decision]
 import hashlib
 selected=json.loads((e09/'selection.json').read_text());first=selected['first_parents'];pool=selected['candidate_batches'];assert len(first)==128 and len(pool)==24
 geometry=np.concatenate([np.array(geometry,copy=True),child_geometry[:,np.array(first)*2+1]],1);geometry[:,first]=child_geometry[:,np.array(first)*2]
 shared=torch.load(e09/'before-second.pt',map_location='cuda');live=Appearance(shared['model']['source_sh'],shared['model']['source_logits'],10128);live.load_state_dict(shared['model']);optimizer=live.optimizer(RATES);optimizer.load_state_dict(shared['optimizer']);assert digest_state(live,optimizer)==shared['digest'] and shared['main_steps']==1140
 future=[i for order in orders[15:] for i in order];assert len(future)==380;dump(out/'design.json',{'first_parents':first,'candidate_batches':pool,'future_orders':orders[15:],'shared_digest':shared['digest'],'monitor_views':{'fit_decision':decision,'holdout':held}})
 measurements=[];training=[];started=time.monotonic();checkpoint_hashes={}
 @torch.no_grad()
 def monitor(name,model,geo,offset):
  row={'arm':name,'offset':offset,'fit_decision':[float(loss(image(model,geo,i),teacher[i])) for i in decision],'holdout':[float(loss(image(model,geo,i),teacher[i])) for i in held]};assert np.isfinite(row['fit_decision']+row['holdout']).all();measurements.append(row)
 for batch in [-1]+list(range(24)):
  assert __import__('shutil').disk_usage(root).free>3*1024**3
  name='reference' if batch<0 else f'batch_{batch:02d}';model,opt=clone_adam(live,optimizer);geo=geometry;parents=[];assert digest_state(model,opt)==shared['digest']
  if batch>=0:
   parents=pool[batch];ids=np.array(parents);fractions=mass[ids*2]/(mass[ids*2]+mass[ids*2+1]);model,opt,error=split_adam(model,opt,ids,fractions);geo=np.concatenate([np.array(geometry,copy=True),child_geometry[:,ids*2+1]],1);geo[:,ids]=child_geometry[:,ids*2]
  monitor(name,model,geo,0);values=[]
  for offset,i in enumerate(future,1):
   values.append(update(model,opt,geo,i))
   if offset in protocol['horizons']:monitor(name,model,geo,offset)
   if offset%76==0:
    row={'arm':name,'offset':offset,'main_steps':1140+offset,'gaussians':len(model.dc),'mean_epoch_loss':float(np.mean(values[-76:]))};training.append(row);print('TRAIN',json.dumps(row),flush=True)
    with (out/'progress.json').open('w') as f:json.dump(row,f)
  assert all(torch.isfinite(p).all() for p in model.parameters()) and digest_state(live,optimizer)==shared['digest']
  path=dest/(name+'-appearance.pt');torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':opt.state_dict(),'main_steps':1520,'added_parents':parents,'start_digest':shared['digest'],'geometry_sha256':hashlib.sha256(np.ascontiguousarray(geo).tobytes()).hexdigest()},path);checkpoint_hashes[path.name]=digest(path);del model,opt,geo
 dump(out/'measurements.json',measurements);dump(out/'training.json',training);dump(out/'training-complete.json',{'all25_frozen_before_GT':True,'checkpoint_hashes':checkpoint_hashes,'new_gradient_updates':25*380});teacher.clear()
 import lpips
 from PIL import Image
 from utils.image_utils import psnr
 criterion=lpips.LPIPS(net='alex',version='0.1').cuda().eval();gtrows=[];test=json.loads((root/'datasets/bouncingballs/transforms_test.json').read_text())['frames']
 with torch.no_grad():
  for batch in [-1]+list(range(24)):
   name='reference' if batch<0 else f'batch_{batch:02d}';ck=torch.load(dest/(name+'-appearance.pt'),map_location='cuda');m=Appearance(ck['state_dict']['source_sh'],ck['state_dict']['source_logits'],len(ck['state_dict']['dc']));m.load_state_dict(ck['state_dict']);geo=geometry
   if batch>=0:
    ids=np.array(pool[batch]);geo=np.concatenate([np.array(geometry,copy=True),child_geometry[:,ids*2+1]],1);geo[:,ids]=child_geometry[:,ids*2]
   for j,r in enumerate(test):
    i=int(np.argmin(abs(times-r['time'])));assert abs(times[i]-r['time'])<1e-7;cam=scene.getTestCameras()[j];gt=tensor(np.asarray(Image.open(base/f'test/ours_20000/gt/{j:05d}.png').convert('RGB'),dtype=np.float32)/255).permute(2,0,1).unsqueeze(0);pred=((image(m,geo,i,cam)*255).round()/255).unsqueeze(0)
    gtrows.append({'arm':name,'test_index':j,'PSNR':float(psnr(pred,gt)),'SSIM':float(ssim(pred,gt)),'LPIPS_alex_v01':float(criterion(pred,gt,normalize=True))})
   print('GT',name,flush=True);del m,ck,geo
 dump(dest/'gt-per-view.json',gtrows)
 for path,h in checkpoint_hashes.items():assert digest(dest/path)==h
 for p,h in inputs.items():assert digest(p)==h
 dump(out/'completion.json',{'status':'completed','inputs_unchanged':True,'source_sha256':digest(out/'source.zip'),'all25_frozen_before_GT':True,'new_gradient_updates':9500,'seconds':time.monotonic()-started});print('SPLIT GAIN PREDICTION COMPLETE',flush=True)
if __name__=='__main__':main()
