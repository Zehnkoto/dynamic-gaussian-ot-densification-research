#!/usr/bin/env python3
"""E12: paired teacher-target and student-renderer controls with fixedCTD choices."""
import argparse,json,os,pathlib,time,zipfile
import numpy as np
import torch
from appearance_model import Appearance
from training_split_core import clone_adam,split_adam,digest_state,render_gate,RATES
from reduction_scene import load_scene,small_camera,render_state
from run_ctd_comparison import digest,dump

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args();root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();repo=root/'projects/dynamic-gaussian-ot-densification-research';out=a.output.resolve()
 assert out.is_relative_to(root) and not out.exists();protocol=json.loads((repo/'experiments/teacher-rendering-control-v1.json').read_text());inp=root/'outputs/ot/e05-split-20261104T103723Z';original=json.loads((inp/'manifest.json').read_text());completed=json.loads((inp/'completion.json').read_text());assert completed['status']=='completed'
 assert (protocol['warmup_epochs']+protocol['post_epochs'])*76==protocol['main_updates_per_arm']
 inputs={pathlib.Path(p):h for p,h in original['inputs'].items()}
 for name in ['child-states.npy','children.npz','proposal-summary.json','scores.npz']:inputs[inp/name]=digest(inp/name)
 source=root/'outputs/ot/e06-training-event-20261104T113535Z'; source_manifest=json.loads((source/'manifest.json').read_text()); assert json.loads((source/'completion.json').read_text())['status']=='completed'
 inputs[source/'manifest.json']=digest(source/'manifest.json')
 e09=root/'outputs/ot/e09-current-state-reranking-20261104T133944Z'
 for name in ['before-second.pt','manifest.json','selection.json','validation/summary.json']:inputs[e09/name]=digest(e09/name)
 dataset=root/'datasets/bouncingballs'
 for name in ['transforms_train.json','transforms_test.json']:
  inputs[dataset/name]=digest(dataset/name)
  if name=='transforms_train.json':
   for frame in json.loads((dataset/name).read_text())['frames']:
    png=dataset/(frame['file_path']+'.png');inputs[png]=digest(png)
 for j in range(20):
  png=root/f'outputs/dnerf/bouncingballs/baseline-numpy1235-20261030T174401Z/test/ours_20000/gt/{j:05d}.png';inputs[png]=digest(png)
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
 assert orders==source_manifest['orders'] and decision==source_manifest['decision_views']
 cameras={};frames=json.loads((root/'datasets/bouncingballs/transforms_train.json').read_text())['frames'];train=scene.getTrainCameras()
 for j,r in enumerate(frames):
  i=int(np.argmin(abs(times-r['time'])));cam=train[j];assert abs(float(cam.time)-r['time'])<1e-7;cameras[i]=small_camera(cam,protocol['training_size'])
 out.mkdir(parents=True);dest=out/'validation';dest.mkdir();dump(out/'manifest.json',{'protocol':protocol,'inputs':{str(p):h for p,h in inputs.items()},'fit':fit,'holdout':held,'decision_views':decision,'decision_window_ids':decision_bins,'probe_training_views':probe_fit,'orders':orders,'GPU':os.environ.get('CUDA_VISIBLE_DEVICES')});dump(out/'proposal-summary.json',proposals)
 with zipfile.ZipFile(out/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
  for name in ['run_teacher_rendering_control.py','reranking_core.py','split_proposal_core.py','test_reranking_core.py','test_repeated_split_adam.py','training_split_core.py','test_training_split_core.py','appearance_model.py','reduction_scene.py','evaluate_teacher_rendering_control.py','run_ctd_comparison.py','reduction_core.py','inspect_baseline.py','common.sh']:z.write(repo/'scripts'/name,'scripts/'+name)
  z.write(repo/'experiments/teacher-rendering-control-v1.json','experiments/teacher-rendering-control-v1.json')
 import hashlib,torch.nn.functional as F
 from utils.loss_utils import ssim
 read=lambda p:json.loads(p.read_text());selection=read(e09/'selection.json');first=selection['first_parents'];second=np.array(selection['candidate_batches'])[selection['chosen_commit_order']['fixed_ctd']].reshape(-1).tolist();assert len(first)==len(second)==128 and not set(first)&set(second)
 geometry_no=np.concatenate([np.array(geometry,copy=True),child_geometry[:,np.array(first)*2+1]],1);geometry_no[:,first]=child_geometry[:,np.array(first)*2]
 shared=torch.load(e09/'before-second.pt',map_location='cuda');live=Appearance(shared['model']['source_sh'],shared['model']['source_logits'],10128);live.load_state_dict(shared['model']);optimizer=live.optimizer(RATES);optimizer.load_state_dict(shared['optimizer']);assert digest_state(live,optimizer)==shared['digest']==selection['shared_second_digest']
 geometries={'no_second':geometry_no};initial={'no_second':(live,optimizer)}
 m,o=clone_adam(live,optimizer);ids=np.array(second);m,o,error=split_adam(m,o,ids,mass[ids*2]/(mass[ids*2]+mass[ids*2+1]));g=np.concatenate([np.array(geometry_no,copy=True),child_geometry[:,ids*2+1]],1);g[:,ids]=child_geometry[:,ids*2];geometries['ctd128']=g;initial['ctd128']=(m,o)
 starts={cap:digest_state(*initial[cap]) for cap in initial}
 future_orders={str(seed):(source_manifest['orders'][15:] if seed==42 else [np.random.default_rng(seq).permutation(fit).tolist() for seq in np.random.SeedSequence(seed).spawn(5)]) for seed in protocol['order_seeds']}
 assert all(len(v)==5 and all(sorted(order)==sorted(fit) for order in v) for v in future_orders.values());assert len({tuple(order) for v in future_orders.values() for order in v})==15
 teacher={'native400':{},'area800':{}};high_cameras={i:small_camera(cameras[i],800) for i in cameras};real400={}
 for j,r in enumerate(frames):
  i=int(np.argmin(abs(times-r['time'])));real400[i]=F.interpolate(train[j].original_image.cuda().unsqueeze(0),size=(400,400),mode='area').squeeze(0)
 def area(image):return F.interpolate(image.unsqueeze(0),size=(400,400),mode='area').squeeze(0)
 with torch.no_grad():
  for i in range(150):
   teacher['native400'][i]=render_state(cameras[i],tensor(full[i]),sh,opacity,cfg.sh_degree).detach();teacher['area800'][i]=area(render_state(high_cameras[i],tensor(full[i]),sh,opacity,cfg.sh_degree)).detach()
 def target_hash(values):
  h=hashlib.sha256()
  for i in sorted(values):h.update(values[i].detach().cpu().contiguous().numpy().tobytes())
  return h.hexdigest()
 target_hashes={n:target_hash(v) for n,v in teacher.items()}
 def image(model,geo,i,operator):
  colors,alpha=model(torch.arange(len(model.dc),device='cuda'));pred=render_state(high_cameras[i] if operator=='area800' else cameras[i],tensor(geo[i]),colors,alpha,cfg.sh_degree);return area(pred) if operator=='area800' else pred
 def loss(p,t):return .8*(p-t).abs().mean()+.2*(1-ssim(p.unsqueeze(0),t.unsqueeze(0)))
 dump(out/'design.json',{'first_parents':first,'second_parents':second,'future_orders':future_orders,'initial_digests':starts,'shared_digest':shared['digest'],'teacher_target_hashes':target_hashes})
 for cap in geometries:np.save(out/('geometry-'+cap+'.npy'),geometries[cap])
 models={};model_geometry={};training=[];model_info={};hashes={};started=time.monotonic()
 for seed in protocol['order_seeds']:
  for cap in protocol['capacities']:
   for condition,recipe in protocol['conditions'].items():
    assert __import__('shutil').disk_usage(root).free>3*1024**3
    name=f'{condition}__{cap}__seed{seed}';model,opt=clone_adam(*initial[cap]);geo=geometries[cap];assert digest_state(model,opt)==starts[cap];begin=time.monotonic()
    for e,order in enumerate(future_orders[str(seed)]):
     values=[]
     for i in order:
      opt.zero_grad(set_to_none=True);value=loss(image(model,geo,i,recipe['student']),teacher[recipe['target']][i]);assert torch.isfinite(value);value.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters());opt.step();values.append(float(value.detach()))
     row={'arm':name,'epoch':e+1,'main_steps':1140+(e+1)*76,'gaussians':len(model.dc),'loss':float(np.mean(values))};training.append(row);print('TRAIN',json.dumps(row),flush=True)
     with (out/'progress.json').open('w') as f:json.dump(row,f)
    assert all(torch.isfinite(p).all() for p in model.parameters());info={'seed':seed,'capacity':cap,'condition':condition,'initial_digest':starts[cap],'student_operator':recipe['student'],'teacher_operator':recipe['target'],'main_steps':1520,'new_updates':380,'train_seconds':time.monotonic()-begin};model_info[name]=info
    path=dest/(name+'-appearance.pt');torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':opt.state_dict(),'info':info},path);hashes[path.name]=digest(path);models[name]=model;model_geometry[name]=geo
 assert all(digest_state(*initial[cap])==starts[cap] for cap in initial);assert {n:target_hash(v) for n,v in teacher.items()}==target_hashes
 dump(out/'training.json',training);dump(out/'training-complete.json',{'all18_frozen_before_evaluation':True,'checkpoint_hashes':hashes,'model_info':model_info,'new_gradient_updates':18*380})
 from evaluate_teacher_rendering_control import evaluate
 evaluate(dest,base,protocol,times,fit,held,full,sh,opacity,cfg,scene,cameras,models,model_geometry,teacher,real400,tensor)
 for path,h in hashes.items():assert digest(dest/path)==h
 for p,h in inputs.items():assert digest(p)==h
 dump(out/'completion.json',{'status':'completed','inputs_unchanged':True,'source_sha256':digest(out/'source.zip'),'all18_frozen_before_evaluation':True,'main_steps_each':1520,'new_updates_each':380,'new_gradient_updates':6840,'counts':{n:len(m.dc) for n,m in models.items()},'seconds':time.monotonic()-started});print('TEACHER RENDERING CONTROL COMPLETE',flush=True)
if __name__=='__main__':main()
