#!/usr/bin/env python3
"""E13: bounded shared-position optimization with exact fixed-plan CTD."""
import argparse,json,os,pathlib,time,zipfile
import numpy as np
import torch
from appearance_model import Appearance
from training_split_core import clone_adam,split_adam,digest_state,render_gate,RATES
from reduction_scene import load_scene,small_camera,render_state
from run_ctd_comparison import digest,dump

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args();root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();repo=root/'projects/dynamic-gaussian-ot-densification-research';out=a.output.resolve()
 assert out.is_relative_to(root) and not out.exists();protocol=json.loads((repo/'experiments/shared-position-ctd-v1.json').read_text());inp=root/'outputs/ot/e05-split-20261104T103723Z';original=json.loads((inp/'manifest.json').read_text());completed=json.loads((inp/'completion.json').read_text());assert completed['status']=='completed'
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
 e12_input=root/'outputs/ot/e12-teacher-rendering-control-20261104T163644Z'
 for name in ['design.json','validation/summary.json']:inputs[e12_input/name]=digest(e12_input/name)
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
  for name in ['run_shared_position_ctd.py','position_ctd_core.py','test_position_ctd_core.py','reranking_core.py','split_proposal_core.py','test_reranking_core.py','test_repeated_split_adam.py','training_split_core.py','test_training_split_core.py','appearance_model.py','reduction_scene.py','evaluate_shared_position_ctd.py','run_ctd_comparison.py','reduction_core.py','inspect_baseline.py','common.sh']:z.write(repo/'scripts'/name,'scripts/'+name)
  z.write(repo/'experiments/shared-position-ctd-v1.json','experiments/shared-position-ctd-v1.json')
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
 from position_ctd_core import source_partition,fixed_plan_stats,SharedTranslation,apply_translation
 from evaluate_shared_position_ctd import evaluate
 e12=root/'outputs/ot/e12-teacher-rendering-control-20261104T163644Z';e12design=read(e12/'design.json');assert future_orders==e12design['future_orders'] and starts['ctd128']==e12design['initial_digests']['ctd128']
 grouping=root/'outputs/ot/e02-ctd-refined-20261103T082856Z/budget-10000/shared-grouping.npz';labels=source_partition(np.load(grouping)['labels'],child['source_child'],first+second);weights=np.load(cache/'opacity.npy').astype(np.float64).reshape(-1);weights/=weights.sum();geo=geometries['ctd128'];stats=fixed_plan_stats(full,geo,labels,weights,fit,held);assert stats['fit_base']>0
 np.savez(out/'ctd-statistics.npz',**stats,labels=labels);np.save(out/'base-geometry.npy',geo)
 targets={'native400':{},'area800':{}};real400={}
 for j,r in enumerate(frames):
  i=int(np.argmin(abs(times-r['time'])));real400[i]=F.interpolate(train[j].original_image.cuda().unsqueeze(0),size=(400,400),mode='area').squeeze(0)
 with torch.no_grad():
  for i in range(150):
   targets['native400'][i]=render_state(cameras[i],tensor(full[i]),sh,opacity,cfg.sh_degree).detach();high=render_state(small_camera(cameras[i],800),tensor(full[i]),sh,opacity,cfg.sh_degree);targets['area800'][i]=F.interpolate(high.unsqueeze(0),size=(400,400),mode='area').squeeze(0).detach()
 def target_hash(values):
  h=hashlib.sha256()
  for i in sorted(values):h.update(values[i].cpu().contiguous().numpy().tobytes())
  return h.hexdigest()
 target_hashes={n:target_hash(v) for n,v in targets.items()};mass_tensor=tensor(stats['mass']);linear_tensor=tensor(stats['fit_linear']);start=starts['ctd128']
 dump(out/'design.json',{'first_parents':first,'second_parents':second,'future_orders':future_orders,'initial_appearance_digest':start,'shared1140_digest':shared['digest'],'target_hashes':target_hashes,'radius':protocol['position_radius'],'fit_CTD_base':stats['fit_base'],'held_CTD_base':stats['held_base']})
 print('CTD BASE',stats['fit_base'],stats['held_base'],flush=True)
 models={};offsets={};model_geometry={};training=[];infos={};hashes={};started=time.monotonic()
 for seed in protocol['order_seeds']:
  for condition,recipe in protocol['conditions'].items():
   assert __import__('shutil').disk_usage(root).free>3*1024**3
   name=f'{condition}__seed{seed}';model,opt=clone_adam(*initial['ctd128']);assert digest_state(model,opt)==start;field=SharedTranslation(mass_tensor,linear_tensor,stats['fit_base'],protocol['position_radius']);field.raw.requires_grad_(recipe['move']);gopt=torch.optim.Adam([field.raw],lr=protocol['position_raw_lr'],eps=1e-8) if recipe['move'] else None
   for e,order in enumerate(future_orders[str(seed)]):
    render_values=[];total_values=[]
    for i in order:
     opt.zero_grad(set_to_none=True)
     if gopt is not None:gopt.zero_grad(set_to_none=True)
     colors,alpha=model(torch.arange(len(model.dc),device='cuda'));features=apply_translation(tensor(geo[i]),field.offset());pred=render_state(cameras[i],features,colors,alpha,cfg.sh_degree);target=targets['native400'][i];render_loss=.8*(pred-target).abs().mean()+.2*(1-ssim(pred.unsqueeze(0),target.unsqueeze(0)));total=render_loss+recipe['ctd_weight']*field.relative_excess();assert torch.isfinite(total);total.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters());opt.step()
     if gopt is not None:
      assert field.raw.grad is not None and torch.isfinite(field.raw.grad).all();gopt.step()
     render_values.append(float(render_loss.detach()));total_values.append(float(total.detach()))
    row={'arm':name,'epoch':e+1,'main_steps':1140+(e+1)*76,'render_loss':float(np.mean(render_values)),'total_loss':float(np.mean(total_values)),'relative_CTD_end':float(field.relative_excess().detach()),'max_offset':float(field.offset().detach().norm(dim=1).max())};training.append(row);print('TRAIN',json.dumps(row),flush=True)
    with (out/'progress.json').open('w') as f:json.dump(row,f)
   assert all(torch.isfinite(p).all() for p in model.parameters()) and torch.isfinite(field.raw).all();assert float(field.offset().detach().norm(dim=1).max())<=protocol['position_radius']+1e-7
   info={'seed':seed,'condition':condition,'ctd_weight':recipe['ctd_weight'],'moving':recipe['move'],'radius':protocol['position_radius'],'initial_digest':start,'appearance_steps':1520,'geometry_steps':380 if recipe['move'] else 0};infos[name]=info
   path=dest/(name+'-state.pt');torch.save({'appearance':{k:v.detach().cpu() for k,v in model.state_dict().items()},'appearance_optimizer':opt.state_dict(),'position':{k:v.detach().cpu() for k,v in field.state_dict().items()},'position_optimizer':gopt.state_dict() if gopt is not None else None,'info':info},path);hashes[path.name]=digest(path);models[name]=model;offsets[name]=field.offset().detach();model_geometry[name]=geo
 assert digest_state(*initial['ctd128'])==start and {n:target_hash(v) for n,v in targets.items()}==target_hashes
 dump(out/'training.json',training);dump(out/'training-complete.json',{'all15_frozen_before_evaluation':True,'checkpoint_hashes':hashes,'model_info':infos,'new_gradient_updates':15*380})
 evaluate(dest,base,protocol,times,fit,held,full,sh,opacity,cfg,scene,cameras,models,model_geometry,targets,real400,tensor,offsets)
 for path,h in hashes.items():assert digest(dest/path)==h
 for p,h in inputs.items():assert digest(p)==h
 dump(out/'completion.json',{'status':'completed','input_hashes_unchanged':True,'source_sha256':digest(out/'source.zip'),'all15_frozen_before_evaluation':True,'new_gradient_updates':5700,'counts':{n:len(m.dc) for n,m in models.items()},'seconds':time.monotonic()-started});print('SHARED POSITION CTD COMPLETE',flush=True)
if __name__=='__main__':main()
