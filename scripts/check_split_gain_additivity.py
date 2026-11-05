#!/usr/bin/env python3
"""Post-hoc read-only E09 joint-loss check against frozen E10 single-batch outcomes."""
import argparse,csv,json,os,pathlib,zipfile
import numpy as np
import torch
from appearance_model import Appearance
from reduction_scene import load_scene,small_camera,render_state
from run_ctd_comparison import dump,digest
ap=argparse.ArgumentParser();ap.add_argument('output',type=pathlib.Path);out=ap.parse_args().output.resolve();root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();assert out.is_relative_to(root)
read=lambda p:json.loads(p.read_text());assert read(out/'completion.json')['status']=='completed';e09=root/'outputs/ot/e09-current-state-reranking-20261104T133944Z';m=read(out/'manifest.json');selection=read(e09/'selection.json');design=read(out/'design.json');assert design['shared_digest']==selection['shared_second_digest'] and design['future_orders']==read(e09/'manifest.json')['orders'][15:]
with (out/'analysis/candidate-outcomes.csv').open() as f:single={int(r['batch']):r for r in csv.DictReader(f)}
reference=read(out/'analysis/report.json')['reference']['horizon_mean_losses']['380'];base=root/'outputs/dnerf/bouncingballs/baseline-numpy1235-20261030T174401Z';cfg,pc,scene,pipe=load_scene(base);pm=read(root/'outputs/ot/e05-split-20261104T103723Z/manifest.json');cache=pathlib.Path(pm['source_cache']);full=np.load(cache/'states.npy',mmap_mode='r');times=np.load(cache/'times.npy')
def tensor(x):return torch.tensor(np.array(x,copy=True),device='cuda')
sh=tensor(np.load(cache/'sh.npy'));opacity=tensor(np.load(cache/'opacity.npy'));views={'fit_decision':design['monitor_views']['fit_decision'],'holdout':design['monitor_views']['holdout']};cameras={};frames=read(root/'datasets/bouncingballs/transforms_train.json')['frames'];train=scene.getTrainCameras()
for j,r in enumerate(frames):cameras[int(np.argmin(abs(times-r['time'])))]=small_camera(train[j],400)
from utils.loss_utils import ssim
inputs={};rows=[]
with torch.no_grad():
 teacher={i:render_state(cameras[i],tensor(full[i]),sh,opacity,cfg.sh_degree) for i in views['fit_decision']+views['holdout']}
 for name,chosen in selection['chosen_commit_order'].items():
  checkpoint=e09/'validation'/(name+'-appearance.pt');geometry=e09/'validation'/(name+'-states.npy');inputs[str(checkpoint)]=digest(checkpoint);inputs[str(geometry)]=digest(geometry);ck=torch.load(checkpoint,map_location='cuda');assert ck['start_digest']==design['shared_digest'];state=ck['state_dict'];model=Appearance(state['source_sh'],state['source_logits'],10256);model.load_state_dict(state);geo=np.load(geometry,mmap_mode='r');colors,alpha=model(torch.arange(10256,device='cuda'))
  for key,indices in views.items():
   losses=[]
   for i in indices:
    pred=render_state(cameras[i],tensor(geo[i]),colors,alpha,cfg.sh_degree);target=teacher[i];losses.append(float(.8*(pred-target).abs().mean()+.2*(1-ssim(pred.unsqueeze(0),target.unsqueeze(0)))))
   actual_gain=reference[key]-float(np.mean(losses));summed=float(sum(float(single[b]['fit_loss_gain_380' if key=='fit_decision' else 'holdout_loss_gain_380']) for b in chosen));rows.append({'arm':name,'views':key,'selected_batches':chosen,'actual_view_losses':losses,'actual_joint_gain':actual_gain,'sum_of_independent_batch_gains':summed,'joint_minus_sum':actual_gain-summed})
for p,h in inputs.items():assert digest(pathlib.Path(p))==h
dest=out/'analysis/joint-additivity.json';report={'status':'post-hoc diagnostic, not predeclared E10 primary analysis','rows':rows,'inputs':inputs,'notes':['No training, new selections or thresholds; E09 selected sets remain frozen','Uses exactly same shared1140 state and380-update orders as E10','Loss is nonlinear and rendering has occlusion interactions; deviation includes these effects and numerical training variation','No additive assumption for logarithmic GT PSNR; only teacher render loss checked']};dump(dest,report)
with zipfile.ZipFile(out/'analysis/joint-additivity-source.zip','x') as z:z.write(pathlib.Path(__file__).resolve(),'check_split_gain_additivity.py')
print(json.dumps({**report,'inputs':'hashes retained in data-volume report'},indent=2))
