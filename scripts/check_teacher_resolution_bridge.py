#!/usr/bin/env python3
"""Post-hoc frozen-teacher native400 versus area-downsampled800 comparison."""
import argparse,json,os,pathlib,zipfile
import numpy as np
import torch
import torch.nn.functional as F
from reduction_scene import load_scene,small_camera,render_state
from run_ctd_comparison import dump,digest
ap=argparse.ArgumentParser();ap.add_argument('output',type=pathlib.Path);out=ap.parse_args().output.resolve();root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();assert out.is_relative_to(root);read=lambda p:json.loads(p.read_text());m=read(out/'manifest.json');assert read(out/'completion.json')['status']=='completed';base=root/'outputs/dnerf/bouncingballs/baseline-numpy1235-20261030T174401Z';cfg,pc,scene,pipe=load_scene(base);pm=read(root/'outputs/ot/e05-split-20261104T103723Z/manifest.json');cache=pathlib.Path(pm['source_cache']);states=np.load(cache/'states.npy',mmap_mode='r')
def tensor(x):return torch.tensor(np.array(x,copy=True),device='cuda')
sh=tensor(np.load(cache/'sh.npy'));opacity=tensor(np.load(cache/'opacity.npy'));train=scene.getTrainCameras();test=scene.getTestCameras();rows=[]
with torch.no_grad():
 for condition in ['fit400','train400','test400']:
  for view,r in enumerate(m['conditions'][condition]):
   cam=(train if r['camera_split']=='train' else test)[r['frame_index']];gt800=cam.original_image.cuda();gt400=F.interpolate(gt800.unsqueeze(0),size=(400,400),mode='area').squeeze(0);features=tensor(states[r['time_index']]);native=render_state(small_camera(cam,400),features,sh,opacity,cfg.sh_degree);high=render_state(small_camera(cam,800),features,sh,opacity,cfg.sh_degree);area=F.interpolate(high.unsqueeze(0),size=(400,400),mode='area').squeeze(0)
   row={'condition':condition,'view':view,'native400_GT400_MSE':float((native-gt400).square().mean()),'areaTeacher800_GT400_MSE':float((area-gt400).square().mean()),'native400_areaTeacher800_MSE':float((native-area).square().mean()),'native800_GT800_MSE':float((high-gt800).square().mean())};assert np.isfinite(list(row.values())[2:]).all();assert row['areaTeacher800_GT400_MSE']<=row['native800_GT800_MSE']+1e-7;rows.append(row)
summary={condition:{k:float(np.mean([r[k] for r in rows if r['condition']==condition])) for k in ['native400_GT400_MSE','areaTeacher800_GT400_MSE','native400_areaTeacher800_MSE','native800_GT800_MSE']} for condition in ['fit400','train400','test400']}
for p,h in m['inputs'].items():assert digest(pathlib.Path(p))==h
report={'status':'post-hoc measurement prompted by E11 resolution difference, not primary predeclared analysis','training_updates':0,'summary':summary,'rows':rows,'input_manifest_sha256':digest(out/'manifest.json'),'notes':['All source/model/data hashes unchanged','Matched camera and white-composited real GT; same area800to400 operator for teacher and GT','No student targets changed or models retrained; lower teacher-to-GT MSE does not establish student quality improvement']};dump(out/'analysis/teacher-resolution-bridge.json',report)
with zipfile.ZipFile(out/'analysis/teacher-resolution-bridge-source.zip','x') as z:z.write(pathlib.Path(__file__).resolve(),'check_teacher_resolution_bridge.py')
print(json.dumps({'summary':summary,'training_updates':0},indent=2))
