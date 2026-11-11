#!/usr/bin/env python3
"""Standard E19 8-bit GT metrics for the frozen official jumpingjacks baseline."""
import argparse,hashlib,json,os,pathlib,sys
import numpy as np
import torch
from PIL import Image

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',type=pathlib.Path,required=True);args=ap.parse_args()
 root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();model=args.model.resolve();assert model.is_relative_to(root/'outputs/dnerf/jumpingjacks')
 out=model/'baseline-validation.json';assert not out.exists()
 manifest=json.loads((model/'run-manifest.json').read_text());assert manifest['phase']=='completed'
 sys.path.insert(0,str(root/'projects/4DGaussians'))
 from utils.loss_utils import ssim
 from utils.image_utils import psnr
 import lpips
 net=lpips.LPIPS(net='alex',version='0.1').cuda().eval().requires_grad_(False)
 directory=model/'test/ours_20000'
 def tensor(p):return torch.tensor(np.asarray(Image.open(p).convert('RGB'),dtype=np.float32)/255,device='cuda').permute(2,0,1).unsqueeze(0).contiguous()
 rows=[]
 with torch.no_grad():
  for j in range(20):
   name=f'{j:05d}.png';p=tensor(directory/'renders'/name);g=tensor(directory/'gt'/name)
   assert p.shape==g.shape and p.shape[-1]==800 and p.shape[-2]==800
   rows.append({'view':j,'PSNR':float(psnr(p,g)),'SSIM':float(ssim(p,g)),'LPIPS_alex_v01':float(net(p,g,normalize=True))})
 model_files={str(p.relative_to(model)):{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted((model/'point_cloud/iteration_20000').glob('*')) if p.is_file()}
 assert set(pathlib.Path(n).name for n in model_files)=={'point_cloud.ply','deformation.pth','deformation_table.pth','deformation_accum.pth'}
 from plyfile import PlyData
 count=len(PlyData.read(model/'point_cloud/iteration_20000/point_cloud.ply')['vertex'])
 report={'scene':'jumpingjacks','views':20,'teacher_gaussians':count,'model_files':model_files,'dataset_manifest_sha256':manifest['dataset_manifest_sha256'],'metrics':{m:float(np.mean([r[m] for r in rows])) for m in ['PSNR','SSIM','LPIPS_alex_v01']},'per_view':rows,'metric_protocol':'saved 8-bit 800x800 RGB, PSNR and SSIM from pinned4DGS utils, LPIPS package Alex v0.1 normalize=True'}
 with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps(report['metrics'],indent=2),flush=True)
if __name__=='__main__':main()
