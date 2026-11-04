#!/usr/bin/env python3
"""Standard normalized LPIPS-Alex v0.1 over all saved test PNG pairs."""
import argparse,hashlib,json,os,pathlib
import numpy as np
from PIL import Image

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);ap.add_argument('--device',default='cuda');args=ap.parse_args()
    import torch,lpips,torchvision
    root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();model=args.model.resolve();output=args.output.resolve()
    if not model.is_relative_to(root) or not output.is_relative_to(root):raise ValueError('Paths outside owner volume')
    if output.exists():raise FileExistsError('Metrics output exists; refusing overwrite')
    weight=pathlib.Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth'
    if not weight.is_file():raise FileNotFoundError('Download official AlexNet weights first')
    h=hashlib.sha256()
    with weight.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    if not h.hexdigest().startswith('7be5be79'):raise ValueError('Official AlexNet checksum mismatch')
    torch.set_num_threads(2)
    net=lpips.LPIPS(net='alex',version='0.1').to(args.device).eval()
    method=model/'test/ours_20000';images=sorted((method/'renders').glob('*.png'));assert len(images)==20
    def tensor(p):return torch.from_numpy(np.asarray(Image.open(p).convert('RGB'),dtype=np.float32)/255).permute(2,0,1).unsqueeze(0).to(args.device)
    per=[]
    with torch.no_grad():
        identical=tensor(images[0]);identity=float(net(identical,identical,normalize=True).item());assert abs(identity)<1e-7
        for i,p in enumerate(images):
            score=float(net(tensor(p),tensor(method/'gt'/p.name),normalize=True).item())
            if not np.isfinite(score) or score<0:raise ValueError('Invalid LPIPS')
            per.append({'image':p.name,'LPIPS_alex_v01':score});print(i+1,'/20',score,flush=True)
    calibration=pathlib.Path(lpips.__file__).parent/'weights/v0.1/alex.pth'
    report={'model_path':str(model),'test_views':20,'LPIPS_alex_v01':float(np.mean([x['LPIPS_alex_v01'] for x in per])),
      'protocol':'lpips package 0.1.4; AlexNet; LPIPS v0.1; saved 8-bit RGB PNG pairs; [0,1] inputs with normalize=True mapping to [-1,1]; mean over views',
      'comparison_note':'Standard normalized protocol differs from pinned upstream metrics.py lpipsPyTorch, which passes [0,1] without explicit range normalization. Do not mix protocols.',
      'device':args.device,'torch':torch.__version__,'torchvision':torchvision.__version__,'backbone_sha256':h.hexdigest(),'calibration_sha256':hashlib.sha256(calibration.read_bytes()).hexdigest(),'identity_check':identity,'per_view':per}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(report,f,indent=2)
    print('Mean LPIPS-alex:',report['LPIPS_alex_v01'],flush=True)
if __name__=='__main__':main()
