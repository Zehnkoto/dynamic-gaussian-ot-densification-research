#!/usr/bin/env python3
"""Create separated camera/time inspections and paired test comparisons without retraining."""
import argparse, ast, hashlib, json, math, os, pathlib, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

def configuration(path):
    node=ast.parse(path.read_text(),mode='eval').body
    if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name) or node.func.id!='Namespace' or node.args:
        raise ValueError('Unsupported cfg_args')
    return argparse.Namespace(**{k.arg:ast.literal_eval(k.value) for k in node.keywords})

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=pathlib.Path,required=True)
    parser.add_argument('--output',type=pathlib.Path,required=True)
    parser.add_argument('--frames',type=int,default=240)
    parser.add_argument('--fps',type=int,default=30)
    parser.add_argument('--size',type=int,default=512)
    parser.add_argument('--camera-index',type=int,default=10)
    args=parser.parse_args()
    if args.frames<2 or args.fps<1 or args.size<128:parser.error('Invalid video parameters')
    root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve()
    model=args.model.resolve();out=args.output.resolve();repo=root/'projects/4DGaussians'
    if not model.is_relative_to(root) or not out.is_relative_to(root):raise ValueError('Paths must stay in owner data volume')
    if out.exists():raise FileExistsError('Inspection output exists; refusing overwrite')
    cfg=configuration(model/'cfg_args')
    if not pathlib.Path(cfg.source_path).resolve().is_relative_to(root):raise ValueError('Dataset outside data volume')
    if pathlib.Path(cfg.model_path).resolve()!=model:raise ValueError('Model path mismatch')
    sys.path.insert(0,str(repo))
    import torch
    from scene import Scene,GaussianModel
    from scene.cameras import Camera
    from gaussian_renderer import render
    from arguments import ModelParams,PipelineParams,ModelHiddenParams
    from matplotlib import cm
    ap=argparse.ArgumentParser();mp=ModelParams(ap);pp=PipelineParams(ap);hp=ModelHiddenParams(ap)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    out.mkdir(parents=True,exist_ok=False)
    np.random.seed(6666);torch.manual_seed(6666)
    gaussians=GaussianModel(cfg.sh_degree,hp.extract(cfg))
    scene=Scene(mp.extract(cfg),gaussians,load_iteration=20000,shuffle=False)
    base=scene.getTestCameras()[args.camera_index]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    def title(image,text):
        panel=Image.new('RGB',(image.width,image.height+40),'white');panel.paste(image,(0,40))
        ImageDraw.Draw(panel).text((8,10),text,fill='black',font=font)
        return np.asarray(panel)
    def camera(R,T,t):
        return Camera(0,np.array(R).copy(),np.array(T).copy(),base.FoVx,base.FoVy,
                      torch.zeros(3,args.size,args.size),None,'inspection',0,time=float(t),data_device='cuda')
    fixed=camera(base.R,base.T,0.)
    projection=fixed.full_proj_transform.clone()
    bg=torch.tensor([1.,1.,1.] if cfg.white_background else [0.,0.,0.],device='cuda')
    times=np.linspace(0.,1.,args.frames)
    manifest={'model_path':str(model),'source_path':cfg.source_path,'iteration':20000,'frames':args.frames,'fps':args.fps,'duration_seconds':args.frames/args.fps,
              'fixed_camera':{'source':'official test camera','index':args.camera_index,'R':base.R.tolist(),'T':base.T.tolist(),'FoVx':base.FoVx,'FoVy':base.FoVy,'times':times.tolist(),'ground_truth_available':False},
              'orbit':{'fixed_time':0.5,'ground_truth_available':False},'notes':['Synthesized motion and orbit videos are visual checks, not ground-truth validation.','Test comparison changes camera across samples; display is a slideshow, not a continuous trajectory.','Reference renders are the unmodified trained 4DGS baseline; no OT reduction applied.']}
    def frame(cam):
        image=render(cam,gaussians,pp.extract(cfg),bg,cam_type=scene.dataset_type)['render'].clamp(0,1)
        return Image.fromarray((image.permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8))
    with torch.no_grad():
        with imageio.get_writer(out/'fixed-camera-motion.mp4',fps=args.fps,codec='libx264',quality=8,macro_block_size=1) as writer:
            for i,t in enumerate(times):
                fixed.time=float(t);img=frame(fixed)
                writer.append_data(title(img,f'Baseline | fixed camera | t={t:.3f}'))
                if i in (0,args.frames//2,args.frames-1):img.save(out/f'fixed-frame-{i:04d}.png')
                if i%60==0:print('Fixed camera',i,'/',args.frames,flush=True)
        assert torch.equal(projection,fixed.full_proj_transform),'Fixed camera changed'
        center=base.camera_center.cpu().numpy();radius=float(np.linalg.norm(center));elevation=math.radians(25)
        manifest['orbit'].update(radius=radius,elevation_degrees=25.,target=[0.,0.,0.])
        with imageio.get_writer(out/'fixed-time-orbit.mp4',fps=args.fps,codec='libx264',quality=8,macro_block_size=1) as writer:
            for i,theta in enumerate(np.linspace(math.atan2(center[1],center[0]),math.atan2(center[1],center[0])+2*math.pi,args.frames,endpoint=False)):
                eye=radius*np.array([math.cos(elevation)*math.cos(theta),math.cos(elevation)*math.sin(theta),math.sin(elevation)])
                back=eye/np.linalg.norm(eye);right=np.cross([0.,0.,1.],back);right/=np.linalg.norm(right);up=np.cross(back,right)
                c2w=np.eye(4);c2w[:3,:3]=np.stack([right,up,back],axis=1);c2w[:3,3]=eye;c2w[:3,1:3]*=-1
                w2c=np.linalg.inv(c2w);cam=camera(w2c[:3,:3].T,w2c[:3,3],0.5)
                img=frame(cam);writer.append_data(title(img,'Baseline | camera orbit | fixed t=0.500'))
                if i==0:img.save(out/'orbit-first-frame.png')
                if i%60==0:print('Orbit',i,'/',args.frames,flush=True)
    paired=model/'test/ours_20000';render_files=sorted((paired/'renders').glob('*.png'))
    test=json.loads((pathlib.Path(cfg.source_path)/'transforms_test.json').read_text())['frames']
    assert len(render_files)==len(test)==20
    comparison=out/'test-comparisons';comparison.mkdir()
    cmap=cm.get_cmap('inferno')
    with imageio.get_writer(out/'test-comparison-slideshow.mp4',fps=10,codec='libx264',quality=8,macro_block_size=1) as writer:
        for i,p in enumerate(render_files):
            pred=Image.open(p).convert('RGB');gt=Image.open(paired/'gt'/p.name).convert('RGB')
            delta=np.abs(np.asarray(pred,dtype=np.float32)/255-np.asarray(gt,dtype=np.float32)/255).mean(axis=2)
            heat=Image.fromarray((cmap(np.clip(delta*20,0,1))[...,:3]*255).astype(np.uint8))
            panel=Image.new('RGB',(1200,460),'white');draw=ImageDraw.Draw(panel)
            draw.text((10,8),f'Test {i:02d} | t={test[i]["time"]:.3f} | independent camera per sample',font=font,fill='black')
            for x,img,label in [(0,gt,'Ground truth'),(400,pred,'Baseline reconstruction'),(800,heat,'Mean absolute RGB error x20 (clipped)')]:
                draw.text((x+8,34),label,font=small,fill='black');panel.paste(img.resize((400,400),Image.Resampling.LANCZOS),(x,60))
            panel.save(comparison/(p.stem+'.jpg'),quality=93)
            for _ in range(10):writer.append_data(np.asarray(panel))
    manifest['comparison']={'count':20,'seconds_per_image':1,'error_scale':20,'error_definition':'mean absolute RGB difference in [0,1], clipped after scale','source_render_directory':str(paired)}
    manifest['artifacts']={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in out.glob('*.mp4')}
    (out/'inspection-manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Inspection complete:',out,flush=True)
if __name__=='__main__':main()
