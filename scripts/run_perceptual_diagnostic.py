#!/usr/bin/env python3
"""E17: read-only spatial diagnosis of E16's PSNR/LPIPS disagreement."""
import argparse,json,os,pathlib,zipfile,shutil,time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from appearance_model import Appearance
from covariance_moment_core import add_scatter
from reduction_scene import load_scene,render_state
from run_ctd_comparison import digest,dump

def masks(gt):
    gray=.299*gt[0]+.587*gt[1]+.114*gt[2]
    gx=torch.zeros_like(gray);gy=torch.zeros_like(gray)
    gx[:,:-1]=(gray[:,1:]-gray[:,:-1]).abs()
    gy[:-1,:]=(gray[1:,:]-gray[:-1,:]).abs()
    gradient=gx+gy
    foreground=(gt<.985).any(0)
    indices=torch.nonzero(foreground.reshape(-1),as_tuple=False).flatten()
    count=max(1,int(len(indices)*.20))
    top=torch.topk(gradient.reshape(-1)[indices],count,largest=True).indices
    edge=torch.zeros_like(foreground).reshape(-1);edge[indices[top]]=True;edge=edge.reshape_as(foreground)
    return gray,gradient,{'edge':edge,'other_foreground':foreground&(~edge),'background':~foreground}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=pathlib.Path);a=p.parse_args()
    root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();repo=root/'projects/dynamic-gaussian-ot-densification-research';out=a.output.resolve()
    if not out.is_relative_to(root) or out.exists():raise ValueError('Unsafe or existing output')
    assert shutil.disk_usage(root).free>3*1024**3
    protocol_path=repo/'experiments/perceptual-diagnostic-v1.json';protocol=json.loads(protocol_path.read_text())
    e16=root/'outputs/ot/e16-covariance-moment-diagnostic-20261110-v1'
    cache=root/'outputs/ot/e02-ctd-20261103T082303Z/trajectory-cache'
    prior=root/'outputs/ot/e02-ctd-refined-20261103T082856Z/budget-10000'
    base=root/'outputs/dnerf/bouncingballs/baseline-numpy1235-20261030T174401Z'
    e16manifest=json.loads((e16/'manifest.json').read_text());e16report=json.loads((e16/'report.json').read_text())
    assert json.loads((e16/'completion.json').read_text())['status']=='completed'
    names={'mean':e16/'mean_covariance__seed42-appearance.pt','moment':e16/'moment_covariance__seed42-appearance.pt'}
    paths=[protocol_path,e16/'manifest.json',e16/'report.json',e16/'audit.json',e16/'completion.json',e16/'training-complete.json',e16/'center-scatter.npy',
           cache/'states.npy',cache/'times.npy',cache/'sh.npy',cache/'opacity.npy',prior/'shared-states.npy',prior/'shared-grouping.npz',
           root/'datasets/bouncingballs/transforms_test.json',*names.values()]
    paths += [base/f'test/ours_20000/gt/{j:05d}.png' for j in range(20)]
    inputs={q:digest(q) for q in paths}
    oldhashes=e16manifest['inputs']
    for p,h in oldhashes.items():assert digest(pathlib.Path(p))==h,p
    for name,path in names.items():
        assert inputs[path]==json.loads((e16/'training-complete.json').read_text())['checkpoint_hashes'][path.name]
        assert path.stat().st_mtime_ns<(e16/'gt-per-view.json').stat().st_mtime_ns
    out.mkdir(parents=True)
    dump(out/'manifest.json',{'protocol':protocol,'inputs':{str(p):h for p,h in inputs.items()},'source_E16_inputs':oldhashes,'GPU':os.environ.get('CUDA_VISIBLE_DEVICES')})
    with zipfile.ZipFile(out/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for q in ['run_perceptual_diagnostic.py','appearance_model.py','covariance_moment_core.py','reduction_scene.py','inspect_baseline.py','run_ctd_comparison.py','common.sh']:
            z.write(repo/'scripts'/q,'scripts/'+q)
        z.write(protocol_path,'experiments/'+protocol_path.name)
    cfg,pc,scene,pipe=load_scene(base)
    for item in vars(pc).values():
        if isinstance(item,torch.nn.Parameter):item.requires_grad_(False)
        elif isinstance(item,torch.nn.Module):item.requires_grad_(False);item.eval()
    def tensor(x):return torch.tensor(np.array(x,copy=True),device='cuda')
    sh=tensor(np.load(cache/'sh.npy'));opacity=tensor(np.load(cache/'opacity.npy'))
    assert torch.equal(sh,pc.get_features) and torch.equal(opacity,pc.get_opacity)
    full=np.load(cache/'states.npy',mmap_mode='r');times=np.load(cache/'times.npy');geometry=np.load(prior/'shared-states.npy',mmap_mode='r');scatter=np.load(e16/'center-scatter.npy',mmap_mode='r')
    ids=tensor(np.load(prior/'shared-grouping.npz')['representatives']).long()
    models={}
    for name,path in names.items():
        ck=torch.load(path,map_location='cuda');m=Appearance(sh,pc._opacity.detach(),10000);m.load_state_dict(ck['state_dict']);m.eval()
        assert ck['steps']==1520 and ck['condition']==('mean_covariance' if name=='mean' else 'moment_covariance')
        models[name]=m
    from utils.image_utils import psnr
    from utils.loss_utils import ssim
    import lpips
    scalar=lpips.LPIPS(net='alex',version='0.1').cuda().eval()
    spatial=lpips.LPIPS(net='alex',version='0.1',spatial=True).cuda().eval()
    tests=json.loads((root/'datasets/bouncingballs/transforms_test.json').read_text())['frames']
    originals={r['arm']:r for r in e16report['rows']}
    rows=[];maps={};summary={}
    for name in ['teacher','mean','moment']:
        summary[name]={region:{'n':0,'spatial_sum':0.,'MSE_sum':0.,'MAE_sum':0.,'gradient_sum':0.} for region in ['edge','other_foreground','background']}
    for j,r in enumerate(tests):
        i=int(np.argmin(abs(times-r['time'])));assert abs(times[i]-r['time'])<1e-7
        cam=scene.getTestCameras()[j]
        gt=tensor(np.asarray(Image.open(base/f'test/ours_20000/gt/{j:05d}.png').convert('RGB'),dtype=np.float32)/255).permute(2,0,1)
        gray_gt,gradient_gt,region=masks(gt)
        frames={}
        with torch.no_grad():
            frames['teacher']=render_state(cam,tensor(full[i]),sh,opacity,cfg.sh_degree)
            for name,model in models.items():
                rep=geometry[i] if name=='mean' else add_scatter(geometry[i],scatter[i],1)
                color,alpha=model(ids);frames[name]=render_state(cam,tensor(rep),color,alpha,cfg.sh_degree)
            spatial_maps={}
            for name,pred in frames.items():
                png=(pred*255).round()/255
                value=float(scalar(png.unsqueeze(0),gt.unsqueeze(0),normalize=True))
                map_=spatial(png.unsqueeze(0),gt.unsqueeze(0),normalize=True).squeeze(0).squeeze(0)
                assert map_.shape==gt.shape[1:] and torch.isfinite(map_).all()
                spatial_maps[name]=map_
                gray_pred,grad_pred,_=masks(png)
                # masks(png) has its own masks; we always use the GT masks below.
                diff=png-gt
                row={'view':j,'time_index':i,'arm':name,'PSNR':float(psnr(png.unsqueeze(0),gt.unsqueeze(0))),
                     'SSIM':float(ssim(png.unsqueeze(0),gt.unsqueeze(0))),'LPIPS':value,'spatial_map_mean':float(map_.mean()),
                     'foreground_fraction':float(region['edge'].logical_or(region['other_foreground']).float().mean())}
                for label,mask in region.items():
                    n=int(mask.sum());assert n>0
                    row[label+'_n']=n
                    row[label+'_spatial_mean']=float(map_[mask].mean())
                    row[label+'_MSE']=float((diff[:,mask]**2).mean())
                    row[label+'_MAE']=float(diff[:,mask].abs().mean())
                    row[label+'_gradient_mean']=float(grad_pred[mask].mean())
                    agg=summary[name][label];agg['n']+=n;agg['spatial_sum']+=float(map_[mask].sum());agg['MSE_sum']+=float((diff[:,mask]**2).sum());agg['MAE_sum']+=float(diff[:,mask].abs().sum());agg['gradient_sum']+=float(grad_pred[mask].sum())
                if name in models:
                    old=originals[f'{name}_covariance__seed42']
                    # Original E16 rows are 'mean_covariance__seed42' and 'moment_covariance__seed42'.
                    original_row=originals[f'{name}_covariance__seed42']
                    for metric,old_key in [('PSNR','PSNR'),('SSIM','SSIM'),('LPIPS','LPIPS_alex_v01')]:
                        priorrow=next(q for q in json.loads((e16/'gt-per-view.json').read_text()) if q['arm']==f'{name}_covariance__seed42' and q['view']==j)
                        assert abs(row[metric]-priorrow[old_key])<2e-4,(j,name,metric,row[metric],priorrow[old_key])
                rows.append(row)
            if j in [0,10,19]:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig,axes=plt.subplots(2,4,figsize=(15,7))
                images=[gt,frames['teacher'],frames['mean'],frames['moment']]
                for ax,img,title in zip(axes[0],images,['GT','Full teacher','Mean covariance','Moment matching']):
                    ax.imshow(img.permute(1,2,0).cpu());ax.set_title(title);ax.axis('off')
                delta=(spatial_maps['moment']-spatial_maps['mean']).cpu().numpy();limit=float(np.quantile(abs(delta),.995));limit=max(limit,1e-6)
                plots=[(spatial_maps['mean'].cpu().numpy(),'Mean LPIPS map'),(spatial_maps['moment'].cpu().numpy(),'Moment LPIPS map'),(delta,'Moment minus mean'),(region['edge'].cpu().numpy(),'GT edge mask')]
                for ax,(arr,title) in zip(axes[1],plots):
                    if title=='Moment minus mean':im=ax.imshow(arr,cmap='coolwarm',vmin=-limit,vmax=limit)
                    elif title=='GT edge mask':im=ax.imshow(arr,cmap='gray')
                    else:im=ax.imshow(arr,cmap='magma',vmin=0,vmax=float(max(spatial_maps['mean'].max(),spatial_maps['moment'].max())))
                    ax.set_title(title);ax.axis('off')
                fig.suptitle(f'E17 view {j}: spatial LPIPS map is an approximate localization, not additive score decomposition')
                fig.tight_layout();fig.savefig(out/f'view{j:02d}-diagnostic.png',dpi=140);plt.close(fig)
        print('VIEW',j,'LPIPS mean/moment',rows[-2]['LPIPS'],rows[-1]['LPIPS'],flush=True)
    for name in summary:
        for label,agg in summary[name].items():
            n=agg['n'];agg['spatial_mean']=agg.pop('spatial_sum')/n;agg['MSE']=agg.pop('MSE_sum')/(3*n);agg['MAE']=agg.pop('MAE_sum')/(3*n);agg['gradient_mean']=agg.pop('gradient_sum')/n
    by={(x['view'],x['arm']):x for x in rows}
    paired=[]
    for j in range(20):
        a=by[(j,'mean')];b=by[(j,'moment')]
        paired.append({'view':j,'PSNR_delta':b['PSNR']-a['PSNR'],'LPIPS_delta':b['LPIPS']-a['LPIPS'],
                       **{label+'_spatial_delta':b[label+'_spatial_mean']-a[label+'_spatial_mean'] for label in ['edge','other_foreground','background']},
                       **{label+'_MSE_delta':b[label+'_MSE']-a[label+'_MSE'] for label in ['edge','other_foreground','background']}})
    assert all(p['LPIPS_delta']>0 for p in paired)
    dump(out/'per-view.json',rows);dump(out/'paired-per-view.json',paired)
    result={'regions':summary,'paired':{'views':20,'LPIPS_worse_views':sum(q['LPIPS_delta']>0 for q in paired),'PSNR_better_views':sum(q['PSNR_delta']>0 for q in paired),
              'mean_LPIPS_delta':float(np.mean([q['LPIPS_delta'] for q in paired])),
              'mean_PSNR_delta':float(np.mean([q['PSNR_delta'] for q in paired]))},
            'scope':protocol['boundaries'],
            'notes':['Spatial LPIPS map is a localization proxy, not an exact decomposition of scalar LPIPS.',
                     'Masks are derived from GT only and fixed by protocol; mask image and edge gradient units are RGB grayscale in [0,1].']}
    dump(out/'report.json',result)
    for p,h in inputs.items():assert digest(p)==h,p
    for p,h in oldhashes.items():assert digest(pathlib.Path(p))==h,p
    dump(out/'completion.json',{'status':'completed','no_training':True,'source_sha256':digest(out/'source.zip'),'inputs_unchanged':True})
    print('RESULT',json.dumps(result),flush=True)
if __name__=='__main__':main()
