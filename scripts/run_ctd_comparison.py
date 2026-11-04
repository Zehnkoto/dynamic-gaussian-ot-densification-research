#!/usr/bin/env python3
"""E02 CTD adaptation pilot; immutable outputs; shared grouping vs per-frame oracle."""
import argparse,hashlib,json,math,os,pathlib,subprocess,time,zipfile
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
from reduction_core import lloyd,barycenters
from reduction_scene import load_scene,state,small_camera,render_state

def dump(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path,required=True);ap.add_argument('--cache',type=pathlib.Path);ap.add_argument('--budgets',nargs='+',type=int,default=[5000]);ap.add_argument('--iterations',type=int,default=20);args=ap.parse_args()
    root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();research=root/'projects/dynamic-gaussian-ot-densification-research';out=args.output.resolve()
    if not out.is_relative_to(root) or out.exists():raise ValueError('Output outside data root or already exists')
    model=root/'outputs/dnerf/bouncingballs/baseline-numpy1235-20261030T174401Z'
    for name,expected in [('4DGaussians','843d5ac636c37e4b611242287754f3d4ed150144'),('GHAP','27b3a581c861069cc97e5629e216dc3d1f42a98a')]:
        actual=subprocess.check_output(['git','-C',str(root/'projects'/name),'rev-parse','HEAD'],text=True).strip()
        if actual!=expected:raise ValueError('Upstream pin mismatch')
        if subprocess.check_output(['git','-C',str(root/'projects'/name),'status','--porcelain'],text=True).strip():raise ValueError('Dirty upstream')
    baseline=json.loads((model/'baseline-validation.json').read_text())
    for relative,record in baseline['model_files'].items():
        if digest(model/relative)!=record['sha256']:raise ValueError('Base checkpoint changed')
    protocol=json.loads((research/'experiments/reduction-ctd-v2.json').read_text());source=json.loads((root/'datasets/bouncingballs/transforms_train.json').read_text())
    times=np.array(sorted(f['time'] for f in source['frames']),dtype=np.float64);assert len(np.unique(times))==150
    fit=np.array([int(np.argmin(abs(times-f['time']))) for f in protocol['fit']]);held=np.array([int(np.argmin(abs(times-f['time']))) for f in protocol['reduction_holdout']])
    assert len(fit)==76 and len(held)==74 and not set(fit)&set(held)
    out.mkdir(parents=True,exist_ok=False)
    cfg,pc,scene,pipe=load_scene(model);pc._deformation.eval();n=len(pc.get_xyz)
    assert all(0<k<=n for k in args.budgets)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    fixed=small_camera(scene.getTestCameras()[10],256)
    provenance={'model':str(model),'base_gaussians':n,'protocol':protocol,'max_iter':args.iterations,'relative_tol':1e-5,'seed':42,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'baseline_model_hashes':baseline['model_files'],'status':'running','method_label':'GHAP-CTD adaptation, not official full pipeline'}
    dump(out/'run-manifest.json',provenance)
    with zipfile.ZipFile(out/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ['run_ctd_comparison.py','reduction_core.py','reduction_scene.py','test_reduction_core.py','inspect_baseline.py','common.sh']:z.write(research/'scripts'/name,'scripts/'+name)
        z.write(research/'experiments/reduction-ctd-v2.json','experiments/reduction-ctd-v2.json')
    cache=args.cache.resolve() if args.cache else out/'trajectory-cache'
    if not cache.is_relative_to(root):raise ValueError('Unsafe cache')
    with torch.no_grad():
        f0,o0,s0=state(pc,0.);fixed.time=0.
        from gaussian_renderer import render as native_render
        ref=native_render(fixed,pc,pipe,torch.ones(3,device='cuda'),cam_type=scene.dataset_type)['render'].clamp(0,1)
        reconstructed=render_state(fixed,f0,s0,o0,cfg.sh_degree);difference=(ref-reconstructed).abs()
        check={'mean_abs_error':float(difference.mean()),'max_abs_error':float(difference.max())}
        if check['mean_abs_error']>1e-5 or check['max_abs_error']>.005:raise ValueError('Native state rendering mismatch: '+str(check))
        dump(out/'native-render-check.json',check);print('Native render parity:',check,flush=True)
        if args.cache:
            meta=json.loads((cache/'manifest.json').read_text())
            if meta['baseline_model_hashes']!=baseline['model_files'] or meta['count']!=n:raise ValueError('Cache model mismatch')
            if not np.array_equal(np.load(cache/'times.npy'),times):raise ValueError('Cache time mismatch')
            for name,record in meta['files'].items():
                if digest(cache/name)!=record['sha256']:raise ValueError('Cache checksum mismatch')
        else:
            cache.mkdir();states=np.lib.format.open_memmap(cache/'states.npy',mode='w+',dtype=np.float32,shape=(len(times),n,12))
            start=time.perf_counter()
            for i,t in enumerate(times):
                feat,op,sh=state(pc,t)
                if not torch.allclose(op,o0,atol=1e-7,rtol=0) or not torch.allclose(sh,s0,atol=1e-7,rtol=0):raise ValueError('Appearance not time-shared')
                if not torch.isfinite(feat).all():raise ValueError('Nonfinite source state')
                states[i]=feat.cpu().numpy()
                if i%15==0:print('Trajectory extraction',i,'/150',flush=True)
            states.flush();del states
            np.save(cache/'times.npy',times);np.save(cache/'opacity.npy',o0.cpu().numpy());np.save(cache/'sh.npy',s0.cpu().numpy())
            meta={'count':n,'times':150,'fit_times':76,'feature_layout':'mean xyz (3), covariance full row-major (9)','seconds':time.perf_counter()-start,'baseline_model_hashes':baseline['model_files'],'files':{name:{'sha256':digest(cache/name)} for name in ['states.npy','times.npy','opacity.npy','sh.npy']}}
            dump(cache/'manifest.json',meta)
        states=np.load(cache/'states.npy',mmap_mode='r');opacity=torch.tensor(np.load(cache/'opacity.npy'),device='cuda');sh=torch.tensor(np.load(cache/'sh.npy'),device='cuda')
        weights=opacity.flatten().double();weights/=weights.sum()
        from utils.loss_utils import ssim
        from utils.image_utils import psnr
        results=[]
        for budget in args.budgets:
            dest=out/f'budget-{budget}';dest.mkdir();torch.cuda.reset_peak_memory_stats()
            print('Begin budget',budget,flush=True)
            initial_ids=torch.randperm(n,generator=torch.Generator().manual_seed(42))[:budget].cuda()
            fitting=torch.tensor(np.transpose(states[fit],(1,0,2)).copy().reshape(n,-1),device='cuda',dtype=torch.float64)/math.sqrt(len(fit))
            shared=lloyd(fitting,weights,initial_ids,max_iter=args.iterations)
            shared_info=shared['info'];labels=shared['labels'];shared_reps=shared['representatives'];del fitting;del shared['centers']
            np.savez(dest/'shared-grouping.npz',labels=labels.cpu().numpy(),representatives=shared_reps.cpu().numpy(),normalized_mass=shared['mass'].cpu().numpy(),initial_source_ids=initial_ids.cpu().numpy())
            print('Shared fit finished',shared_info,flush=True)
            shared_states=np.lib.format.open_memmap(dest/'shared-states.npy',mode='w+',dtype=np.float32,shape=(150,budget,12))
            frame_states=np.lib.format.open_memmap(dest/'perframe-states.npy',mode='w+',dtype=np.float32,shape=(150,budget,12))
            frame_reps=np.lib.format.open_memmap(dest/'perframe-representatives.npy',mode='w+',dtype=np.int32,shape=(150,budget))
            frame_labels=np.lib.format.open_memmap(dest/'perframe-labels.npy',mode='w+',dtype=np.int32,shape=(150,n))
            rows=[];previous=None;temporal={'shared':[],'perframe':[]};apply_seconds=0.;frame_fit_seconds=0.
            with imageio.get_writer(dest/'fixed-camera-comparison.mp4',fps=15,codec='libx264',quality=8,macro_block_size=1) as video:
                for i,t in enumerate(times):
                    features=torch.tensor(np.array(states[i],copy=True),device='cuda',dtype=torch.float64)
                    torch.cuda.synchronize();started=time.perf_counter();centers,mass=barycenters(features,labels,weights,budget);torch.cuda.synchronize();apply_seconds+=time.perf_counter()-started
                    frame=lloyd(features,weights,initial_ids,max_iter=args.iterations);frame_fit_seconds+=frame['info']['seconds']
                    shared_states[i]=centers.float().cpu().numpy();frame_states[i]=frame['centers'].float().cpu().numpy();frame_reps[i]=frame['representatives'].cpu().numpy();frame_labels[i]=frame['labels'].cpu().numpy()
                    fixed.time=float(t)
                    images={'full':render_state(fixed,features,sh,opacity,cfg.sh_degree),'shared':render_state(fixed,centers,sh[shared_reps],opacity[shared_reps],cfg.sh_degree),'perframe':render_state(fixed,frame['centers'],sh[frame['representatives']],opacity[frame['representatives']],cfg.sh_degree)}
                    row={'time':float(t),'split':'fit' if i in set(fit) else 'shared-fit-holdout','shared_ctd':float((weights*((features-centers[labels])**2).sum(1)).sum()),'perframe_ctd':frame['info']['objective_history'][-1],'perframe_fit':frame['info']}
                    for method in ['shared','perframe']:
                        row[method+'_psnr_to_full_float_256']=float(psnr(images[method].unsqueeze(0),images['full'].unsqueeze(0)))
                        if previous is not None:temporal[method].append(float(((images[method]-previous[method])-(images['full']-previous['full'])).abs().mean()))
                    previous=images
                    panel=Image.new('RGB',(768,296),'white');draw=ImageDraw.Draw(panel)
                    for x,method,text in [(0,'full','Full baseline'),(256,'perframe','Per-frame CTD oracle'),(512,'shared','Shared CTD grouping')]:
                        draw.text((x+5,5),text,font=font,fill='black');draw.text((x+5,22),f'K={n if method=="full" else budget}  t={t:.3f}',font=font,fill='black')
                        panel.paste(Image.fromarray((images[method].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)),(x,40))
                    video.append_data(np.asarray(panel));rows.append(row)
                    if i%10==0:print('Budget',budget,'time',i,'/150','frame iterations',frame['info']['iterations'],flush=True)
                    del frame
            for mm in [shared_states,frame_states,frame_reps,frame_labels]:mm.flush()
            import lpips
            criterion=lpips.LPIPS(net='alex',version='0.1').cuda().eval();gt_results={method:[] for method in ['shared','perframe']}
            test_frames=json.loads((root/'datasets/bouncingballs/transforms_test.json').read_text())['frames']
            for method in gt_results:(dest/(method+'-test')).mkdir()
            for i,record in enumerate(test_frames):
                ti=int(np.argmin(abs(times-record['time'])));assert abs(times[ti]-record['time'])<1e-7
                cam=scene.getTestCameras()[i];gt_np=np.asarray(Image.open(model/f'test/ours_20000/gt/{i:05d}.png').convert('RGB'),dtype=np.float32)/255
                gt=torch.tensor(gt_np,device='cuda').permute(2,0,1).unsqueeze(0)
                for method,cache_states,reps in [('shared',shared_states,shared_reps),('perframe',frame_states,torch.tensor(np.array(frame_reps[ti],copy=True),device='cuda',dtype=torch.long))]:
                    features=torch.tensor(np.array(cache_states[ti],copy=True),device='cuda')
                    pred=render_state(cam,features,sh[reps],opacity[reps],cfg.sh_degree);pred=(pred*255).round()/255
                    pred=pred.unsqueeze(0)
                    gt_results[method].append({'test_index':i,'time':record['time'],'PSNR':float(psnr(pred,gt)),'SSIM':float(ssim(pred,gt)),'LPIPS_alex_v01':float(criterion(pred,gt,normalize=True))})
                    Image.fromarray((pred[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)).save(dest/(method+'-test')/f'{i:05d}.png')
            dump(dest/'time-results.json',rows);dump(dest/'gt-per-view.json',gt_results)
            summary={'budget':budget,'gaussians_before':n,'counts_exact':True,'max_iter':args.iterations,'perframe_converged':sum(x['perframe_fit']['converged'] for x in rows),'perframe_total':len(rows),'perframe_empty_cluster_repairs':sum(x['perframe_fit']['empty_cluster_repairs'] for x in rows),'shared_fit':shared_info,'shared_application_150_times_seconds':apply_seconds,'perframe_fitting_150_times_seconds':frame_fit_seconds,'fixed_camera_temporal_delta_residual':{m:float(np.mean(v)) for m,v in temporal.items()},'fit_and_holdout':{},'official_20_view_gt':{method:{metric:float(np.mean([x[metric] for x in values])) for metric in ['PSNR','SSIM','LPIPS_alex_v01']} for method,values in gt_results.items()},'peak_process_gpu_bytes_including_render_lpips':torch.cuda.max_memory_allocated(),'metric_notes':['Per-frame is an oracle fitted at each evaluation time.','Temporal residual uses full-model float renders as reference, not GT.','GT metrics use saved-equivalent 8-bit RGB at original 800x800 resolution; LPIPS normalize=True.','No standalone compressed dynamic model, appearance fine-tuning, velocity penalty or adaptive split.','Iteration cap recorded in max_iter; inspect convergence flags before drawing conclusions.']}
            for split in ['fit','shared-fit-holdout']:
                selected=[x for x in rows if x['split']==split]
                summary['fit_and_holdout'][split]={key:float(np.mean([x[key] for x in selected])) for key in ['shared_ctd','perframe_ctd','shared_psnr_to_full_float_256','perframe_psnr_to_full_float_256']}
            dump(dest/'summary.json',summary);results.append(summary);del criterion
            print('BUDGET COMPLETE',json.dumps(summary),flush=True)
        dump(out/'results.json',results)
        dump(out/'completion.json',{'status':'completed','budgets':args.budgets,'trajectory_cache':str(cache),'source_snapshot_sha256':digest(out/'source.zip')})
        print('Comparison completed:',out,flush=True)
if __name__=='__main__':main()
