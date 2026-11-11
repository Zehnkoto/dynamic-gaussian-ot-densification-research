#!/usr/bin/env python3
"""E19: pinned official 4DGS jumpingjacks teacher, isolated owner-volume output."""
import argparse,hashlib,json,os,pathlib,shutil,subprocess,sys,zipfile
from datetime import datetime,timezone

SCENE='jumpingjacks'
PIN='843d5ac636c37e4b611242287754f3d4ed150144'

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def utc():return datetime.now(timezone.utc).isoformat()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path,required=True);ap.add_argument('--gpu',type=int,default=0);args=ap.parse_args()
    root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();research=root/'projects/dynamic-gaussian-ot-densification-research';four=root/'projects/4DGaussians';data=root/'datasets'/SCENE;out=args.output.resolve()
    assert out.is_relative_to(root/'outputs/dnerf'/SCENE) and not out.exists()
    assert root.stat().st_uid==os.getuid() and shutil.disk_usage(root).free>8*1024**3
    protocol=research/'experiments/second-scene-validation-v1.json';spec=json.loads(protocol.read_text())
    assert spec['scene']=='D-NeRF jumpingjacks' and spec['teacher']['repo_commit']==PIN
    def git(*args):return subprocess.check_output(['git','-C',str(four),*args],text=True,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'}).strip()
    assert git('rev-parse','HEAD')==PIN and not git('status','--porcelain')
    manifest=json.loads((data/'data-manifest.json').read_text())
    assert manifest['archive_sha256']==spec['source_archive_sha256']
    assert manifest['splits']['train']['frames']==200 and manifest['splits']['test']['frames']==20
    for name,entry in manifest['files'].items():
        path=data/name;assert path.is_file() and path.resolve().is_relative_to(data.resolve()) and digest(path)==entry['sha256'],name
    assert digest(root/'cache/downloads/dnerf-data.zip')==manifest['archive_sha256']
    cmd=[sys.executable,'train.py','-s',str(data),'-m',str(out),'--configs','arguments/dnerf/jumpingjacks.py','--expname',out.name,'--port','6017','--save_iterations','20000','--test_iterations','20000']
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(args.gpu)}
    out.parent.mkdir(parents=True,exist_ok=True);out.mkdir(exist_ok=False)
    record={'phase':'training','started_utc':utc(),'scene':SCENE,'dataset':str(data),'dataset_manifest_sha256':digest(data/'data-manifest.json'),'protocol_sha256':digest(protocol),'repository_commit':PIN,'gpu':args.gpu,'training_command':cmd,'source_path':str(data),'output_path':str(out),'upstream_seed':6666}
    with zipfile.ZipFile(out/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in [research/'scripts/train_second_scene_baseline.py',research/'scripts/prepare_second_scene.py',protocol]:z.write(p,p.relative_to(research))
    record['source_snapshot_sha256']=digest(out/'source.zip')
    def write(): (out/'run-manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    write()
    with (out/'train.log').open('x') as f:
        result=subprocess.run(cmd,cwd=four,env=env,stdout=f,stderr=subprocess.STDOUT)
    record['train_exit_code']=result.returncode;record['train_finished_utc']=utc();write()
    if result.returncode:raise SystemExit(result.returncode)
    ply=out/'point_cloud/iteration_20000/point_cloud.ply';cfg=out/'cfg_args'
    assert ply.is_file() and cfg.is_file()
    from inspect_baseline import configuration
    conf=configuration(cfg)
    assert pathlib.Path(conf.source_path).resolve()==data.resolve()
    assert pathlib.Path(conf.model_path).resolve()==out
    assert conf.coarse_iterations==3000 and conf.iterations==20000
    assert conf.no_do and conf.no_dshs and conf.configs=='arguments/dnerf/jumpingjacks.py'
    record['checkpoint_sha256']=digest(ply);record['cfg_args_sha256']=digest(cfg);record['gaussian_count_pending']=True
    record['phase']='rendering_test';write()
    render=[sys.executable,'render.py','-m',str(out),'--iteration','20000','--skip_train','--skip_video']
    with (out/'render-test.log').open('x') as f:result=subprocess.run(render,cwd=four,env=env,stdout=f,stderr=subprocess.STDOUT)
    record['render_exit_code']=result.returncode;write()
    if result.returncode:raise SystemExit(result.returncode)
    gt=list((out/'test/ours_20000/gt').glob('*.png'));pred=list((out/'test/ours_20000/renders').glob('*.png'))
    assert len(gt)==len(pred)==20
    record['phase']='completed';record['test_gt_count']=20;record['test_render_count']=20;record['finished_utc']=utc();write()
    print(json.dumps({'status':'completed','output':str(out),'checkpoint_sha256':record['checkpoint_sha256'],'test_views':20},indent=2),flush=True)

if __name__=='__main__':main()
