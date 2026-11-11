#!/usr/bin/env python3
"""Safely extract one fixed D-NeRF validation scene from the archived source."""
import hashlib,json,os,pathlib,shutil,tempfile,zipfile
from datetime import datetime,timezone

SCENE='jumpingjacks'
ARCHIVE_SHA256='4e901bd3463e874bed5dd6bbe3236a8b2348f9adb5ef20484f8ba8b6d96876ec'

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def validate(data):
    result={}
    for split,expected in [('train',200),('val',20),('test',20)]:
        doc=json.loads((data/f'transforms_{split}.json').read_text())
        frames=doc['frames'];assert len(frames)==expected
        times=[]
        for frame in frames:
            path=data/frame['file_path']
            if not path.suffix:path=path.with_suffix('.png')
            assert path.resolve().is_relative_to(data.resolve()) and path.is_file(),str(path)
            assert 0<=frame['time']<=1
            times.append(frame['time'])
        assert len(set(times))==expected
        result[split]={'frames':len(frames),'unique_times':len(set(times))}
    return result

def main():
    root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve()
    assert root.is_relative_to('/data3/store1')
    assert root.stat().st_uid==os.getuid()
    target=root/'datasets'/SCENE
    if target.exists():
        print('Existing scene validated, not overwritten:',validate(target));return
    archive=root/'cache/downloads/dnerf-data.zip'
    assert digest(archive)==ARCHIVE_SHA256
    assert shutil.disk_usage(root).free>8*1024**3
    (root/'tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='jumpingjacks-',dir=root/'tmp') as tmp:
        staging=pathlib.Path(tmp)/SCENE;staging.mkdir()
        with zipfile.ZipFile(archive) as z:
            prefix=f'data/{SCENE}/';count=0
            for member in z.infolist():
                if not member.filename.startswith(prefix):continue
                relative=pathlib.PurePosixPath(member.filename[len(prefix):])
                if not relative.parts:continue
                assert not relative.is_absolute() and '..' not in relative.parts
                assert ((member.external_attr>>16)&0o170000)!=0o120000
                dest=staging.joinpath(*relative.parts)
                assert dest.resolve().is_relative_to(staging.resolve())
                if member.is_dir():dest.mkdir(parents=True,exist_ok=True);continue
                dest.parent.mkdir(parents=True,exist_ok=True)
                with z.open(member) as src,dest.open('xb') as dst:shutil.copyfileobj(src,dst)
                count+=1
        splits=validate(staging)
        train={f['time'] for f in json.loads((staging/'transforms_train.json').read_text())['frames']}
        test={f['time'] for f in json.loads((staging/'transforms_test.json').read_text())['frames']}
        assert len(train&test)==20
        files={str(p.relative_to(staging)):{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(staging.rglob('*')) if p.is_file()}
        manifest={'source_archive':str(archive),'archive_sha256':ARCHIVE_SHA256,'prepared_utc':datetime.now(timezone.utc).isoformat(),'scene':SCENE,'splits':splits,'teacher_train_test_time_overlap':20,'extracted_files':count,'files':files}
        (staging/'data-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        staging.rename(target)
    print(json.dumps({'target':str(target),'splits':splits,'files':count,'bytes':sum(v['bytes'] for v in files.values())},indent=2))

if __name__=='__main__':main()
