#!/usr/bin/env python3
"""Install minimal CUDA 11.7.1 build toolkit from NVIDIA's SHA256-pinned redistributions."""
import hashlib, json, os, pathlib, shutil, subprocess, tarfile, tempfile
root = pathlib.Path(os.environ['RESEARCH_ROOT']).resolve()
target = root / 'envs/cuda-11.7.1'
components = [
 ('cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-11.7.99-archive.tar.xz', '4a75e5bd6a0e0e664ec5895ba050cf9d7cda08f41db2b6b4c36fb91cfb9627bc'),
 ('cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-11.7.99-archive.tar.xz', '7892409299c6865d0652373cb385add31b0faa2e5421a931ae9fbc103e1472ad'),
 ('cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-11.7.91-archive.tar.xz', 'bf051d7b22581e31fac992c12af1400036d38e6a53e230b08f2aac3fa93d593e')]
if target.exists():
    if not (target/'redistribution-manifest.json').is_file(): raise SystemExit('Existing toolkit needs inspection')
    subprocess.run([str(target/'bin/nvcc'),'--version'],check=True)
    raise SystemExit(0)
cache = root/'cache/downloads/cuda-11.7.1'; cache.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory(prefix='cuda-',dir=root/'tmp') as temp:
    staging=pathlib.Path(temp); toolkit=staging/'toolkit'; toolkit.mkdir()
    for index,(relative,expected) in enumerate(components):
        archive=cache/pathlib.PurePosixPath(relative).name
        if not archive.exists():
            part=archive.with_suffix('.xz.part')
            for attempt in range(5):
                print('CUDA fetch',archive.name,'attempt',attempt+1,flush=True)
                result=subprocess.run(['curl','--fail','--location','--silent','--show-error','--connect-timeout','20','--max-time','180',
                                       '--continue-at','-','-o',str(part),'https://developer.download.nvidia.com/compute/cuda/redist/'+relative])
                if part.exists() and hashlib.sha256(part.read_bytes()).hexdigest()==expected: break
                if result.returncode==0: raise ValueError('CUDA archive checksum mismatch')
            else: raise RuntimeError('CUDA download interrupted; resumable partial preserved')
            part.rename(archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected: raise ValueError('Cached CUDA archive checksum mismatch')
        unpack=staging/str(index); unpack.mkdir()
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                dest=(unpack/member.name).resolve()
                if not dest.is_relative_to(unpack): raise ValueError('Unsafe archive path')
                if member.isdev() or member.isfifo() or member.islnk(): raise ValueError('Unsupported archive entry')
                if member.issym() and not (dest.parent/member.linkname).resolve().is_relative_to(unpack):
                    raise ValueError('Unsafe archive link')
            tar.extractall(unpack)
        dirs=list(unpack.iterdir())
        if len(dirs)!=1 or not dirs[0].is_dir(): raise ValueError('Unexpected NVIDIA archive layout')
        license_file=dirs[0]/'LICENSE'
        if license_file.is_file():
            licenses=toolkit/'licenses'; licenses.mkdir(exist_ok=True)
            shutil.copy2(license_file,licenses/(str(index)+'-LICENSE'))
            license_file.unlink()
        # Some vendor files are read-only. Replace duplicates only inside our staging tree.
        for source in dirs[0].rglob('*'):
            dest=toolkit/source.relative_to(dirs[0])
            if not source.is_dir() and (dest.exists() or dest.is_symlink()): dest.unlink()
        shutil.copytree(dirs[0],toolkit,dirs_exist_ok=True,symlinks=True)
    (toolkit/'redistribution-manifest.json').write_text(json.dumps({'source':'https://developer.download.nvidia.com/compute/cuda/redist/redistrib_11.7.1.json','components':components},indent=2))
    toolkit.rename(target)
subprocess.run([str(target/'bin/nvcc'),'--version'],check=True)
