#!/usr/bin/env python3
"""Print a baseline plan by default. --run explicitly starts training in a fresh output."""
import argparse, hashlib, json, os, pathlib, shlex, subprocess, sys, zipfile
from datetime import datetime, timezone

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--run-id', default=datetime.now(timezone.utc).strftime('baseline-%Y%m%dT%H%M%SZ'))
    args = parser.parse_args()
    if not args.run_id or pathlib.Path(args.run_id).name != args.run_id or args.run_id in ('.', '..'):
        parser.error('run-id must be a single directory name')
    root = pathlib.Path(os.environ['RESEARCH_ROOT']).resolve()
    repo = root / 'projects/4DGaussians'
    data = root / 'datasets/bouncingballs'
    out = root / 'outputs/dnerf/bouncingballs' / args.run_id
    if out.exists(): raise SystemExit('Output already exists; refusing to overwrite: ' + str(out))
    if not out.resolve().is_relative_to(root): raise SystemExit('Unsafe output path')
    cmd = [sys.executable, 'train.py', '-s', str(data), '-m', str(out),
           '--configs', 'arguments/dnerf/bouncingballs.py', '--expname', args.run_id,
           '--port', '6017', '--save_iterations', '20000', '--test_iterations', '20000']
    print('cwd:', repo)
    print('CUDA_VISIBLE_DEVICES=' + str(args.gpu), shlex.join(cmd), flush=True)
    if not args.run:
        print('PLAN ONLY. No training or output creation.'); return
    subprocess.run([sys.executable, str(pathlib.Path(__file__).with_name('preflight.py')), '--require-ready'], check=True)
    subprocess.run([sys.executable, str(pathlib.Path(__file__).with_name('verify_env.py'))], check=True,
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(args.gpu)})
    out.mkdir(parents=True, exist_ok=False)
    def git(path, *params):
        return subprocess.check_output(['git', '-C', str(path), *params], text=True).strip()
    manifest = {'command': cmd, 'cwd': str(repo), 'gpu': args.gpu,
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'upstream_commit': git(repo, 'rev-parse', 'HEAD'),
                'research_commit': git(pathlib.Path(__file__).resolve().parents[1], 'rev-parse', 'HEAD'),
                'research_status': git(pathlib.Path(__file__).resolve().parents[1], 'status', '--porcelain'),
                'source_path': str(data), 'output_path': str(out), 'upstream_seed': 6666}
    research=pathlib.Path(__file__).resolve().parents[1]
    names=subprocess.check_output(['git','-C',str(research),'ls-files','--cached','--others','--exclude-standard','-z']).split(b'\0')
    snapshot=out/'research-source.zip'
    with zipfile.ZipFile(snapshot,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for raw in sorted(set(names)):
            if not raw: continue
            relative=os.fsdecode(raw); source=research/relative
            if not source.resolve().is_relative_to(research): raise ValueError('Source snapshot redirects outside research repository')
            if source.is_file(): archive.write(source,relative)
    manifest['research_snapshot_sha256']=hashlib.sha256(snapshot.read_bytes()).hexdigest()
    (out / 'run-manifest.json').write_text(json.dumps(manifest, indent=2))
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True, env={**os.environ, 'PIP_REQUIRE_VIRTUALENV': 'false'})
    (out / 'pip-freeze.txt').write_text(freeze)
    with (out / 'train.log').open('x') as log:
        result = subprocess.run(cmd, cwd=repo, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(args.gpu)})
    manifest['exit_code'] = result.returncode
    manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (out / 'run-manifest.json').write_text(json.dumps(manifest, indent=2))
    raise SystemExit(result.returncode)

if __name__ == '__main__': main()
