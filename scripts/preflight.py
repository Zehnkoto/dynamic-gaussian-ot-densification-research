#!/usr/bin/env python3
"""Read-only storage, source version and dependency audit; no GPU allocation."""
import argparse, getpass, importlib.util, json, os, pathlib, shutil, subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-ready', action='store_true')
    args = parser.parse_args()
    root = pathlib.Path('/data3/store1') / getpass.getuser()
    result = {'root': str(root), 'repositories': {}, 'storage': {}, 'dataset': {}, 'dependencies': {}}
    failures = []
    for name, expected in [('4DGaussians', '843d5ac636c37e4b611242287754f3d4ed150144'), ('GHAP', '27b3a581c861069cc97e5629e216dc3d1f42a98a')]:
        repo = root / 'projects' / name
        def git(*cmd):
            return subprocess.check_output(['git', '-C', str(repo), *cmd], text=True, env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'}).strip()
        try:
            sha = git('rev-parse', 'HEAD')
            status = git('status', '--porcelain')
            subs = git('submodule', 'status', '--recursive')
            result['repositories'][name] = {'commit': sha, 'expected': expected, 'status': status, 'submodules': subs}
            if sha != expected or status or any(s.startswith(('-', '+', 'U')) for s in subs.splitlines()):
                failures.append(name + ': version/worktree/submodule mismatch')
        except subprocess.CalledProcessError:
            failures.append(name + ': unavailable')
    for name in ['projects', 'datasets', 'outputs', 'envs', 'cache', 'tmp']:
        path = root / name
        okay = path.is_dir() and path.resolve().is_relative_to(root.resolve()) and path.stat().st_uid == os.getuid()
        result['storage'][name] = {'resolved': str(path.resolve()), 'owned_and_contained': okay}
        if not okay: failures.append('Unsafe storage: ' + name)
    result['free_gib'] = round(shutil.disk_usage(root).free / 2**30, 2)
    if result['free_gib'] < 8: failures.append('Less than 8 GiB free')
    data = root / 'datasets/bouncingballs'
    for split in ['train', 'val', 'test']:
        path = data / ('transforms_' + split + '.json')
        if not path.is_file():
            result['dataset'][split] = 'MISSING'
            continue
        doc = json.loads(path.read_text())
        missing = []
        for frame in doc['frames']:
            image = data / frame['file_path']
            if not image.suffix: image = image.with_suffix('.png')
            if not image.resolve().is_relative_to(data.resolve()) or not image.is_file(): missing.append(str(image))
        result['dataset'][split] = {'frames': len(doc['frames']), 'missing_images': missing}
        if missing: failures.append('Missing/unsafe images: ' + split)
    for name in ['torch', 'torchvision', 'numpy', 'scipy', 'mmcv', 'open3d', 'cv2', 'lpips', 'plyfile', 'diff_gaussian_rasterization', 'simple_knn']:
        try: available = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError): available = False
        result['dependencies'][name] = available
    for key in ['TMPDIR', 'CONDA_PKGS_DIRS', 'PIP_CACHE_DIR', 'HF_HOME', 'TORCH_HOME', 'TORCH_EXTENSIONS_DIR', 'CUDA_CACHE_PATH', 'HOME']:
        value = os.environ.get(key)
        okay = bool(value) and pathlib.Path(value).resolve().is_relative_to(root.resolve())
        result['storage'][key] = {'path': value, 'contained': okay}
        if not okay: failures.append('Unconfigured/outside-root environment: ' + key)
    if args.require_ready:
        for split, value in result['dataset'].items():
            if value == 'MISSING': failures.append('Missing data: ' + split)
        for name, okay in result['dependencies'].items():
            if not okay: failures.append('Missing dependency: ' + name)
    result['failures'] = failures
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return bool(failures)

if __name__ == '__main__':
    raise SystemExit(main())
