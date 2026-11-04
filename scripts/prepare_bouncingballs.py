#!/usr/bin/env python3
"""Download official D-NeRF archive and extract only bouncingballs, without overwriting."""
import argparse, hashlib, json, os, pathlib, shutil, subprocess, tempfile, zipfile
from datetime import datetime, timezone

URL = 'https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=1'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def validate(data):
    splits = {}
    for split in ['train', 'val', 'test']:
        doc = json.loads((data / ('transforms_' + split + '.json')).read_text())
        assert doc['frames'], 'Empty split ' + split
        for frame in doc['frames']:
            path = data / frame['file_path']
            if not path.suffix: path = path.with_suffix('.png')
            if not path.resolve().is_relative_to(data.resolve()) or not path.is_file():
                raise ValueError('Missing/unsafe image: ' + str(path))
            if not 0 <= frame['time'] <= 1: raise ValueError('Invalid frame time')
        splits[split] = len(doc['frames'])
    return splits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=pathlib.Path, help='Existing archive located inside RESEARCH_ROOT')
    args = parser.parse_args()
    root = pathlib.Path(os.environ['RESEARCH_ROOT']).resolve()
    target = root / 'datasets/bouncingballs'
    if target.exists():
        print('Existing dataset preserved:', validate(target)); return
    archive = args.archive or root / 'cache/downloads/dnerf-data.zip'
    if not archive.resolve().is_relative_to(root): raise ValueError('Archive must be inside RESEARCH_ROOT')
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        if shutil.disk_usage(root).free < 8 * 2**30: raise RuntimeError('Need 8 GiB free')
        part = archive.with_suffix('.zip.part')
        subprocess.run(['curl', '--fail', '--location', '--retry', '2', '--connect-timeout', '20',
                        '--speed-time', '60', '--speed-limit', '1024', '--max-time', '1800',
                        '--output', str(part), URL], check=True)
        with zipfile.ZipFile(part) as z:
            if not any('bouncingballs' in pathlib.PurePosixPath(n).parts for n in z.namelist()):
                raise ValueError('No bouncingballs in downloaded archive')
        part.rename(archive)
    with tempfile.TemporaryDirectory(prefix='bouncingballs-', dir=root / 'tmp') as tmp:
        staging = pathlib.Path(tmp) / 'scene'; staging.mkdir()
        with zipfile.ZipFile(archive) as z:
            prefixes = set()
            for member in z.infolist():
                parts = pathlib.PurePosixPath(member.filename).parts
                if 'bouncingballs' not in parts: continue
                index = parts.index('bouncingballs')
                relative = pathlib.PurePosixPath(*parts[index+1:])
                prefixes.add(parts[:index+1])
                if '..' in relative.parts or relative.is_absolute(): raise ValueError('Unsafe archive entry')
                if ((member.external_attr >> 16) & 0o170000) == 0o120000: raise ValueError('Symlink in archive')
                dest = staging.joinpath(*relative.parts)
                if member.is_dir(): dest.mkdir(parents=True, exist_ok=True); continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, dest.open('xb') as out: shutil.copyfileobj(src, out)
            if len(prefixes) != 1: raise ValueError('Ambiguous scene prefixes')
        splits = validate(staging)
        manifest = {'source': URL, 'archive_sha256': digest(archive), 'splits': splits,
                    'prepared_utc': datetime.now(timezone.utc).isoformat(),
                    'files': {str(p.relative_to(staging)): {'bytes': p.stat().st_size, 'sha256': digest(p)}
                              for p in sorted(staging.rglob('*')) if p.is_file()}}
        (staging / 'data-manifest.json').write_text(json.dumps(manifest, indent=2))
        staging.rename(target)
    print('Dataset prepared:', target, splits)

if __name__ == '__main__': main()
