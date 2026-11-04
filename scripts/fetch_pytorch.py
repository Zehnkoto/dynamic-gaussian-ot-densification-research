#!/usr/bin/env python3
"""Resume bounded HTTP ranges from official PyTorch CDN; verify official full SHA256."""
import argparse, concurrent.futures, fcntl, hashlib, html.parser, json, os, pathlib
import re, shutil, time, threading, urllib.parse, urllib.request

try:
    import requests
except ImportError:
    from pip._vendor import requests
HOST='https://download-r2.pytorch.org'
PACKAGES=[('torch','1.13.1'),('torchvision','0.14.1'),('torchaudio','0.13.1')]
class Links(html.parser.HTMLParser):
    def __init__(self): super().__init__(); self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a': self.links.extend(v for k,v in attrs if k=='href')
def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def metadata(name,version):
    if name=='torch' and version=='1.13.1':
        return ('torch-1.13.1+cu117-cp310-cp310-linux_x86_64.whl', HOST+'/whl/cu117/torch-1.13.1%2Bcu117-cp310-cp310-linux_x86_64.whl', '14c5c9db09df8cf1b3942a3479c779da6e293a84a162d8a6ac71e2bde30e30c5')
    index='https://download.pytorch.org/whl/cu117/'+name+'/'
    last=None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(index,timeout=30) as response: page=response.read().decode()
            parser=Links(); parser.feed(page)
            filename=f'{name}-{version}+cu117-cp310-cp310-linux_x86_64.whl'
            matches=[]
            for href in parser.links:
                url=urllib.parse.urljoin(index,href); parts=urllib.parse.urlsplit(url)
                if urllib.parse.unquote(parts.path.rsplit('/',1)[-1])!=filename: continue
                digest=urllib.parse.parse_qs(parts.fragment).get('sha256',[''])[0]
                if not re.fullmatch('[0-9a-f]{64}',digest): raise ValueError('Missing official SHA256')
                if parts.hostname not in ('download.pytorch.org','download-r2.pytorch.org'): raise ValueError('Unexpected host')
                matches.append((filename,HOST+parts.path,digest))
            if len(set(matches))!=1: raise ValueError('Ambiguous/missing wheel: '+filename)
            return matches[0]
        except Exception as exc: last=exc; time.sleep(min(attempt+1,5))
    raise RuntimeError('Cannot read official wheel index') from last

def fetch_wheel(root,name,url,expected,workers,chunk_size,*,hash_prefix=False):
    if not re.fullmatch(r"[0-9a-f]{8}" if hash_prefix else r"[0-9a-f]{64}",expected):
        raise ValueError("Invalid expected SHA256 format")
    def matches(actual): return actual.startswith(expected) if hash_prefix else actual==expected
    target=root/name
    if target.exists():
        if not matches(sha256(target)): raise ValueError('Existing wheel hash mismatch; preserved for inspection')
        print('Verified cached wheel:',name,flush=True); return
    saved_manifest=root/(name+'.chunks')/'manifest.json'
    if saved_manifest.is_file():
        saved=json.loads(saved_manifest.read_text())
        if saved['url']!=url or saved['sha256']!=expected or saved['chunk_size']!=chunk_size:
            raise ValueError('Cached range metadata mismatch')
        size=int(saved['size'])
    else:
        for attempt in range(6):
            try:
                req=urllib.request.Request(url+'?probe=0',headers={'Range':'bytes=0-0','Accept-Encoding':'identity'})
                with urllib.request.urlopen(req,timeout=25) as response:
                    match=re.fullmatch(r'bytes 0-0/(\d+)',response.headers.get('Content-Range',''))
                    if response.status!=206 or not match or len(response.read(2))!=1:
                        raise ValueError('Server does not honor ranges')
                    size=int(match.group(1))
                break
            except ValueError: raise
            except Exception:
                if attempt==5: raise
                time.sleep(attempt+1)
    if shutil.disk_usage(root).free < 2*size+3*2**30: raise RuntimeError('Insufficient space for wheel assembly')
    chunks=root/(name+'.chunks'); chunks.mkdir(exist_ok=True)
    state={'url':url,'sha256':expected,'size':size,'chunk_size':chunk_size}
    manifest=chunks/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=state: raise ValueError('Chunk metadata mismatch')
    manifest.write_text(json.dumps(state,indent=2))
    count=(size+chunk_size-1)//chunk_size
    stop=threading.Event()
    local=threading.local()
    def fetch(index):
        start=index*chunk_size; end=min(size,start+chunk_size)-1; wanted=end-start+1
        complete=chunks/f'{index:06d}.chunk'; partial=chunks/f'{index:06d}.part'
        if complete.exists():
            if complete.stat().st_size!=wanted: raise ValueError('Invalid cached chunk')
            return
        if not hasattr(local,'session'): local.session=requests.Session()
        failures=0
        for attempt in range(160):
            if stop.is_set(): raise concurrent.futures.CancelledError()
            offset=partial.stat().st_size if partial.exists() else 0
            if offset==wanted: partial.rename(complete); return
            if offset>wanted: raise ValueError('Oversized partial chunk')
            begin=start+offset
            request_end=min(end,begin+1024*1024-1)
            try:
                with local.session.get(url+f'?range={begin}-{request_end}',
                      headers={'Range':f'bytes={begin}-{request_end}','Accept-Encoding':'identity'},
                      stream=True,timeout=(15,30)) as response:
                    response.raise_for_status()
                    cr=response.headers.get('Content-Range','')
                    if response.status_code!=206 or cr!=f'bytes {begin}-{request_end}/{size}':
                        raise ValueError('Incorrect response range: '+cr)
                    remaining=request_end-begin+1
                    with partial.open('ab') as out:
                        for block in response.iter_content(chunk_size=64*1024):
                            if stop.is_set(): raise concurrent.futures.CancelledError()
                            if not block: continue
                            if len(block)>remaining: raise ValueError('Oversized response body')
                            out.write(block); out.flush(); remaining-=len(block)
                    if remaining: raise IOError('Interrupted range; resuming')
                failures=0
                if partial.stat().st_size==wanted: partial.rename(complete); return
            except ValueError as exc:
                print('Range protocol failure',index,str(exc),flush=True); raise
            except concurrent.futures.CancelledError: raise
            except Exception as exc:
                after=partial.stat().st_size if partial.exists() else 0
                failures=failures+1 if after==offset else 0
                if failures in (1,4,8,12): print('Retry chunk',index,'consecutive failures',failures,type(exc).__name__,str(exc),flush=True)
                if failures>=12: raise
                time.sleep(min(failures+1,5))
        raise RuntimeError('Range retries exhausted')
    print(f'Downloading {name}: {size/2**20:.1f} MiB, {count} resumable chunks, {workers} workers',flush=True)
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    pending={pool.submit(fetch,i) for i in range(count)}
    try:
        while pending:
            done,pending=concurrent.futures.wait(pending,timeout=30,return_when=concurrent.futures.FIRST_EXCEPTION)
            for f in done: f.result()
            total=sum(p.stat().st_size for pattern in ('*.chunk','*.part') for p in chunks.glob(pattern))
            print(f'{name}: {total/size:.1%} ({total/2**20:.1f}/{size/2**20:.1f} MiB)',flush=True)
    finally:
        stop.set()
        for future in pending: future.cancel()
        pool.shutdown(wait=True,cancel_futures=True)
    assembled=root/(name+'.assembling')
    h=hashlib.sha256()
    with assembled.open('wb') as out:
        for index in range(count):
            with (chunks/f'{index:06d}.chunk').open('rb') as part:
                for block in iter(lambda:part.read(1024*1024),b''): out.write(block); h.update(block)
    if not matches(h.hexdigest()): raise ValueError('Full wheel SHA256 mismatch; chunks preserved')
    assembled.rename(target)
    (root/(name+'.metadata.json')).write_text(json.dumps({**state,'actual_sha256':h.hexdigest(),'hash_verification':'official prefix' if hash_prefix else 'full SHA256'},indent=2))
    for part in chunks.glob('*.chunk'): part.unlink()
    print('Official SHA256 prefix passed:' if hash_prefix else 'Full SHA256 passed:',name,flush=True)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--chunk-mib',type=int,default=8)
    args=parser.parse_args()
    if not 1<=args.workers<=16 or not 1<=args.chunk_mib<=16: parser.error('Invalid parallelism/chunk size')
    workspace=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve()
    root=workspace/'cache/downloads/pytorch'; root.mkdir(parents=True,exist_ok=True)
    if not root.resolve().is_relative_to(workspace): raise ValueError('Unsafe download destination')
    with (root/'.download.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for name,version in PACKAGES:
            filename,url,digest=metadata(name,version)
            fetch_wheel(root,filename,url,digest,args.workers,args.chunk_mib*2**20)
if __name__=='__main__': main()
