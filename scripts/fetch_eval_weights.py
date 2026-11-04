#!/usr/bin/env python3
"""Resume official torchvision weights with bounded ranges; verify official hash prefix."""
import argparse,fcntl,hashlib,json,os,pathlib,re,urllib.parse,urllib.request
from fetch_pytorch import fetch_wheel,sha256

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--net',choices=['alex','vgg'],default='alex');args=ap.parse_args()
    import torch,torchvision.models as models
    root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();cache=pathlib.Path(torch.hub.get_dir())/'checkpoints'
    if not cache.resolve().is_relative_to(root):raise ValueError('Cache outside data volume')
    cache.mkdir(parents=True,exist_ok=True)
    url=(models.AlexNet_Weights.IMAGENET1K_V1 if args.net=='alex' else models.VGG16_Weights.IMAGENET1K_V1).url
    assert urllib.parse.urlsplit(url).hostname=='download.pytorch.org'
    name=url.rsplit('/',1)[-1];prefix=re.search(r'-([0-9a-f]{8})[.]pth$',name).group(1);chunk_size=8*2**20
    chunks=cache/(name+'.chunks');target=cache/name;partial=cache/(name+'.part')
    with (cache/(name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if partial.exists() and not chunks.exists() and not target.exists():
            req=urllib.request.Request(url+'?probe=0',headers={'Range':'bytes=0-0','Accept-Encoding':'identity'})
            with urllib.request.urlopen(req,timeout=30) as response:
                cr=response.headers.get('Content-Range','');match=re.fullmatch(r'bytes 0-0/(\d+)',cr)
                if response.status!=206 or not match or len(response.read(2))!=1:raise RuntimeError('Range probe failed')
                size=int(match.group(1))
            saved=partial.stat().st_size
            if saved>size:raise ValueError('Partial too large')
            chunks.mkdir();(chunks/'manifest.json').write_text(json.dumps({'url':url,'sha256':prefix,'size':size,'chunk_size':chunk_size},indent=2))
            with partial.open('rb') as inp:
                index=0;remaining=saved
                while remaining:
                    count=min(chunk_size,remaining);data=inp.read(count)
                    if len(data)!=count:raise IOError('Partial changed')
                    suffix='.chunk' if count==min(chunk_size,size-index*chunk_size) else '.part'
                    (chunks/(f'{index:06d}'+suffix)).write_bytes(data);remaining-=count;index+=1
            print('Preserved existing bytes:',saved,flush=True)
        fetch_wheel(cache,name,url,prefix,4,chunk_size,hash_prefix=True)
        metadata={'source':url,'official_sha256_prefix':prefix,'sha256':sha256(target),'bytes':target.stat().st_size}
        (cache/(name+'.provenance.json')).write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata),flush=True)
if __name__=='__main__':main()
