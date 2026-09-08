import urllib.request, urllib.error, json
from pathlib import Path
out = Path(__file__).resolve().parent

REPOS = [
 'nvidia/Qwen3.5-122B-A10B-NVFP4',
 'Qwen/Qwen3.8-27B-FP8',
 'incoai/Qwen3.8-27B-DFlash2',
 'Qwen/Qwen3.5-122B-A10B',
 'Qwen/Qwen3.8-27B',
 'Qwen/Qwen3-32B',
]
EXTRA = ['config.json','hf_quant_config.json','model.safetensors.index.json','generation_config.json']

def get(url):
    req = urllib.request.Request(url, headers={'User-Agent':'research/1.0'})
    return urllib.request.urlopen(req, timeout=90).read()

for repo in REPOS:
    tag = repo.replace('/','--')
    try:
        api = json.loads(get('https://huggingface.co/api/models/'+repo+'?blobs=true'))
    except Exception as e:
        print('API FAIL', repo, e); continue
    summary = {'repo':repo,'revision':api.get('sha'),'safetensors':api.get('safetensors'),
               'tags':api.get('tags'),'files':[]}
    for f in api.get('siblings',[]):
        if f['rfilename'].endswith('.safetensors'):
            summary['files'].append({'rfilename':f['rfilename'],'size':f.get('size'),'lfs':f.get('lfs')})
    summary['weight_file_bytes'] = sum((f.get('size') or (f.get('lfs') or {}).get('size',0)) for f in summary['files'])
    summary['n_weight_files'] = len(summary['files'])
    summary.pop('files')
    (out/(tag+'-meta2.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(repo, summary['revision'], summary['weight_file_bytes'], summary['n_weight_files'])
    for fn in EXTRA:
        p = out/(tag+'-'+fn)
        if p.exists(): continue
        try:
            b = get('https://huggingface.co/'+repo+'/resolve/'+api['sha']+'/'+fn)
            p.write_bytes(b); print('   got', fn, len(b))
        except urllib.error.HTTPError as e:
            print('   miss', fn, e.code)
        except Exception as e:
            print('   err', fn, e)
