import urllib.request,json
from pathlib import Path
out=Path(__file__).resolve().parent
for repo in ['nvidia/Qwen3.5-122B-A10B-NVFP4','Qwen/Qwen3.8-27B-FP8','incoai/Qwen3.8-27B-DFlash2']:
    api=json.load(urllib.request.urlopen('https://huggingface.co/api/models/'+repo+'?blobs=true',timeout=60))
    summary={'repo':repo,'revision':api.get('sha'),'safetensors':api.get('safetensors'),'files':[{k:f.get(k) for k in ['rfilename','size','lfs']} for f in api.get('siblings',[]) if f['rfilename'].endswith('.safetensors')]}
    summary['weight_file_bytes']=sum(f.get('size') or (f.get('lfs') or {}).get('size',0) for f in summary['files'])
    for filename in ['config.json','hf_quant_config.json']:
        try:
            cfg=json.load(urllib.request.urlopen('https://huggingface.co/'+repo+'/resolve/'+api['sha']+'/'+filename,timeout=60))
            (out/(repo.replace('/','--')+'-'+filename)).write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
        except urllib.error.HTTPError as e:
            if e.code!=404:raise
    (out/(repo.replace('/','--')+'-metadata.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(repo,summary['revision'],summary['weight_file_bytes'])
