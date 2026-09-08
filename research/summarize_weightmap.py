# -*- coding: utf-8 -*-
"""把 safetensors index（最大 16 MB）壓成可版控的張量名稱摘要。

這份摘要就是文章裡「NVFP4 只量化了 routed expert」那個結論的直接證據：
帶 weight_scale 的張量才是被量化的。原始 index 用 fetch_more.py 重抓即可。
"""
import json, re, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = {}
for f in sorted(HERE.glob("*-model.safetensors.index.json")):
    idx = json.loads(f.read_text(encoding="utf-8"))
    keys = list(idx.get("weight_map", {}))
    pats = collections.Counter()
    for k in keys:
        p = re.sub(r"\.\d+\.", ".N.", k)
        p = re.sub(r"experts\.N\.", "experts.*.", p)
        pats[p] += 1
    tag = f.name.replace("-model.safetensors.index.json", "")
    OUT[tag] = {
        "metadata": idx.get("metadata", {}),
        "n_tensors": len(keys),
        "n_mtp_tensors": sum(1 for k in keys if "mtp" in k.lower()),
        "n_quantized_tensors": sum(1 for k in keys if k.endswith(("weight_scale",
                                                                 "weight_scale_inv"))),
        "tensor_name_patterns": dict(sorted(pats.items())),
    }
(HERE / "weightmap_summary.json").write_text(
    json.dumps(OUT, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote weightmap_summary.json for", len(OUT), "checkpoints")
