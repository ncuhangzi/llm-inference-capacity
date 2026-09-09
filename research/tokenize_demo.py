# -*- coding: utf-8 -*-
"""用官方 tokenizer.json 的 vocab + merges 實作 byte-level BPE，產生真實的 token id。

不依賴 transformers / tokenizers，因為報告只需要一個可驗證的小例子。
輸出寫進 tokenize_demo.json，給投影片與部落格使用。
"""
import json, urllib.request, functools
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = "Qwen/Qwen3.8-27B"
CACHE = HERE / "Qwen--Qwen3.8-27B-tokenizer.json"


def fetch():
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    url = f"https://huggingface.co/{REPO}/resolve/main/tokenizer.json"
    req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    raw = urllib.request.urlopen(req, timeout=180).read()
    CACHE.write_bytes(raw)
    return json.loads(raw)


@functools.lru_cache(maxsize=1)
def byte_encoder():
    """GPT-2 的 byte → unicode 對應表。"""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def bpe(word, ranks):
    parts = list(word)
    while len(parts) > 1:
        pairs = [(ranks.get((parts[i], parts[i + 1]), 1 << 30), i)
                 for i in range(len(parts) - 1)]
        rank, i = min(pairs)
        if rank == 1 << 30:
            break
        parts[i:i + 2] = [parts[i] + parts[i + 1]]
    return parts


@functools.lru_cache(maxsize=1)
def byte_decoder():
    return {v: k for k, v in byte_encoder().items()}


def show(tok):
    """把 byte-level 的表示還原成看得懂的字。"""
    bd = byte_decoder()
    try:
        return bytes(bd[c] for c in tok).decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        return tok


def encode(text, vocab, ranks, pretok_re):
    import regex
    be = byte_encoder()
    ids = []
    for piece in regex.findall(pretok_re, text):
        w = "".join(be[b] for b in piece.encode("utf-8"))
        for tok in bpe(w, ranks):
            ids.append((tok, vocab[tok]))
    return ids


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    tj = fetch()
    vocab = tj["model"]["vocab"]
    merges = tj["model"]["merges"]
    if merges and isinstance(merges[0], str):
        merges = [tuple(m.split(" ")) for m in merges]
    else:
        merges = [tuple(m) for m in merges]
    ranks = {m: i for i, m in enumerate(merges)}
    print("vocab size:", len(vocab), " merges:", len(merges))

    pat = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
           r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
    demo = {}
    for name, text in [("zh", "量子電腦的錯誤修正是什麼？"),
                       ("en", "What is quantum error correction?")]:
        try:
            toks = encode(text, vocab, ranks, pat)
        except ImportError:
            print("需要 regex 套件：pip install regex"); raise SystemExit(1)
        demo[name] = {"text": text,
                      "tokens": [{"s": show(t), "raw": t, "id": i} for t, i in toks],
                      "n": len(toks), "n_chars": len(text)}
        print(f"\n{name}: {text}")
        print("  tokens:", " | ".join(show(t) for t, _ in toks))
        print("  ids   :", [i for _, i in toks])
        print(f"  {len(text)} 字 → {len(toks)} tokens")

    added = {t["content"]: t["id"] for t in tj.get("added_tokens", [])}
    demo["special"] = {k: added[k] for k in
                       ("<|im_start|>", "<|im_end|>", "<|endoftext|>") if k in added}
    demo["vocab_bpe"] = len(vocab)
    demo["vocab_added"] = len(added)
    demo["vocab_config"] = 248320          # config.json 的 vocab_size（有 padding）
    print("BPE vocab", len(vocab), "+ added", len(added),
          "=", len(vocab) + len(added), " → config 宣告", 248320,
          f"（padding {248320 - len(vocab) - len(added)} 個）")
    (HERE / "tokenize_demo.json").write_text(
        json.dumps(demo, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nspecial:", demo["special"])
    print("wrote tokenize_demo.json")
