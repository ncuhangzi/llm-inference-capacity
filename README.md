# LLM 推論與服務容量

拆解 LLM 推論的成本結構，並回答一個實務問題：**這台機器到底能服務幾個人？**

對象是兩個要上線的模型 —— `nvidia/Qwen3.5-122B-A10B-NVFP4` 與 `Qwen/Qwen3.8-27B-FP8`，
跑在 vLLM 上、NVIDIA B200 180 GB。所有架構數字都由官方 config 逐張量推導，
再跟 HuggingFace 上 safetensors 的實際位元組統計對帳，誤差 < 0.02%。

---

## Quick start

不用安裝任何東西。三個檔案都是單一 HTML，完全離線，用 Chrome 或 Edge 打開就好。

| 我想… | 開這個 | 大小 |
| --- | --- | --- |
| **快速讀懂**（30–40 分鐘，長文＋可互動圖表） | [`LLM推論技術部落格.html`](LLM推論技術部落格.html) | 330 KB |
| **上台講**（82 頁投影片，逐步動畫、講稿、目錄） | [`LLM推論報告_v2.html`](LLM推論報告_v2.html) | 1.2 MB |
| **列印或傳給人** | [`LLM推論報告_v2.pdf`](LLM推論報告_v2.pdf) | 4.3 MB |

```bash
git clone https://github.com/ncuhangzi/llm-inference-capacity.git
cd llm-inference-capacity
start LLM推論技術部落格.html      # Windows
# open LLM推論技術部落格.html     # macOS
```

線上版（GitHub Pages）：

- 部落格 <https://ncuhangzi.github.io/llm-inference-capacity/LLM推論技術部落格.html>
- 簡報 <https://ncuhangzi.github.io/llm-inference-capacity/LLM推論報告_v2.html>

### 簡報的操作

| 按鍵 | 功能 |
| --- | --- |
| <kbd>→</kbd> <kbd>←</kbd> | 換頁（<kbd>Shift</kbd>+<kbd>→</kbd> 跳章） |
| <kbd>Space</kbd> | 動畫下一步；跑完才換頁 |
| <kbd>A</kbd> | 切換「進入每頁自動播放動畫」 |
| <kbd>P</kbd> / <kbd>R</kbd> | 重播 / 重設本頁動畫 |
| <kbd>N</kbd> <kbd>O</kbd> <kbd>G</kbd> | 講者筆記 / 目錄 / 名詞表 |
| <kbd>F</kbd> <kbd>?</kbd> | 全螢幕 / 操作說明 |

網址加 `#54` 可直接跳到第 54 頁。虛線底的粗體字滑過去會出現名詞解釋。

---

## 這份報告在講什麼

<table>
<tr><td width="50%">

**1. Decode 卡在頻寬，不是算力**

一次 forward 要把整份權重讀過一遍，卻只服務 batch 裡那幾個 token。
serving 上的所有最佳化，都是在提高「每次讀取分攤到的 token 數」。

</td><td width="50%">

**2. B200 的臨界值是 292 token/step，而且與精度無關**

`N* = F·b/(2·BW)`。Blackwell 每降半位寬就把算力加倍，
分子加倍、分母減半剛好相消 —— BF16 / FP8 / NVFP4 算出來都是 292。

</td></tr>
<tr><td>

**3. 混合架構要分開算兩種 cache**

122B 的 KV 只有 12 KiB/token（FP8），但多了每條序列固定 ~150 MiB 的 SSM state。
短對話時記憶體是被那個固定成本吃掉的，`--max-model-len` 調小沒用。

</td><td>

**4. NVFP4 只量化了 routed expert**

36 層 GDN、12 層 attention、lm_head、MTP head 全部還是 BF16，佔 checkpoint 的 21.9%
與 batch-1 頻寬的一半。這是從 safetensors 的 weight map 直接讀出來的。

</td></tr>
<tr><td colspan="2">

**5. Speculative decoding 的效益，由 `B × (k+1)` 有沒有超過 292 決定**

27B 配 DFlash2 猜 7 個 → 安全併發約 36，超過就開始賠。
而且接受率不是常數：實測顯示 context 從 2k 長到 30k，平均接受率從 65% 掉到 39%，
吞吐從「比不開快 129%」變成「比不開慢 51%」。

</td></tr>
</table>

---

## 數字的可信度

| 等級 | 涵蓋哪些 | 能不能引用 |
| --- | --- | --- |
| 實測 / 官方 | 架構參數、checkpoint 位元組數、量化設定、NVIDIA 的 NVFP4 精度評測、DFlash2 模型卡的接受長度、vLLM issue 的實測數字、B200 規格 | 可以直接引用 |
| 公式推導 | KV / SSM cache 大小、記憶體帳本、roofline 轉折點、speculative decoding 的加速估計 | 公式都印在頁面上，可以自行驗算。標成「解析上限」的，實測通常是它的 30–60% |
| 教學示例 | 吞吐 / 延遲 / 佇列曲線與 SLO 掃描圖，以排隊理論生成 | **不是任何硬體的實測結果**，不可當成 B200 實測成績引用 |

要拿去做正式的容量報告，得換成自家實測資料，並記錄引擎版本、workload、量測窗與 SLO 定義。

---

## 自己重跑

需要 Python 3.11+；要重新產生 PDF 與截圖才需要 Node 和 Playwright。

```bash
# 1. 抓官方 config 與 safetensors 索引（會寫進 research/）
python research/fetch_more.py
python research/summarize_weightmap.py

# 2. 解析模型的自我檢查：參數量、cache、roofline、speculative decoding
python v2/model.py

# 3. 重新產生兩份文件
python v2/blog.py      # → LLM推論技術部落格.html
python v2/build.py     # → LLM推論報告_v2.html + 講者筆記_v2.md

# 4.（選用）逐頁截圖、版面溢出檢查、輸出 PDF
node v2/qa.cjs --pdf
node v2/interact.cjs   # 27 項互動與數值一致性測試
```

`python v2/model.py` 會印出所有推導結果，並跟 checkpoint 的實際位元組數比對：

```
roofline knee (precision-invariant): bf16=292.2 fp8=292.2 nvfp4=292.2 tokens/step

=== Qwen3.5-122B-A10B [NVFP4]
  params analytic 125,074,705,904  vs real 125,086,503,920  ratio 0.99991
  weight bytes analytic 83.45 GB  real 83.47 GB = 77.74 GiB
  KV/token [fp8] 12 KiB   @262144 ctx = 3.00 GiB
  SSM recurrent 144.0 MiB   conv(k=3) 5.06 MiB
  ...
```

---

## 專案結構

```
LLM推論技術部落格.html      長文版（單檔、離線可用）
LLM推論報告_v2.html         簡報版 82 頁
LLM推論報告_v2.pdf          簡報的靜態版
講者筆記_v2.md              每頁講稿與來源
使用說明.md                 中文的詳細說明

v2/
  model.py                 解析模型：參數量、cache 公式、roofline、spec decoding
                           ── 所有數字的單一來源，投影片上沒有手打的常數
  svgkit.py                極小的 SVG 產生器
  figs_core.py             CH1–CH2 的圖：推論迴圈、KV cache、roofline
  figs_arch.py             CH3–CH4 的圖：vLLM 堆疊、架構、MoE、Gated DeltaNet
  figs_spec.py             CH5–CH7 的圖：量化位元佈局、weight map、speculative decoding
  figs_load.py             CH8–CH10 的圖：延遲、負載測試、決策樹
  widgets.js               6 個互動試算器
  common.py                來源清單、名詞表（47 條）、版面工具
  template.html            簡報的樣式與播放引擎（1440×810 固定畫布）
  blog_template.html       部落格的樣式與播放引擎
  slides_part1..4.py       簡報內容
  blog_body.py             部落格內容
  assemble.py / build.py   組裝簡報
  blog.py                  組裝部落格
  humanize*.py             文字修潤的改寫表（一次性腳本，已套用）
  qa.cjs / interact.cjs    版面溢出檢查、PDF 輸出、27 項互動測試

research/                  官方 config、safetensors 統計、weight map 摘要
qa2/                       QA 結果（截圖用 .gitignore 排除，重跑 qa.cjs 會回來）
archive/v1/                第一版（ChatGPT 產出），保留供對照
archive/dark-v1/           更早的深色版本
```

改任何架構參數只要動 `v2/model.py`，重跑 build 之後，內文、表格、圖與互動試算器會一起更新。

---

## 評估對象

| 用途 | Repo | 量化 | Speculative decoding |
| --- | --- | --- | --- |
| 大模型 | `nvidia/Qwen3.5-122B-A10B-NVFP4` | NVFP4（W4A4）+ FP8 KV | 內建 MTP，k = 3 |
| 中模型 | `Qwen/Qwen3.8-27B-FP8` | FP8 blockwise-128 | `incoai/Qwen3.8-27B-DFlash2`，k = 7 |
| 對照 | `Qwen/Qwen3-32B` | BF16 | 無（checkpoint 沒有 MTP head） |

checkpoint 的 revision hash 列在文件的附錄裡。資料查核日期 2026-09-08。

## 主要參考

vLLM 官方文件與部落格、Gated Delta Networks（arXiv 2412.06464）、
DFlash（arXiv 2602.06036）、EAGLE-3（arXiv 2503.01840）、
SmartSpec（arXiv 2406.14066）、NVIDIA 與 Qwen 的模型卡、NVIDIA HGX B200 規格表。
完整的 27 筆來源與連結在部落格文末的參考文獻。

## 授權

程式碼（`v2/`、`research/` 的腳本）採 MIT。文章與圖表採
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.zh-hant)，
轉載請標明出處。引用到的第三方 benchmark 數字，版權屬原作者。
