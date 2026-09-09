/* ===================================================== interactive 試算 ==
   每個 widget 都直接使用 MODELS / HW（與投影片上的靜態數字同源）。          */
const G=1073741824, Mi=1048576, Ki=1024;
const f1=(v)=>v.toLocaleString('en-US',{maximumFractionDigits:1});
const f2=(v)=>v.toLocaleString('en-US',{maximumFractionDigits:2});
const f0=(v)=>Math.round(v).toLocaleString('en-US');
const KVB={bf16:2,fp8:1};

function kvPerTok(m,d){return 2*m.nFull*m.nKv*m.headDim*KVB[d];}
function ssmRec(m){return m.nGdn*m.nv*m.dv*m.dk*4;}
function ssmConv(m,k){return m.nGdn*m.convDim*(m.convK-1+k)*2;}
function ssmAll(m,k){return ssmRec(m)+ssmConv(m,k);}
function seqBytes(m,tok,d,k){return tok*kvPerTok(m,d)+ssmAll(m,k);}
function expertsTouched(m,n){if(!m.nExperts)return 0;
  return m.nExperts*(1-Math.pow(1-m.topK/m.nExperts,n));}
function stepBytes(m,n){return m.denseRead+expertsTouched(m,n)*m.L*m.expertBytes;}
function stepTime(m,n){return Math.max(stepBytes(m,n)/HW.bw,n*m.tokenCompute);}
function draftBytes(m,B,k){
  if(!m.specK)return 0;
  if(m.draftSerial)   /* MTP：k 次 autoregressive，每次都要重讀 lm_head 與被選到的 expert */
    return k*(m.mtpDenseParams+expertsTouched(m,B)*m.expertParams)*2;
  return m.draftBytes;   /* DFlash2：獨立 drafter，整塊一次算完 */
}
function draftCompute(m,B,k){
  if(!m.specK)return 0;
  if(m.draftSerial)
    return k*B*2*(m.mtpDenseParams+m.topK*m.expertParams)/HW.flops.bf16;
  return B*(k+1)*2*(m.draftBytes/2)/HW.flops.bf16*0.83;
}
function specIter(m,B,k,tau){
  const vn=B*(k+1);
  const tV=Math.max(stepBytes(m,vn)/HW.bw, vn*m.tokenCompute);
  const tD=Math.max(draftBytes(m,B,k)/HW.bw,draftCompute(m,B,k));
  const base=stepTime(m,B);
  const t=tV+tD, tok=B*Math.min(tau,k+1);
  return {tV,tD,t,tok,base,tps:tok/t,tpsBase:B/base,
          tpot:t/Math.min(tau,k+1),tpotBase:base,
          bound:(vn*m.tokenCompute>stepBytes(m,vn)/HW.bw)?'compute':'memory'};
}
function tauOf(a,k){return a>=1?k+1:(1-Math.pow(a,k+1))/(1-a);}

/* ---- 控制項小工具 ------------------------------------------------------ */
function ctl(host,spec,onchange){
  const box=document.createElement('div'); box.className='ctl';
  const st={};
  for(const s of spec){
    if(s.type==='seg'){
      const w=document.createElement('label'); w.innerHTML=`<span>${s.label}</span>`;
      const seg=document.createElement('span'); seg.className='seg';
      s.opts.forEach(([v,t])=>{const b=document.createElement('button');
        b.textContent=t; b.dataset.v=v;
        b.className=(v==s.value?'on':'');
        b.onclick=()=>{st[s.key]=v;[...seg.children].forEach(c=>c.classList.toggle('on',c.dataset.v==v));onchange(st)};
        seg.appendChild(b);});
      w.appendChild(seg); box.appendChild(w); st[s.key]=s.value;
    } else {
      const w=document.createElement('label');
      const cap=document.createElement('span'); cap.innerHTML=`${s.label} <b class="lv"></b>`;
      const inp=document.createElement('input'); inp.type='range';
      inp.min=s.min; inp.max=s.max; inp.step=s.step||1; inp.value=s.value;
      const upd=()=>{st[s.key]=+inp.value;
        cap.querySelector('.lv').textContent=s.fmt?s.fmt(+inp.value):inp.value;};
      inp.oninput=()=>{upd();onchange(st)};
      w.appendChild(cap); w.appendChild(inp); box.appendChild(w); st[s.key]=+s.value; upd();
    }
  }
  host.appendChild(box);
  return st;
}
function panelHTML(){const d=document.createElement('div');return d;}

const WIDGETS={
/* ================================================= 1. 單卡記憶體帳本 ==== */
budget(el){
  const out=document.createElement('div');
  const spec=[
    {type:'seg',key:'m',label:'模型',value:'q35',
     opts:[['q35','122B-A10B NVFP4'],['q38','27B FP8'],['q332','32B BF16']]},
    {type:'seg',key:'kv',label:'KV cache 精度',value:'fp8',opts:[['fp8','FP8'],['bf16','BF16']]},
    {type:'seg',key:'sd',label:'speculative decoding',value:'1',opts:[['1','開'],['0','關']]},
    {key:'ctx',label:'每條序列 context',min:10,max:18,step:1,value:13,
     fmt:v=>f0(Math.pow(2,v))+' tok'},
    {key:'seq',label:'同時併發序列',min:1,max:512,step:1,value:32,fmt:v=>v},
    {key:'util',label:'gpu_memory_utilization',min:70,max:98,step:1,value:90,fmt:v=>(v/100).toFixed(2)},
  ];
  const st=ctl(el,spec,render); el.appendChild(out);
  function render(){
    const m=MODELS[st.m], ctx=Math.min(Math.pow(2,st.ctx),m.maxCtx), sd=st.sd==='1'&&m.specK>0;
    const k=sd?m.specK:0;
    const total=HW.hbm/G, usable=total*st.util/100;
    const w=m.weightBytes/G, dr=sd?m.draftBytes/G:0;
    const act=Math.max(3, 1.2+st.seq*0.02);
    const kv=st.seq*ctx*kvPerTok(m,st.kv)/G, ss=st.seq*ssmAll(m,k)/G;
    const used=w+dr+act+kv+ss, free=usable-used;
    const rows=[
      ['模型權重',[[w,'compute']],f2(w)+' GiB'],
      ...(sd?[['起草器權重',[[dr,'spec']],f2(dr)+' GiB']]:[]),
      ['KV cache',[[kv,'mem']],f2(kv)+' GiB'],
      ['SSM / conv state',[[ss,'ssm']],f2(ss)+' GiB'],
      ['activation 等其他',[[act,'neutral']],f2(act)+' GiB'],
      [free>=0?'剩餘可用':'⚠ 超出預算',[[Math.abs(free),free>=0?'ok':'bad']],
       (free>=0?'':'−')+f2(Math.abs(free))+' GiB'],
    ];
    const bar=(lab,segs,disp)=>`<div class="bar"><span class="t">${lab}</span>
      <span class="track">${segs.map(([v,c])=>`<i style="width:${Math.min(100,v/usable*100)}%;background:var(--${c})"></i>`).join('')}</span>
      <span class="v">${disp}</span></div>`;
    const perSeq=seqBytes(m,ctx,st.kv,k);
    const maxSeq=Math.max(0,Math.floor((usable-w-dr-act)/(perSeq/G)));
    out.innerHTML=
      `<div class="bars" style="margin-top:7px">${rows.map(r=>bar(...r)).join('')}</div>
       <div class="stats" style="margin-top:9px">
        <div class="stat b"><div class="v">${f1(usable)}<small>GiB</small></div>
          <div class="k">可配置額度 = 180 GB × ${(st.util/100).toFixed(2)}</div></div>
        <div class="stat m"><div class="v">${f1(perSeq/Mi)}<small>MiB</small></div>
          <div class="k">每條序列 @ ${f0(ctx)} token（KV ${f1(ctx*kvPerTok(m,st.kv)/Mi)} + state ${f1(ssmAll(m,k)/Mi)}）</div></div>
        <div class="stat ${free>=0?'g':'p'}"><div class="v">${f0(maxSeq)}</div>
          <div class="k">此 context 下單卡可容納的序列數上限</div></div>
        <div class="stat s"><div class="v">${f0(ssmAll(m,k)/kvPerTok(m,st.kv))}<small>tok</small></div>
          <div class="k">SSM state 相當於幾個 token 的 KV cache</div></div>
       </div>
       <div class="hint" style="margin-top:6px;font-size:11.6px">activation 估計值僅為教學用途；實際由 vLLM 依 max_num_batched_tokens、CUDA graph 捕捉的 batch 尺寸與 kernel workspace 決定。權重採用 HuggingFace 上實際 safetensors 位元組數。</div>`;
  }
  render();
  return {print:render};
},

/* ============================================ 2. 兩種 cache 的成長曲線 == */
cache(el){
  const out=document.createElement('div');
  const st=ctl(el,[
    {type:'seg',key:'kv',label:'KV 精度',value:'fp8',opts:[['fp8','FP8'],['bf16','BF16']]},
    {key:'ctx',label:'context 長度',min:9,max:18,step:1,value:13,fmt:v=>f0(Math.pow(2,v))+' tok'},
  ],render); el.appendChild(out);
  function render(){
    const ctx=Math.pow(2,st.ctx);
    const keys=['q332','q38','q35'];
    const rows=keys.map(k=>{
      const m=MODELS[k]; const c=Math.min(ctx,m.maxCtx);
      const kv=c*kvPerTok(m,st.kv), ss=ssmAll(m,m.specK);
      return {m,kv,ss,tot:kv+ss,clip:c<ctx};
    });
    const tot=Math.max(...rows.map(r=>r.tot));
    out.innerHTML=`<div class="bars" style="margin-top:10px">${rows.map(r=>
      `<div class="bar"><span class="t">${r.m.short} <span class="tagc n">${r.m.quant}</span></span>
       <span class="track">
        <i style="width:${r.kv/tot*100}%;background:var(--mem)"></i>
        <i style="width:${r.ss/tot*100}%;background:var(--ssm)"></i></span>
       <span class="v">${f1(r.tot/Mi)} MiB</span></div>`).join('')}</div>
      <div class="legend" style="margin-top:9px">
       <span><i style="background:var(--mem)"></i>KV cache（隨 context 線性成長）</span>
       <span><i style="background:var(--ssm)"></i>SSM + conv state（固定）</span></div>
      <table class="compact" style="margin-top:10px"><thead><tr><th>模型</th>
       <th class="n">KV / token</th><th class="n">KV @ ${f0(ctx)}</th>
       <th class="n">SSM state</th><th class="n">合計 / 序列</th><th class="n">SSM = 幾個 token 的 KV</th></tr></thead><tbody>
       ${rows.map(r=>`<tr><td>${r.m.label}${r.clip?' <span class="tagc n">超過原生 context，已截至上限</span>':''}</td>
        <td class="n">${f0(kvPerTok(r.m,st.kv)/Ki)} KiB</td>
        <td class="n">${f1(r.kv/Mi)} MiB</td>
        <td class="n">${r.ss?f1(r.ss/Mi)+' MiB':'—'}</td>
        <td class="n"><b>${f1(r.tot/Mi)} MiB</b></td>
        <td class="n">${r.ss?f0(r.ss/kvPerTok(r.m,st.kv)):'—'}</td></tr>`).join('')}
      </tbody></table>`;
  }
  render(); return {print:render};
},

/* ================================================== 3. roofline 互動 ==== */
roofline(el){
  const out=document.createElement('div');
  const st=ctl(el,[
    {type:'seg',key:'m',label:'模型',value:'q38',
     opts:[['q38','27B FP8'],['q35','122B NVFP4'],['q332','32B BF16']]},
    {key:'B',label:'同時併發序列 B',min:1,max:256,step:1,value:32,fmt:v=>v},
    {key:'k',label:'speculative 草稿數 k',min:0,max:15,step:1,value:0,fmt:v=>v===0?'關閉':v},
  ],render); el.appendChild(out);
  function render(){
    const m=MODELS[st.m], N=st.B*(st.k+1);
    const tMem=stepBytes(m,N)/HW.bw, tCmp=N*m.tokenCompute, t=Math.max(tMem,tCmp);
    const bound=tCmp>tMem?'compute-bound':'memory-bound';
    const util=Math.min(tMem,tCmp)/Math.max(tMem,tCmp);
    const tau=st.k?tauOf(m.alpha||0.75,st.k):1;
    const eff=st.k?specIter(m,st.B,st.k,tau):null;
    out.innerHTML=`<div class="stats" style="margin-top:10px">
      <div class="stat b"><div class="v">${f0(N)}</div><div class="k">本輪 forward 的 token 數 = B × (k+1)</div></div>
      <div class="stat ${bound==='compute-bound'?'m':'g'}"><div class="v" style="font-size:22px">${bound}</div>
        <div class="k">轉折點 N* ≈ ${f0(HW.knee)}（此模型實際 ${f0(st.m==='q35'?1765:292)}）</div></div>
      <div class="stat s"><div class="v">${f2(tMem*1000)}<small>ms</small></div>
        <div class="k">讀權重時間：${f1(stepBytes(m,N)/1e9)} GB ÷ 7.7 TB/s</div></div>
      <div class="stat e"><div class="v">${f2(tCmp*1000)}<small>ms</small></div>
        <div class="k">張量運算時間（B200 dense 峰值）</div></div>
      </div>
      <div class="bars" style="margin-top:12px">
       <div class="bar"><span class="t">頻寬佔用</span><span class="track">
        <i style="width:${tMem/t*100}%;background:var(--mem)"></i></span>
        <span class="v">${f0(tMem/t*100)}%</span></div>
       <div class="bar"><span class="t">算力佔用</span><span class="track">
        <i style="width:${tCmp/t*100}%;background:var(--compute)"></i></span>
        <span class="v">${f0(tCmp/t*100)}%</span></div>
      </div>
      <div class="stats" style="margin-top:12px">
       <div class="stat g"><div class="v">${f0(eff?eff.tps:st.B/t)}<small>tok/s</small></div>
         <div class="k">整機輸出（解析上限）</div></div>
       <div class="stat p"><div class="v">${f2((eff?eff.tpot:t)*1000)}<small>ms</small></div>
         <div class="k">每位使用者的 TPOT（解析上限）</div></div>
       <div class="stat ${st.k?'p':'n'}"><div class="v">${st.k?f2(eff.tps/eff.tpsBase)+'×':'—'}</div>
         <div class="k">${st.k?`相對不用 spec 的加速（τ=${f2(tau)}）`:'未啟用 speculative decoding'}</div></div>
       <div class="stat n"><div class="v">${f0(util*100)}<small>%</small></div>
         <div class="k">較閒的那一側被用掉多少（越低越浪費）</div></div>
      </div>
      <div class="hint" style="margin-top:8px">這是 roofline 解析上限：只算「讀權重」與「張量運算」。
       實測還要加上 attention kernel、routing / all-to-all、取樣、Python 排程與 CUDA graph 之外的開銷，
       常見落在上限的 30–60%。</div>`;
  }
  render(); return {print:render};
},

/* ============================================ 4. speculative decoding == */
spec(el){
  const out=document.createElement('div');
  const st=ctl(el,[
    {type:'seg',key:'m',label:'模型 / 起草器',value:'q38',
     opts:[['q38','27B + DFlash2'],['q35','122B + MTP']]},
    {key:'k',label:'草稿 token 數 k',min:1,max:15,step:1,value:7,fmt:v=>v},
    {key:'a',label:'逐位置接受率 α',min:40,max:95,step:1,value:80,fmt:v=>(v/100).toFixed(2)},
    {key:'B',label:'同時併發序列 B',min:1,max:256,step:1,value:16,fmt:v=>v},
  ],render); el.appendChild(out);
  function render(){
    const m=MODELS[st.m], a=st.a/100, tau=tauOf(a,st.k);
    const r=specIter(m,st.B,st.k,tau);
    const sp=r.tps/r.tpsBase;
    // 找出 break-even 的 B
    let be=null; for(let B=1;B<=1024;B++){const x=specIter(m,B,st.k,tau);
      if(x.tps/x.tpsBase<1){be=B;break;}}
    const positions=Array.from({length:st.k},(_,i)=>Math.pow(a,i+1));
    out.innerHTML=`<div class="stats" style="margin-top:10px">
      <div class="stat p"><div class="v">${f2(tau)}<small>tok</small></div>
        <div class="k">期望接受長度 τ = (1−α^(k+1))/(1−α)，上限 ${st.k+1}</div></div>
      <div class="stat ${sp>=1?'g':'m'}"><div class="v">${f2(sp)}×</div>
        <div class="k">相對不用 spec 的輸出速率</div></div>
      <div class="stat b"><div class="v">${f2(r.tpot*1000)}<small>ms</small></div>
        <div class="k">TPOT：${f2(r.tpotBase*1000)} ms → ${f2(r.tpot*1000)} ms</div></div>
      <div class="stat ${be&&st.B>=be?'m':'n'}"><div class="v">${be?f0(be):'>1024'}</div>
        <div class="k">損益兩平的併發數（超過就別開）</div></div>
     </div>
     <div class="bars" style="margin-top:12px">
      <div class="bar"><span class="t">起草 draft</span><span class="track">
        <i style="width:${r.tD/r.t*100}%;background:var(--spec)"></i></span>
        <span class="v">${f2(r.tD*1000)} ms</span></div>
      <div class="bar"><span class="t">驗證 verify</span><span class="track">
        <i style="width:${r.tV/r.t*100}%;background:var(--compute)"></i></span>
        <span class="v">${f2(r.tV*1000)} ms</span></div>
      <div class="bar"><span class="t">驗證階段瓶頸</span><span class="track">
        <i style="width:100%;background:var(--${r.bound==='compute'?'mem':'ok'})"></i></span>
        <span class="v">${r.bound==='compute'?'算力':'頻寬'}</span></div>
     </div>
     <div style="margin-top:12px;display:flex;gap:5px;align-items:flex-end;height:56px">
      ${positions.map((p,i)=>`<div style="flex:1;display:flex;flex-direction:column;
        align-items:center;gap:3px">
        <span class="hint" style="font-size:10px">${(p*100).toFixed(0)}%</span>
        <div style="width:100%;height:${p*34}px;background:var(--spec);opacity:${0.35+p*0.65};
          border-radius:3px 3px 0 0"></div>
        <span class="hint" style="font-size:10px">+${i+1}</span></div>`).join('')}
      <div style="flex:2;padding-left:12px" class="hint">第 i 個草稿位置能被接受的機率是 α<sup>i</sup>：
       越後面越難中，所以 k 加大到某個點之後 τ 幾乎不再成長，但驗證成本仍線性上升。</div>
     </div>`;
  }
  render(); return {print:render};
},

/* ================================ 4b. 高併發模擬：max_num_seqs 的取捨 ==== */
serve(el){
  const out=document.createElement('div');
  const st=ctl(el,[
    {type:'seg',key:'m',label:'模型',value:'q38',
     opts:[['q38','27B FP8'],['q35','122B NVFP4'],['q332','32B BF16']]},
    {type:'seg',key:'sd',label:'speculative',value:'0',opts:[['0','關'],['1','開']]},
    {key:'users',label:'同時上門的人數',min:8,max:512,step:8,value:128,fmt:v=>v},
    {key:'cap',label:'max_num_seqs',min:8,max:512,step:8,value:48,fmt:v=>v},
    {key:'inl',label:'平均輸入長度',min:256,max:16384,step:256,value:2048,fmt:v=>f0(v)},
    {key:'outl',label:'平均輸出長度',min:64,max:2048,step:64,value:512,fmt:v=>f0(v)},
  ],render); el.appendChild(out);

  function sim(m,users,cap,inl,outl,k,tau){
    const B=Math.max(1,Math.min(users,cap)), queued=Math.max(0,users-B);
    const need=B*tau*inl/outl;
    const P=Math.min(need,Math.max(0,8192-B*(k+1)));
    const n=B*(k+1)+P;
    const tMem=stepBytes(m,n)/HW.bw, tCmp=n*m.tokenCompute, t=Math.max(tMem,tCmp);
    const tpot=t/tau, dec=outl*tpot, pre=(inl/Math.max(P,1e-9))*t;
    const svc=pre+dec, rate=B/svc, q=queued/rate;
    return {B,queued,n,t,tMem,tCmp,bound:tCmp>tMem?'算力':'頻寬',
            tpot:tpot*1e3,ttft:(q+pre)*1e3,queue:q*1e3,pre:pre*1e3,
            e2e:q+pre+dec,tps:B*tau/t,rate};
  }
  function render(){
    const m=MODELS[st.m], sd=st.sd==='1'&&m.specK>0;
    const k=sd?m.specK:0, tau=sd?m.tau:1;
    const r=sim(m,st.users,st.cap,st.inl,st.outl,k,tau);
    const cands=[8,16,24,32,48,64,96,128,192,256];
    const rows=cands.map(c=>({c,...sim(m,st.users,c,st.inl,st.outl,k,tau)}));
    const mxT=Math.max(...rows.map(x=>x.ttft)), mxP=Math.max(...rows.map(x=>x.tpot));
    const mxG=Math.max(...rows.map(x=>x.tps));
    out.innerHTML=`
     <div class="stats" style="margin-top:10px">
      <div class="stat b"><div class="v">${f0(r.B)}<small>/ ${f0(st.users)}</small></div>
        <div class="k">同時在跑 / 上門人數　排隊 ${f0(r.queued)} 人</div></div>
      <div class="stat ${r.bound==='算力'?'m':'g'}"><div class="v" style="font-size:21px">${f0(r.n)} tok/step</div>
        <div class="k">瓶頸在<b>${r.bound}</b>（N* ≈ ${f0(HW.knee)}）</div></div>
      <div class="stat s"><div class="v">${f2(r.tpot)}<small>ms</small></div>
        <div class="k">ITL / TPOT：使用者看到的打字速度</div></div>
      <div class="stat p"><div class="v">${r.ttft>=1000?f2(r.ttft/1000)+'s':f0(r.ttft)+'ms'}</div>
        <div class="k">TTFT＝排隊 ${f0(r.queue)} ms ＋ prefill ${f0(r.pre)} ms</div></div>
     </div>
     <div class="stats" style="margin-top:12px">
      <div class="stat e"><div class="v">${f2(r.e2e)}<small>s</small></div>
        <div class="k">E2E：使用者實際等待的總時間</div></div>
      <div class="stat g"><div class="v">${f0(r.tps)}<small>tok/s</small></div>
        <div class="k">整機輸出（解析上限）</div></div>
      <div class="stat n"><div class="v">${f2(r.rate)}<small>req/s</small></div>
        <div class="k">穩定態完成率</div></div>
      <div class="stat n"><div class="v">${f0(r.tMem/r.t*100)}<small>%</small></div>
        <div class="k">頻寬佔用（另一側是算力 ${f0(r.tCmp/r.t*100)}%）</div></div>
     </div>
     <div style="margin-top:14px">
      <div class="legend" style="margin-bottom:7px">
       <span><i style="background:var(--compute)"></i>TTFT</span>
       <span><i style="background:var(--mem)"></i>ITL / TPOT</span>
       <span><i style="background:var(--ok)"></i>整機 tok/s</span>
       <span style="color:var(--mut2)">灰底 = 目前選的 max_num_seqs</span></div>
      <div style="display:flex;gap:4px;align-items:flex-end;height:120px">
       ${rows.map(x=>`<div style="flex:1;display:flex;flex-direction:column;
          justify-content:flex-end;gap:2px;height:100%;
          background:${x.c===st.cap?'#eef1f5':'transparent'};border-radius:5px;padding:3px 2px">
         <div style="display:flex;gap:2px;align-items:flex-end;height:88px">
          <div style="flex:1;height:${x.ttft/mxT*100}%;background:var(--compute);border-radius:2px 2px 0 0"></div>
          <div style="flex:1;height:${x.tpot/mxP*100}%;background:var(--mem);border-radius:2px 2px 0 0"></div>
          <div style="flex:1;height:${x.tps/mxG*100}%;background:var(--ok);border-radius:2px 2px 0 0"></div>
         </div>
         <span class="hint" style="font-size:10px;text-align:center">${x.c}</span></div>`).join('')}
      </div>
     </div>
     <div class="hint" style="margin-top:9px">封閉式穩定態近似：running = min(上門人數, max_num_seqs)，
      其餘排隊；每個 step 撥給 prefill 的 token 數由「進來的人 = 出去的人」決定；
      step 時間走 roofline。沒有擬合參數，每個數字都可以手算驗證，但也不含 attention kernel、
      排程與通訊開銷，實測會更慢。</div>`;
  }
  render(); return {print:render};
},

/* ======================================= 5. RPS → 服務人數（Little's Law）*/
little(el){
  const out=document.createElement('div');
  const st=ctl(el,[
    {key:'tps',label:'整機 goodput（實測 tok/s）',min:200,max:20000,step:100,value:6000,fmt:v=>f0(v)},
    {key:'out',label:'平均輸出長度',min:64,max:4096,step:64,value:512,fmt:v=>f0(v)+' tok'},
    {key:'w',label:'平均 E2E 延遲（實測）',min:1,max:120,step:1,value:22,fmt:v=>v+' s'},
    {key:'turns',label:'每人每小時對話輪數',min:1,max:120,step:1,value:20,fmt:v=>v},
  ],render); el.appendChild(out);
  function render(){
    const lam=st.tps/st.out;                 // λ：每秒完成的 request
    const L=lam*st.w;                        // Little's Law：系統內平均 request 數
    const perUser=st.turns/3600;
    const users=lam/perUser;
    out.innerHTML=`<div class="stats" style="margin-top:10px">
      <div class="stat b"><div class="v">${f2(lam)}<small>req/s</small></div>
        <div class="k">λ = goodput ÷ 平均輸出長度</div></div>
      <div class="stat s"><div class="v">${f1(L)}</div>
        <div class="k">L = λ × W：平均同時在系統內的 request 數<br>→ 對照 max_num_seqs 是否夠</div></div>
      <div class="stat m"><div class="v">${f0(1/perUser)}<small>s</small></div>
        <div class="k">每位使用者平均隔多久發一次</div></div>
      <div class="stat g"><div class="v">${f0(users)}</div>
        <div class="k">可服務人數（每人 ${st.turns} 輪/小時）</div></div>
     </div>
     <div class="formula" style="margin-top:12px">
      λ = <em>${f0(st.tps)}</em> ÷ <em>${st.out}</em> = <em>${f2(lam)}</em> req/s　·　
      L = λ·W = <span class="c3">${f1(L)}</span>　·　
      人數 = λ ÷ (${st.turns}/3600) = <span class="c2">${f0(users)}</span></div>
     <div class="hint" style="margin-top:8px">前提：系統穩定（到達率 &lt; 服務率、佇列不成長）、
      輸出長度分布與量測窗一致、在該吞吐下 SLO 仍然滿足。任何一項不成立，這個人數就不能引用。
      注意 goodput 要用「通過 SLO」的部分，不是原始 throughput。</div>`;
  }
  render(); return {print:render};
},

/* ================================== 6. 兩階段負載測試（教學示例曲線）=== */
sweep(el){ return sweepImpl(el,'closed'); },
sweep2(el){ return sweepImpl(el,'open'); },
};
function sweepImpl(el,mode0){
  const out=document.createElement('div');
  const st=ctl(el,[
    {type:'seg',key:'mode',label:'測試型態',value:mode0,
     opts:[['closed','Stage 1 閉環（控併發）'],['open','Stage 2 開環（控到達率）']]},
    {key:'x',label:'控制變數',min:1,max:40,step:1,value:12,fmt:v=>v},
  ],render); el.appendChild(out);
  const CAP=14;   // 教學示例：此設定的 SLO 容量約 14
  function render(){
    const x=st.x, closed=st.mode==='closed';
    const pts=[];
    for(let i=1;i<=40;i++){
      let tps,lat;
      if(closed){ tps=1400*(1-Math.exp(-i/9)); lat=40+i*i*0.55; }
      else { const rho=Math.min(0.995,i/CAP*0.93);
             tps=Math.min(i,CAP*1.02)*100; lat=45/(1-rho); }
      pts.push({i,tps,lat});
    }
    const cur=pts[x-1];
    const H=138,Wd=560,pad=34;
    const mxT=Math.max(...pts.map(p=>p.tps)), mxL=Math.min(2200,Math.max(...pts.map(p=>p.lat)));
    const px=i=>pad+(i-1)/39*(Wd-pad-8);
    const pyT=v=>H-8-v/mxT*(H-26), pyL=v=>H-8-Math.min(v,mxL)/mxL*(H-26);
    const line=(f,c,w)=>`<path d="M${pts.map(p=>`${px(p.i).toFixed(1)},${f(p).toFixed(1)}`).join(' L')}"
        fill="none" stroke="var(--${c})" stroke-width="${w}" stroke-linejoin="round"/>`;
    out.innerHTML=`
     <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:10px">
      <div class="pane"><h3>吞吐</h3><svg viewBox="0 0 ${Wd} ${H}" style="width:100%">
        ${line(p=>pyT(p.tps),'compute',2.2)}
        <line x1="${px(x)}" y1="8" x2="${px(x)}" y2="${H-8}" stroke="var(--mut2)"
          stroke-width="1" stroke-dasharray="4 4"/>
        <circle cx="${px(x)}" cy="${pyT(cur.tps)}" r="4.5" fill="var(--compute)"/>
        <line x1="${pad}" y1="${H-8}" x2="${Wd-4}" y2="${H-8}" stroke="var(--line)"/>
      </svg><div class="hint">${closed?'併發拉高，吞吐先線性、後飽和':'到達率低於容量時吞吐 = 到達率；超過就被削平'}</div></div>
      <div class="pane"><h3>p95 延遲</h3><svg viewBox="0 0 ${Wd} ${H}" style="width:100%">
        ${line(p=>pyL(p.lat),'mem',2.2)}
        ${closed?'':`<line x1="${px(CAP)}" y1="8" x2="${px(CAP)}" y2="${H-8}"
          stroke="var(--bad)" stroke-width="1.4" stroke-dasharray="5 4"/>`}
        <line x1="${px(x)}" y1="8" x2="${px(x)}" y2="${H-8}" stroke="var(--mut2)"
          stroke-width="1" stroke-dasharray="4 4"/>
        <circle cx="${px(x)}" cy="${pyL(cur.lat)}" r="4.5" fill="var(--mem)"/>
        <line x1="${pad}" y1="${H-8}" x2="${Wd-4}" y2="${H-8}" stroke="var(--line)"/>
      </svg><div class="hint">${closed?'延遲隨併發單調上升，不會爆掉，所以閉環量不到容量'
        :'接近容量時延遲以 1/(1−ρ) 爆炸，紅線就是 SLO 邊界'}</div></div>
     </div>
     <div class="stats" style="margin-top:12px">
      <div class="stat b"><div class="v">${f0(cur.tps)}<small>tok/s</small></div>
        <div class="k">${closed?`併發 = ${x}`:`到達率 = ${x} req/s`}</div></div>
      <div class="stat m"><div class="v">${f0(cur.lat)}<small>ms</small></div>
        <div class="k">p95 ${closed?'ITL':'TTFT'}</div></div>
      <div class="stat ${closed?'n':(x<=CAP?'g':'p')}">
        <div class="v" style="font-size:20px">${closed?'量到飽和點':(x<=CAP?'SLO 通過':'SLO 失敗')}</div>
        <div class="k">${closed?'閉環只能回答「機器最多做多少」':'開環才能回答「能承諾多少」'}</div></div>
     </div>
     <div class="warn" style="margin-top:8px">此圖為<b>教學示例曲線</b>（以排隊理論生成），
      不是任何硬體的實測結果。正式報告必須換成實測資料，並附上引擎版本、workload、量測窗與 SLO 定義。</div>`;
  }
  render(); return {print:render};
}
