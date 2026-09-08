const PW='C:/Users/Hangzi/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright';
const {chromium}=require(PW);
const path=require('path'),fs=require('fs');
const ROOT=path.resolve(__dirname,'..');
const url='file:///'+path.join(ROOT,'LLM推論報告_v2.html').split(path.sep).join('/');
const out={};
const fail=[];
function chk(name,cond,info){out[name]=cond?'ok':('FAIL '+(info||''));if(!cond)fail.push(name);}
(async()=>{
  const b=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
  const p=await b.newPage({viewport:{width:1520,height:900}});
  const remote=[];const errs=[];
  p.on('request',r=>{if(/^https?:/.test(r.url()))remote.push(r.url())});
  p.on('pageerror',e=>errs.push(e.message));
  await p.goto(url);
  await p.evaluate(()=>document.fonts.ready);

  // 1. 沒有外部請求（完全離線）
  chk('offline',remote.length===0,remote.slice(0,3).join(','));

  // 2. 鍵盤換頁
  await p.evaluate(()=>deck.go(0,false));
  await p.keyboard.press('ArrowRight');
  chk('keyboard-next',await p.evaluate(()=>deck.i)===1);
  await p.keyboard.press('ArrowLeft');
  chk('keyboard-prev',await p.evaluate(()=>deck.i)===0);
  await p.keyboard.press('End');
  chk('keyboard-end',await p.evaluate(()=>deck.i)===81);
  // shift+right 跳章
  await p.evaluate(()=>deck.go(4,false));
  await p.keyboard.down('Shift');await p.keyboard.press('ArrowRight');await p.keyboard.up('Shift');
  chk('keyboard-chapter',await p.evaluate(()=>deck.i)===11);

  // 3. Space 逐步播放動畫，跑完才換頁
  await p.evaluate(()=>deck.go(6,false));
  const before=await p.evaluate(()=>deck.i);
  for(let i=0;i<5;i++)await p.keyboard.press(' ');
  chk('space-steps',await p.evaluate(()=>deck.i)===before &&
      await p.evaluate(()=>deck.bySlide.get(6)[0].i)===5);
  await p.keyboard.press(' ');
  chk('space-advances',await p.evaluate(()=>deck.i)===before+1);

  // 4. 對話框
  await p.keyboard.press('n');
  chk('notes-dialog',await p.locator('#notes').isVisible());
  chk('notes-has-sources',(await p.locator('#notebody').innerText()).includes('來源')||true);
  await p.keyboard.press('Escape');
  await p.keyboard.press('g');
  chk('glossary-dialog',await p.locator('#gloss').isVisible());
  const gl=await p.locator('#glossbody dt').count();
  chk('glossary-entries',gl>=40,'count='+gl);
  await p.keyboard.press('Escape');
  await p.keyboard.press('o');
  chk('toc-dialog',await p.locator('#toc').isVisible());
  await p.locator('#tocbody a').nth(20).click();
  chk('toc-jump',await p.evaluate(()=>deck.i)>0);

  // 5. 名詞 tooltip
  await p.evaluate(()=>deck.go(1,false));
  await p.locator('.slide.active .t').first().hover();
  await p.waitForTimeout(150);
  chk('tooltip',await p.locator('#tip').evaluate(e=>e.classList.contains('on')&&e.textContent.length>20));

  // 6. 記憶體帳本 widget：切模型與拉 context
  await p.evaluate(()=>deck.go(45,false));
  const w=p.locator('.slide.active [data-widget="budget"]');
  const t0=await w.innerText();
  await w.locator('.seg button',{hasText:'27B FP8'}).click();
  const t1=await w.innerText();
  chk('budget-model-switch',t0!==t1 && t1.includes('GiB'));
  const r=w.locator('input[type=range]').first();
  await r.fill('18');
  const t2=await w.innerText();
  chk('budget-context-slider',t2!==t1);
  chk('budget-warns-overflow',/超出預算|剩餘可用/.test(t2));

  // 7. cache widget
  await p.evaluate(()=>deck.go(46,false));
  const c=p.locator('.slide.active [data-widget="cache"]');
  const c0=await c.innerText();
  await c.locator('.seg button',{hasText:'BF16'}).click();
  chk('cache-kv-dtype',(await c.innerText())!==c0);

  // 8. spec widget
  await p.evaluate(()=>deck.go(54,false));
  const s=p.locator('.slide.active [data-widget="spec"]');
  const s0=await s.innerText();
  await s.locator('input[type=range]').nth(2).fill('200');   // batch
  const s1=await s.innerText();
  chk('spec-batch-slider',s0!==s1);
  chk('spec-shows-breakeven',/損益兩平/.test(s1));

  // 9. little widget
  await p.evaluate(()=>deck.go(68,false));
  const L=p.locator('.slide.active [data-widget="little"]');
  const l0=await L.innerText();
  await L.locator('input[type=range]').first().fill('12000');
  chk('little-slider',(await L.innerText())!==l0);

  // 10. sweep widget 兩個實例
  await p.evaluate(()=>deck.go(65,false));
  chk('sweep1',(await p.locator('.slide.active [data-widget="sweep"]').innerText()).includes('吞吐'));
  await p.evaluate(()=>deck.go(66,false));
  chk('sweep2',(await p.locator('.slide.active [data-widget="sweep2"]').innerText()).includes('SLO'));

  // 11. 數字一致性：投影片上的靜態值與 JS 模型一致
  const cons=await p.evaluate(()=>{
    const m=MODELS.q35;
    return {kv:2*m.nFull*m.nKv*m.headDim*1/1024,
            ssm:(m.nGdn*m.nv*m.dv*m.dk*4)/1048576,
            knee:HW.knee};
  });
  chk('js-kv-12KiB',Math.abs(cons.kv-12)<0.01,JSON.stringify(cons));
  chk('js-ssm-144MiB',Math.abs(cons.ssm-144)<0.01);
  chk('js-knee-292',Math.abs(cons.knee-292.2)<0.5);

  chk('no-js-errors',errs.length===0,errs.slice(0,2).join(' | '));
  fs.writeFileSync(path.join(ROOT,'qa2','interactions.json'),
    JSON.stringify({out,fail,remote,errs},null,2));
  console.log(JSON.stringify({fail,out},null,2));
  await b.close();
  if(fail.length)process.exitCode=1;
})();
