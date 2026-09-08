const PW='C:/Users/Hangzi/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright';
const {chromium}=require(PW);
const path=require('path'),fs=require('fs');
const ROOT=path.resolve(__dirname,'..');
const FILE=path.join(ROOT,'LLM推論報告_v2.html');
const QA=path.join(ROOT,'qa2');
const ONLY=process.argv.slice(2).filter(a=>!a.startsWith('--')).map(Number);
const PDF=process.argv.includes('--pdf');
(async()=>{
  fs.mkdirSync(QA,{recursive:true});
  const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',
    headless:true,args:['--no-sandbox','--font-render-hinting=none']});
  const page=await browser.newPage({viewport:{width:1520,height:900},deviceScaleFactor:1.35});
  const errors=[],remote=[];
  page.on('pageerror',e=>errors.push(String(e.message)));
  page.on('console',m=>{if(m.type()==='error')errors.push('console: '+m.text())});
  page.on('request',r=>{if(/^https?:/.test(r.url()))remote.push(r.url())});
  await page.goto('file:///'+FILE.replaceAll('\\','/'));
  await page.evaluate(()=>document.fonts.ready);
  const n=await page.evaluate(()=>deck.slides.length);
  const checks=[];
  const list=ONLY.length?ONLY:Array.from({length:n},(_,i)=>i+1);
  for(const p of list){
    const i=p-1;
    await page.evaluate(i=>{deck.go(i,false);
      (deck.bySlide.get(i)||[]).forEach(f=>f.finish());},i);
    await page.waitForTimeout(90);
    const c=await page.evaluate(()=>{
      const s=document.querySelector('.slide.active');
      const b=s.querySelector('.body'), fo=s.querySelector('.foot');
      const lim=fo?fo.getBoundingClientRect().top:s.getBoundingClientRect().bottom;
      const sr=s.getBoundingClientRect();
      const bad=b?[...b.querySelectorAll('*')].filter(e=>{
        const r=e.getBoundingClientRect();
        return r.width>1&&r.height>1&&(r.bottom>lim+1||r.right>sr.right-8||r.left<sr.left+56);
      }).map(e=>({tag:e.tagName,cls:e.className&&String(e.className).slice(0,40),
                  txt:(e.textContent||'').replace(/\s+/g,' ').slice(0,60),
                  over:Math.round(e.getBoundingClientRect().bottom-lim)})):[];
      return {title:(s.querySelector('h1')||{textContent:'(cover/section)'}).textContent.trim(),
              bodyH:b?b.scrollHeight:0, clip:b?b.scrollHeight-b.clientHeight:0,
              bad:bad.slice(0,4)};
    });
    checks.push({page:p,...c});
    await page.screenshot({path:path.join(QA,`s-${String(p).padStart(2,'0')}.png`),
      clip:await page.evaluate(()=>{const r=document.querySelector('.slide.active').getBoundingClientRect();
        return {x:r.x,y:r.y,width:r.width,height:r.height}})});
  }
  // step through every figure state looking for overflow
  const stepBad=await page.evaluate(()=>{
    const bad=[];
    for(let i=0;i<deck.slides.length;i++){
      deck.go(i,false);
      const figs=deck.bySlide.get(i)||[];
      for(const f of figs){
        f.reset();
        for(let j=0;j<=f.max;j++){
          const s=document.querySelector('.slide.active');
          const fo=s.querySelector('.foot');
          const lim=fo?fo.getBoundingClientRect().top:1e9;
          const ov=[...s.querySelectorAll('.body *')].filter(e=>{
            const r=e.getBoundingClientRect();return r.width>1&&r.height>1&&r.bottom>lim+1;});
          if(ov.length)bad.push({slide:i+1,fig:f.name,step:j,
            txt:(ov.at(-1).textContent||'').replace(/\s+/g,' ').slice(0,50)});
          f.next();
        }
      }
    }
    return bad;
  });
  let pdfBytes=0;
  if(PDF){
    await page.evaluate(()=>deck.preparePrint());
    await page.waitForTimeout(400);
    await page.pdf({path:path.join(ROOT,'LLM推論報告_v2.pdf'),preferCSSPageSize:true,printBackground:true});
    pdfBytes=fs.statSync(path.join(ROOT,'LLM推論報告_v2.pdf')).size;
  }
  const bad=checks.filter(c=>c.bad.length||c.clip>2);
  fs.writeFileSync(path.join(QA,'checks.json'),JSON.stringify({errors,remote,checks,stepBad},null,2));
  console.log(JSON.stringify({slides:n,errors:errors.slice(0,12),remote:remote.slice(0,5),
    overflow:bad.map(c=>({page:c.page,title:c.title,clip:c.clip,bad:c.bad})),
    stepBad:stepBad.slice(0,20),pdfBytes},null,2));
  await browser.close();
})();
