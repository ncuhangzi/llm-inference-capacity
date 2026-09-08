const {chromium}=require('C:/Users/Hangzi/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const path=require('path');const fs=require('fs');
(async()=>{
fs.mkdirSync('qa',{recursive:true});
const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--no-sandbox']});
const page=await browser.newPage({viewport:{width:1600,height:960},deviceScaleFactor:1});
let errors=[];page.on('pageerror',e=>errors.push(e.message));
await page.goto('file:///'+path.resolve('LLM推論報告.html').replaceAll('\\','/'));
await page.evaluate(()=>document.fonts.ready);
const checks=[];
for(let i=0;i<40;i++){
 await page.evaluate(i=>{deck.go(i);const c=deck.controllers.get(deck.slides[i].demo);if(c)c.finish();},i);
 const check=await page.evaluate(()=>{let s=document.querySelector('.slide.active'),b=s.querySelector('.body'),f=s.querySelector('.footer').getBoundingClientRect(),r=b.getBoundingClientRect();let bad=[...b.querySelectorAll('*')].filter(e=>{let r=e.getBoundingClientRect();return r.width&&r.height&&(r.bottom>f.top-10||r.right>1525||r.left<70);}).map(e=>({tag:e.tagName,text:e.textContent.slice(0,70),bottom:e.getBoundingClientRect().bottom}));return{title:s.querySelector('h1').textContent,height:b.scrollHeight,bad};});
 checks.push({page:i+1,...check});
 await page.screenshot({path:`qa/slide-${String(i+1).padStart(2,'0')}.png`});
}
// Check all step states for layout and numerical invariants.
const functional=await page.evaluate(()=>{const out=[];for(const [name,c] of deck.controllers){if(c.reset){c.reset();for(let j=0;j<10;j++){c.next();}}out.push(name);}return{demos:out,cache32k:deck.cache(32768),images:[...document.images].map(i=>({ok:i.complete&&i.naturalWidth>0,width:i.naturalWidth})),unresolved:document.documentElement.innerHTML.includes('@@')};});
await page.evaluate(()=>deck.preparePrint());
await page.pdf({path:'LLM推論報告.pdf',preferCSSPageSize:true,printBackground:true});
fs.writeFileSync('qa/checks.json',JSON.stringify({errors,checks,functional},null,2));
console.log(JSON.stringify({errors,overflow:checks.filter(c=>c.bad.length),functional},null,2));
await browser.close();
})();
