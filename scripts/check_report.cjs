// Set NODE_PATH to a directory containing Playwright when it is not local.
const {chromium} = require('playwright');
const path = require('node:path');
const fs = require('node:fs');

(async () => {
  const root = path.resolve(__dirname, '..');
  const browser = await chromium.launch({channel:'chrome', headless:true});
  const page = await browser.newPage({viewport:{width:1500,height:1000}});
  const errors = [];
  let phase='load';
  page.on('pageerror',e=>errors.push({message:e.message,stack:e.stack,phase}));
  await page.goto('file://' + path.join(root,'artifacts/week01-interactive.html'));
  await page.waitForFunction(()=>document.querySelectorAll('.js-plotly-plot').length===7 && [...document.querySelectorAll('.js-plotly-plot')].every(p=>p._fullLayout));
  const before = await page.evaluate(() => {
    const p=document.querySelector('.js-plotly-plot');
    return {frameCount:p._transitionData._frames.length, first:JSON.stringify(p.data[0].z)};
  });
  phase='play';
  await page.getByText('Play',{exact:true}).first().click();
  await page.waitForFunction(old=>JSON.stringify(document.querySelector('.js-plotly-plot').data[0].z)!==old,before.first);
  phase='pause';
  await page.getByText('Pause',{exact:true}).first().click();
  const paused=await page.evaluate(()=>JSON.stringify(document.querySelector('.js-plotly-plot').data[0].z));
  await page.waitForTimeout(800);
  const stillPaused=await page.evaluate(()=>JSON.stringify(document.querySelector('.js-plotly-plot').data[0].z));
  if(paused!==stillPaused) throw new Error('Pause did not freeze the frame');
  phase='slider';
  const rail=await page.locator('.slider-rail-rect').first().boundingBox();
  if(!rail) throw new Error('Missing depth slider rail');
  await page.mouse.click(rail.x+rail.width-1,rail.y+rail.height/2);
  await page.waitForFunction(()=>document.querySelector('.js-plotly-plot')._fullLayout.sliders[0].active===24);
  await page.screenshot({path:path.join(root,'artifacts/report-preview.png')});
  const figures=await page.evaluate(()=>[...document.querySelectorAll('.js-plotly-plot')].map(p=>({
    title:p.layout.title?.text,frames:p._transitionData._frames.length,traces:p.data.length,
    width:p.clientWidth,height:p.clientHeight
  })));
  await page.setViewportSize({width:900,height:900});
  await page.evaluate(()=>window.dispatchEvent(new Event('resize')));
  const evidence={errors,playChangedResidual:true,pauseFreezesFrame:true,sliderReachedLastStage:true,figures};
  fs.writeFileSync(path.join(root,'artifacts/browser-validation.json'),JSON.stringify(evidence,null,2));
  await browser.close();
  if(errors.length) throw new Error(errors.join('\n'));
  console.log(JSON.stringify(evidence,null,2));
})().catch(e=>{console.error(e);process.exit(1)});
