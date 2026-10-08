'use strict';
const $ = id => document.getElementById(id);
let session, state, selected, contract, poll;
const colorRanges = {exposure:[-1,1,.01],contrast:[-25,25,1],highlights:[-40,20,1],shadows:[-20,40,1],whites:[-20,20,1],blacks:[-20,20,1],temperature:[-1200,1200,10],tint:[-15,15,1],saturation:[-25,25,1],vibrance:[-25,25,1],grain:[0,10,.1]};
const media = (area,path) => '/media/'+area+'/'+path.split('/').map(encodeURIComponent).join('/');
function message(text){$('message').textContent=text;}
function elapsed(seconds){seconds=Math.max(0,Math.floor(seconds||0));return seconds<60?`${seconds}s`:`${Math.floor(seconds/60)}m ${seconds%60}s`;}
async function api(path,body){
 const response=await fetch('/api/'+path,{method:body===undefined?'GET':'POST',headers:{'Content-Type':'application/json','X-Gallery-Token':session?.token||''},body:body===undefined?undefined:JSON.stringify(body)});
 const data=await response.json(); if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));return data;
}
async function action(fn){try{await fn();}catch(e){message(e.message);if(selected)cropPreview();}}
function control(parent,name,value,min,max,step,prefix,oninput){
 const label=document.createElement('label');label.textContent=name.replaceAll('_',' ');
 const input=document.createElement('input');Object.assign(input,{type:'number',id:prefix+name,value,min,max,step});input.setAttribute('aria-label',prefix+name);input.addEventListener('input',()=>oninput?.(input.value));label.append(input);parent.append(label);
}
function showStyle(style){
 if(!style)return;contract=structuredClone(style);$('styleSource').textContent=`${style.style} · source: ${style.source} · confidence: ${Math.round(style.confidence*100)}%`;
 $('styleControls').replaceChildren();
 for(const key of ['warmth','contrast','saturation','highlight_protection','shadow_lift','processing_level'])control($('styleControls'),key,contract[key],0,1,.05,'style-',value=>{contract[key]=Number(value);contract.source='user';$('contract').value=JSON.stringify(contract,null,2);});
 $('contract').value=JSON.stringify(contract,null,2);
 $('clusters').replaceChildren();for(const rule of style.conditional_rules||[]){const item=document.createElement('details');const title=document.createElement('summary');title.textContent=rule.label;const explanation=document.createElement('p');explanation.textContent=`${rule.rationale} · ${rule.legitimate_differences.join('; ')}`;const controls=document.createElement('div');controls.className='controls';for(const key of ['warmth_offset','contrast_offset','saturation_offset','shadow_lift_offset','highlight_protection_offset','exposure_match_strength','white_balance_match_strength'])control(controls,key,rule[key],key.endsWith('strength')?0:-.25,key.endsWith('strength')?.5:.25,.05,rule.cluster_id+'-',value=>{contract.conditional_rules.find(r=>r.cluster_id===rule.cluster_id)[key]=Number(value);contract.source='user';$('contract').value=JSON.stringify(contract,null,2);});item.append(title,explanation,controls);$('clusters').append(item);}
}
async function refresh(){
 state=await api('gallery');const report=state.report;const records=new Map((report?.photos||[]).map(r=>[r.id,r]));
 $('inventory').textContent=`Input directory: input/ · Found ${state.scan.count} photographs · ${Object.entries(state.scan.types||{}).map(([k,v])=>`${k}: ${v}`).join(' · ')}`;
 $('report').hidden=!report;$('performance').hidden=!state.performance;$('gallery').replaceChildren();
 if(!state.scan.count){const empty=document.createElement('div');empty.className='empty';empty.textContent='Place your photographs in input/ and scan. Originals stay untouched.';$('gallery').append(empty);}
 for(const p of state.scan.photos){const r=records.get(p.id);const card=document.createElement('button');card.className='card';
 const img=document.createElement('img');img.loading='lazy';img.src=media('cache',r?.edited_preview||p.thumbnail);img.alt=p.file;
 const body=document.createElement('div');body.className='cardBody';const name=document.createElement('strong');name.textContent=p.file;const info=document.createElement('small');
 info.textContent=r&&r.status!=='blocked'?`Edited${r.manual?' · Manual':''} · Crop ${r.crop_metrics.crop_percentage}% · ${Math.round(Math.min(...Object.values(r.confidence))*100)}% confidence · Review ${r.review.score}/100`:`Original · ${p.width} × ${p.height}`;
 body.append(name,info);if(r&&r.status!=='approved'){const warning=document.createElement('small');warning.className='warning';warning.textContent=r.status==='blocked'?'Export blocked':'Manual review required';body.append(warning);}card.append(img,body);card.onclick=()=>openPhoto(p,r);$('gallery').append(card);}
 if(!contract&&state.style)showStyle(state.style);
 const busy=['queued','scanning','analyzing','processing','reviewing'].includes(state.progress.state);
 $('progressPanel').hidden=state.progress.state==='idle';$('phase').textContent=state.progress.state==='analyzing'?'ANALYZING LIBRARY':state.progress.state==='reviewing'?'REVIEWING GALLERY':state.progress.state==='complete'?'COMPLETE':'EDITING PHOTOGRAPHS';
 $('stageProgress').max=Math.max(1,state.progress.total);$('stageProgress').value=state.progress.completed;$('stages').replaceChildren();for(const text of state.progress.finished_stages||[]){const li=document.createElement('li');li.textContent='✓ '+text;$('stages').append(li);}const stage=document.createElement('li');stage.textContent=`${state.progress.stage||state.progress.state} · ${state.progress.completed}/${state.progress.total}`;$('stages').append(stage);
 for(const id of ['scan','style','process','save','reset'])$(id).disabled=busy;
 if(busy)message(`${state.progress.stage||state.progress.state}: ${state.progress.completed}/${state.progress.total} · ${state.progress.file||''} · stage ${elapsed(state.progress.stage_elapsed_seconds)} · total ${elapsed(state.progress.elapsed_seconds)}`);
 if(state.progress.state==='failed')message(state.progress.error);
 if(!busy&&poll){clearInterval(poll);poll=null;if(state.progress.state!=='failed'){showStyle(state.style);const p=state.performance;message(state.progress.operation==='style'?`Deep album analysis complete in ${elapsed(p?.elapsed_seconds)}. Review global and conditional rules before editing.`:state.progress.operation==='variants'?`Crop and color alternatives are ready in ${elapsed(p?.elapsed_seconds)}. Review the variants for each photograph.`:`Processing finished in ${elapsed(p?.elapsed_seconds)}. Performance profile saved to reports/performance.json; review flagged images before using the exports.`);}}
 if(state.scan.errors?.length&&!busy)message(state.scan.errors.map(e=>`${e.file}: ${e.error}`).join(' · '));
}
function cropValue(){return Object.fromEntries(['x','y','width','height'].map(k=>[k,Number($('crop-'+k).value)]));}
function cropPreview(){
 if(!selected?.record)return;const c=cropValue(),p=selected.photo;
 const w=Math.round((c.x+c.width)*p.width)-Math.round(c.x*p.width),h=Math.round((c.y+c.height)*p.height)-Math.round(c.y*p.height);
 const s=session.settings;const valid=c.x>=0&&c.y>=0&&c.width>0&&c.height>0&&c.x+c.width<=1.000001&&c.y+c.height<=1.000001&&Math.max(w,h)>=s.min_long_edge&&w>=s.min_width&&h>=s.min_height&&w*h/1e6>=s.min_megapixels;
 $('metrics').textContent=`${w} × ${h} · ${(w*h/1e6).toFixed(2)} MP · ${(100*(1-w*h/(p.width*p.height))).toFixed(1)}% removed · ${valid?'Meets floor':'REJECTED: bounds or resolution floor'} · floor: ${s.min_long_edge}px long edge`;
 $('save').disabled=!valid;
 const image=$('original');const stage=image.parentElement;const offset=image.offsetTop;
 Object.assign($('cropBox').style,{left:`${c.x*100}%`,top:`${offset+c.y*image.clientHeight}px`,width:`${c.width*100}%`,height:`${c.height*image.clientHeight}px`});
 const paper=Object.entries({A4:[8.27,11.69],A3:[11.69,16.54],A2:[16.54,23.39]}).map(([name,[a,b]])=>{const ppi=Math.min(Math.min(w,h)/a,Math.max(w,h)/b);return `${name}: ${ppi>=300?'✓':ppi>=200?'~':'✕'}`;});
 $('printability').textContent=`Digital: ${Math.max(w,h)>=3000?'✓':'✕'} · ${paper.join(' · ')}. Print at 300/240/200 PPI: ${[300,240,200].map(ppi=>`${(w/ppi*2.54).toFixed(1)} × ${(h/ppi*2.54).toFixed(1)} cm`).join(' / ')}. ~ means 200–299 PPI.`;
}
function showVariants(photo,record){
 const grid=$('variantGrid');grid.replaceChildren();const unavailable=record.unavailable_variants||[];
 $('variantWarnings').textContent=unavailable.map(v=>`${v.label}: ${v.reason}`).join(' · ');
 for(const variant of record.variants||[]){
  const card=document.createElement('article');card.className='variantCard';
  const img=document.createElement('img');img.src=media('cache',variant.preview);img.alt=variant.label;
  const title=document.createElement('strong');title.textContent=variant.label;
  const m=variant.crop_metrics;const details=document.createElement('small');
  details.textContent=`${m.result_width} × ${m.result_height} · ${m.megapixels.toFixed(2)} MP · ${m.crop_percentage.toFixed(1)}% removed · A4 ${m.printability.A4} / A3 ${m.printability.A3} / A2 ${m.printability.A2}`;
  const grade=document.createElement('small');const c=variant.color;
  grade.textContent=`${variant.color_label} (${(c.look||'natural').replaceAll('_',' ')}): exposure ${c.exposure>=0?'+':''}${c.exposure.toFixed(2)} EV · contrast ${c.contrast>=0?'+':''}${c.contrast.toFixed(0)} · temperature ${c.temperature>=0?'+':''}${c.temperature.toFixed(0)} K · saturation ${c.saturation>=0?'+':''}${c.saturation.toFixed(0)} · vibrance ${c.vibrance>=0?'+':''}${c.vibrance.toFixed(0)}`;
  const notes=document.createElement('small');notes.textContent=(variant.notes||[]).join(' · ');
  let exportLink;if(variant.output){exportLink=document.createElement('a');exportLink.href=media('output',variant.output);exportLink.target='_blank';exportLink.rel='noopener';exportLink.textContent='Open full-resolution JPEG';}
  const button=document.createElement('button');const selected=record.selected_variant===variant.id||record.selected_option_id===variant.id;
  button.textContent=selected?(record.selected_option_id===variant.id?'AI reviewer selection':'Selected manually'):'Use this version';button.disabled=selected;
  button.onclick=()=>action(async()=>{button.disabled=true;message(`Rendering full-resolution ${variant.label}…`);const updated=await api(`photos/${photo.id}/variants/${encodeURIComponent(variant.id)}/select`,{});await refresh();openPhoto(photo,updated);message(`Selected ${variant.label}; full-resolution image exported and reviewed.`);});
  card.append(img,title,details,grade);if(notes.textContent)card.append(notes);if(exportLink)card.append(exportLink);card.append(button);grid.append(card);
 }
}
function openPhoto(photo,record){
 if(!record||record.status==='blocked'){message(record?.summary||'Process the gallery to edit this photograph.');return;}
 selected={photo,record};$('editor').hidden=false;$('filename').textContent=photo.file;
 $('original').src=media('cache',photo.preview);$('edited').src=media('cache',record.edited_preview)+'?v='+Date.now();$('zoom').value=1;
 $('cropControls').replaceChildren();for(const [key,value] of Object.entries(record.crop))control($('cropControls'),key,value,0,1,.001,'crop-',cropPreview);
 $('colorControls').replaceChildren();const lookLabel=document.createElement('label');lookLabel.textContent='Per-photo look';const lookSelect=document.createElement('select');lookSelect.id='color-look';for(const look of ['natural','monochrome','warm_monochrome','cinematic']){const option=document.createElement('option');option.value=look;option.textContent=look.replaceAll('_',' ');lookSelect.append(option);}lookSelect.value=record.color.look||'natural';lookLabel.append(lookSelect);$('colorControls').append(lookLabel);for(const [key,range] of Object.entries(colorRanges))control($('colorControls'),key,record.color[key],...range,'color-');
 $('aspect').value='original';$('summary').textContent=record.summary;$('confidence').textContent=Object.entries(record.confidence).map(([k,v])=>`${k.replaceAll('_',' ')}: ${Math.round(v*100)}%`).join(' · ');
 $('review').textContent=JSON.stringify(record.review_history,null,2);$('original').onload=cropPreview;zoom();$('editor').scrollIntoView({behavior:'smooth'});
 showVariants(photo,record);
}
function zoom(){for(const stage of document.querySelectorAll('.imageStage'))stage.style.width=`${Number($('zoom').value)*100}%`;cropPreview();}
function fitAspect(){let c=cropValue();const ratio=$('aspect').value;if(ratio!=='original'){const [a,b]=ratio.split(':').map(Number),p=selected.photo;let w=c.width,h=c.height;if(w*p.width/(h*p.height)>a/b)w=h*p.height*a/b/p.width;else h=w*p.width*b/a/p.height;c={x:c.x+(c.width-w)/2,y:c.y+(c.height-h)/2,width:w,height:h};}for(const [k,v]of Object.entries(c))$('crop-'+k).value=v;cropPreview();}
$('scan').onclick=()=>action(async()=>{message('Scanning input/…');await api('scan',{});await refresh();message('Scan complete.');});
$('style').onclick=()=>action(async()=>{message('Starting full-library analysis…');await api('style',{});$('album').open=true;poll=setInterval(()=>action(refresh),1500);await refresh();});
$('process').onclick=()=>action(async()=>{await api('process',{style:contract});message('Processing gallery…');poll=setInterval(()=>action(refresh),1500);await refresh();});
$('applyContract').onclick=()=>action(async()=>{const next=JSON.parse($('contract').value);next.source='user';showStyle(next);message('Contract applied locally; server validates it when processing starts.');});
 $('save').onclick=()=>action(async()=>{const color=Object.fromEntries(Object.keys(colorRanges).map(k=>[k,Number($('color-'+k).value)]));color.look=$('color-look').value;message('Rendering and reviewing manual edit…');$('save').disabled=true;const record=await api(`photos/${selected.photo.id}/override`,{crop:cropValue(),color,aspect_ratio:$('aspect').value});await refresh();openPhoto(selected.photo,record);message('Manual edit saved and protected.');});
$('reset').onclick=()=>action(async()=>{const record=await api(`photos/${selected.photo.id}/reset`,{});await refresh();openPhoto(selected.photo,record);message('Restored automatic version.');});
$('disableCrop').onclick=()=>{for(const k of ['x','y','width','height'])$('crop-'+k).value=['width','height'].includes(k)?1:0;$('aspect').value='original';cropPreview();};
$('aspect').onchange=fitAspect;$('zoom').oninput=zoom;$('close').onclick=()=>{$('editor').hidden=true;};
$('view').onchange=()=>{$('originalFigure').hidden=$('view').value==='edited';$('editedFigure').hidden=$('view').value==='original';cropPreview();};
for(const port of document.querySelectorAll('.viewport')){let drag;port.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,left:port.scrollLeft,top:port.scrollTop};port.setPointerCapture(e.pointerId);e.preventDefault();});port.addEventListener('pointermove',e=>{if(drag){port.scrollLeft=drag.left+drag.x-e.clientX;port.scrollTop=drag.top+drag.y-e.clientY;}});port.addEventListener('pointerup',()=>drag=null);port.addEventListener('pointercancel',()=>drag=null);}
window.addEventListener('resize',cropPreview);
action(async()=>{session=await api('session');$('mode').textContent=session.settings.vision_model?'Local vision model configured':'Measured fallback · no model';await refresh();});
