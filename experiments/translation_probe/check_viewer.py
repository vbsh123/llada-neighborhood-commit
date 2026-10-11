"""Execute slider JavaScript in Node with a minimal DOM and check all frames."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

HARNESS=r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const text=fs.readFileSync(process.argv[2],'utf8'),script=text.match(/<script>([\s\S]*?)<\/script>/)[1];
class Element{constructor(tag=''){this.tag=tag;this.value='';this.checked=false;this.children=[];this.style={};this.attrs={};this.textContent=''}append(n){this.children.push(n);if(this.tag==='select'&&this.children.length===1)this.value=n.value}replaceChildren(){this.children=[]}setAttribute(k,v){this.attrs[k]=v}}
const nodes={},document={querySelector:s=>nodes[s]??=new Element(s==='#sample'?'select':''),createElement:t=>new Element(t),createElementNS:(ns,t)=>new Element(t)};
document.querySelector('#policy').value='top1';document.querySelector('#boundary').value='4';document.querySelector('#time').value='0';
document.querySelector('#full').checked=/<input id="full"[^>]*\bchecked\b/.test(text);
const ctx={document,clearInterval(){},setInterval(){return 1}};vm.createContext(ctx);vm.runInContext(script,ctx);
vm.runInContext('globalThis.reports=data;globalThis.build=buildFrames;globalThis.auditDraw=draw;globalThis.clearCache=()=>cache.clear()',ctx);
let total=0,variants=0,controls=0;
for(const r of ctx.reports){for(const gap of [1,2,3,4,5,6,7,8,9,10])for(const showFull of [false,true]){
 const limit=showFull?r.length:r.answer_end,frames=ctx.build(r,gap,limit);assert.equal(frames.length,r.frames.length+1);
 let expected=Array(limit).fill(null),everBorn=new Set(),parent={};
 for(let i=0;i<frames.length;i++){
  if(i>0)for(const [p,tid] of r.frames[i-1].commits){if(p<limit){assert.equal(expected[p],null);expected[p]=tid}}
  const f=JSON.parse(JSON.stringify(frames[i]));assert.deepStrictEqual(f.tokens,expected);
  assert.equal(f.step,i===0?null:r.frames[i-1].step);
  const independent=[];let masks=gap;
  for(let p=0;p<limit;p++){if(expected[p]===null)masks++;else{if(masks>=gap)independent.push([]);independent.at(-1).push(p);masks=0}}
  assert.deepStrictEqual(f.groups.map(g=>g.positions),independent);
  for(const b of f.births){assert(b.step<=f.step);if(!everBorn.has(b.id)){assert.equal(b.step,f.step);parent[b.id]=b.id;everBorn.add(b.id)}}
  for(const m of f.merges){assert(m.step<=f.step);parent[m.from]=m.to;assert(m.to<m.from)}
  const find=id=>{let seen=new Set();while(parent[id]!==id){assert(!seen.has(id));seen.add(id);id=parent[id]}return id};
  const ids=new Set();for(const g of f.groups){assert.equal(g.root,Math.min(...g.members));for(const id of g.members){assert(!ids.has(id));ids.add(id);assert.equal(find(id),g.root)}}
  assert.equal(ids.size,f.births.length);total++;
 }
 // Exercise browser event handlers at first, intermediate and final frames.
 document.querySelector('#sample').value=r.sample;document.querySelector('#policy').value=r.policy;document.querySelector('#policy').onchange();
 document.querySelector('#boundary').value=String(gap);document.querySelector('#boundary').oninput();
 document.querySelector('#full').checked=showFull;document.querySelector('#full').onchange();
 for(const index of [0,Math.floor((frames.length-1)/2),frames.length-1]){
  document.querySelector('#time').value=String(index);document.querySelector('#time').oninput();
  assert.equal(+document.querySelector('#time').max,r.frames.length);
  assert.equal(document.querySelector('#source').textContent,r.source);
  assert.equal(document.querySelector('#reference').textContent,r.reference);
  assert.equal(document.querySelector('#translation').textContent,r.translation);
  assert.equal(document.querySelector('#tokens').children.length,limit);
  assert.equal(document.querySelector('#warning').textContent.length>0,!r.visible_translation);
  assert.equal(document.querySelector('#boundary-value').textContent,String(gap));controls++;
  const hidden=index===0?0:r.frames[index-1].commits.filter(c=>c[0]>=limit).length;
  assert.equal(document.querySelector('#stats').textContent.includes('commitments hidden'),hidden>0);
 }
 variants++;ctx.clearCache();
}}
console.log(JSON.stringify({questions:10,policy_examples:ctx.reports.length,variants_checked:variants,frames_checked:total,control_frames_checked:controls,all_checks_passed:true}));
'''


def check(path,node=None):
    node=node or shutil.which('node')
    if not node:raise ValueError('Node required for JavaScript checks')
    with tempfile.TemporaryDirectory() as tmp:
        harness=Path(tmp)/'check.js';harness.write_text(HARNESS)
        result=subprocess.run([node,str(harness),str(path)],check=True,text=True,capture_output=True)
    report=json.loads(result.stdout)
    Path(path).with_name('viewer_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('html',type=Path);parser.add_argument('--node')
    args=parser.parse_args();check(args.html,args.node)
