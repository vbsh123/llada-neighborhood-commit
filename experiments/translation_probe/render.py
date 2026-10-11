"""Verify downloaded traces and build a self-contained commitment-location slider.

CPU-only; no model/tokenizer downloads. All displayed states come from commits
verified against the saved before-forward states and final IDs.
"""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import unicodedata

from .evaluate import evaluate


def visible_text(text):
    return any(not char.isspace() and unicodedata.category(char) not in ('Cf','Cc','Cs') for char in text)


def collect(run):
    """Validate every forward and retain compact actual commits for the viewer."""
    run=Path(run)
    manifest=json.loads((run/'manifest.json').read_text())
    samples=[json.loads(line) for line in (run/'samples.jsonl').read_text().splitlines()]
    reports=[];audit={}
    for policy in manifest['config']['policies']:
        counts=Counter();blank=[];example_stats=[]
        for sample in samples:
            directory=run/policy/'samples'/sample['id']
            result=json.loads((directory/'result.json').read_text())
            assert result['source_text']==sample['source_text'] and result['reference']==sample['answer']
            end=result['answer_token_length'];frames=[];previous=[];config=None;last_result=None
            expected=None;dictionary=result['token_dictionary']
            with gzip.open(directory/'trace.jsonl.gz','rt',encoding='utf-8') as handle:
                for line in handle:
                    record=json.loads(line)
                    if record['type']=='header':
                        assert config is None
                        config=record['config'];mask=config['mask_id'];length=config['length']
                        assert config['policy']==policy and record['user_message']==sample['question']
                        assert record['prompt_protocol']==manifest['config']['prompt_protocol']
                        expected=[mask]*length;stop_ids=set(record['stop_ids'])
                        continue
                    if record['type']=='result':last_result=record;continue
                    if record['type']!='step':continue
                    assert record['step']==len(frames)
                    state=record['state_ids'];assert state==expected
                    assert record['previous_commits']==previous
                    rows={r['position']:r for r in record['positions']}
                    assert len(rows)==len(record['positions'])
                    assert set(rows)=={p for p,t in enumerate(state) if t==mask}
                    candidates=sorted((r for r in rows.values() if r['eligible']),key=lambda r:(-r['confidence'],r['position']))
                    assert candidates and all(math.isfinite(r['confidence']) and 0<=r['confidence']<=1 for r in rows.values())
                    chosen=([r for r in candidates if r['confidence']>=config['commit_threshold']] or candidates[:1]) if policy=='threshold' else candidates[:1]
                    commits=[r['position'] for r in chosen]
                    assert commits==record['commit_positions']
                    assert [rows[p]['token_id'] for p in commits]==record['commit_ids']
                    assert all(r['committed']==(p in commits) for p,r in rows.items())
                    batch=[]
                    for p in commits:
                        tid=rows[p]['token_id'];info=dictionary[str(tid)]
                        assert tid!=mask and info['special']==rows[p]['special']
                        assert info['text']==rows[p]['text']
                        batch.append([p,tid,rows[p]['confidence']]);expected[p]=tid
                        counts['all_commits']+=1
                        counts['valid_answer_commits']+=p<end and not info['special']
                        counts['special_commits']+=info['special']
                        counts['post_stop_commits']+=p>=end
                    counts['answer_unfinished_forwards']+=mask in state[:end]
                    frames.append({'step':record['step'],'commits':batch,'forward_seconds':record['forward_seconds']})
                    previous=commits
            assert config is not None and expected==result['final_ids'] and last_result==result
            assert len(frames)==result['steps']
            assert end==min((p for p,t in enumerate(expected) if t in stop_ids),default=length)
            counts['questions']+=1;counts['all_forwards']+=len(frames)
            meaningful=visible_text(result['translation'])
            if not meaningful:blank.append(sample['id'])
            valid_batches=[[c for c in f['commits'] if c[0]<end and not dictionary[str(c[1])]['special']] for f in frames]
            total=sum(map(len,valid_batches));early=[0.]*5;before=0
            for batch in valid_batches:
                if not batch:continue
                fraction=max(0,min(1,(.1*total-before)/len(batch)))
                for p,tid,confidence in batch:early[min(4,int(5*p/end))]+=fraction
                before+=len(batch)
            example_stats.append({'sample':sample['id'],'visible_translation':meaningful,'answer_end':end,
                                  'valid_answer_commits':total,
                                  'first_answer_commit_forward':next((f['step'] for f,b in zip(frames,valid_batches) if b),None),
                                  'early10_valid_commit_band_percent':[100*v/(.1*total) for v in early] if total else None})
            reports.append({'policy':policy,'sample':sample['id'],'source':sample['source_text'],
                            'reference':sample['answer'],'translation':result['translation'],
                            'chrfpp':result['chrfpp'],'visible_translation':meaningful,
                            'length':length,'answer_end':end,'dictionary':dictionary,'frames':frames})
        good=[e for e in example_stats if e['visible_translation'] and e['early10_valid_commit_band_percent'] is not None]
        audit[policy]={**dict(counts),'invisible_only_translation_ids':blank,'all_states_and_selections_verified':True,
                       'examples':example_stats,
                       'early10_valid_commit_band_percent_visible_outputs_equal_question':
                           [sum(e['early10_valid_commit_band_percent'][j] for e in good)/len(good) for j in range(5)] if good else None}
    by_id={}
    for r in reports:by_id.setdefault(r['sample'],{})[r['policy']]=r['translation']
    matching=sum(v.get('top1')==v.get('threshold') for v in by_id.values()) if {'top1','threshold'}<=set(manifest['config']['policies']) else None
    audit['identical_translation_pairs']=matching
    return reports,audit


def render(run,out):
    run,out=Path(run),Path(out);out.mkdir(parents=True,exist_ok=True)
    saved=json.loads((run/'comparison.json').read_text())
    rescored=evaluate(run)
    assert saved==rescored,'Saved corpus metrics do not match independent re-scoring'
    reports,audit=collect(run)
    audit['corpus_scores_recomputed_and_verified']=True
    payload={'reports':reports,'scores':rescored['policies']}
    text=TEMPLATE.replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c'))
    text=text.replace('__COUNT__',str(len({r['sample'] for r in reports})))
    text=text.replace('__THRESHOLD__',str(json.loads((run/'manifest.json').read_text())['config']['commit_threshold']))
    target=out/'translation_commitment_slider.html'
    target.write_text(text,encoding='utf-8')
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (out/'README.md').write_text(
        '# Translation commitment viewer\n\nOpen `translation_commitment_slider.html`; it is self-contained. '
        'All ten examples and both policies are included. No model inference was run.\n\n'
        'Default view shows the full response window. Unchecking the full-window box restricts positions to the final translation before its first stop token. '
        'The full-window checkbox includes special/post-stop fills. Frame 0 is before decoding; '
        'frame t+1 is after forward t. Regions are filled-token groups separated by at least '
        'the selected number of MASKs. Threshold commits are applied simultaneously. '
        'Union-find retains the lowest birth ID on mergers. Earlier frames retain their own history.\n\n'
        'The commitment timeline shows all saved commitment locations in the selected view, '
        'including future forwards, for retrospective inspection. The orange line identifies the '
        'selected frame. The token panel and merge graph show only history up through that frame.\n\n'
        'Two outputs contain only U+200B (zero-width space), rather than usable translations. '
        'They are flagged explicitly; original raw output/scoring are preserved. Corpus scores '
        'are not accuracy percentages.\n\n'
        'Code: `experiments/translation_probe/render.py`. Validation: `audit.json`.\n',encoding='utf-8')
    print(json.dumps(audit,indent=2));print(target)
    return target,audit


TEMPLATE=r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Translation — commitment locations</title>
<style>
body{font:16px system-ui;color:#233247;max-width:1400px;margin:20px auto;padding:16px;background:#f8fafc}h1{font-size:26px}h2{font-size:19px;margin-bottom:10px}section{background:white;padding:18px;margin:14px 0;border:1px solid #d9e1eb;border-radius:10px}.controls{display:flex;gap:18px;align-items:center;flex-wrap:wrap}select,button{font:inherit;padding:7px}input[type=range]{width:240px;vertical-align:middle}#time{width:min(700px,65vw)}svg{width:100%;background:#fafcff;border:1px solid #e2e8f0}.muted{color:#52657c;font-size:14px}.text{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.65}.warning{color:#99400b;font-weight:600}.token{display:inline-block;margin:2px;padding:3px 5px;white-space:pre-wrap;border-radius:3px;background:#edf1f6;font-family:monospace}.mask{color:#8b97a9;font-size:12px}.new{outline:2px solid #ed8317}#tokens{max-height:350px;overflow:auto}#legend{display:flex;gap:14px;flex-wrap:wrap}output{font-weight:bold}.columns{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}@media(max-width:700px){.columns{grid-template-columns:1fr}}
</style>
<h1>Translation — commitment locations</h1>
<p class="muted">__COUNT__ matched Chinese→English examples · TOP1 and threshold __THRESHOLD__ · saved traces only</p>
<section class="controls">
<label>Example <select id="sample" aria-label="Example"></select></label>
<label>Policy <select id="policy" aria-label="Decoding policy"><option value="top1">TOP1</option><option value="threshold">THRESHOLD __THRESHOLD__</option></select></label>
<label>Region boundary <input id="boundary" aria-label="MASK separation boundary" type="range" min="1" max="10" value="4" step="1"> <output id="boundary-value">4</output> MASKs</label>
<label><input id="full" type="checkbox" checked> Show full response window</label>
</section>
<section><div class="columns"><div><h2>Chinese source</h2><div id="source" class="text" lang="zh"></div></div><div><h2>Human reference</h2><div id="reference" class="text"></div></div></div>
<h2>Final generated translation</h2><div id="translation" class="text"></div><p id="quality" class="muted"></p><p id="warning" class="warning"></p></section>
<section><div class="controls"><button id="play">Play</button><input id="time" aria-label="Decoding frame" type="range" min="0" value="0" step="1"><output id="clock"></output></div>
<p id="stats"></p><h2>Response positions at this frame</h2><svg id="space" viewBox="0 0 1200 160"></svg><p id="legend"></p>
<p class="muted">Gray dots are MASKs. Colors identify union-find regions; orange outlines mark this forward’s commitments. Hover for token, position, and commitment confidence.</p>
<h2>Tokens at this frame</h2><div id="tokens"></div></section>
<section><h2>Commitment positions across all forwards</h2><svg id="timeline" viewBox="0 0 1200 360"></svg>
<p class="muted">Each dot is a commitment; a threshold forward can have multiple dots. Orange dots and the vertical line mark the selected forward. This overview includes future forwards. Hollow gray dots are special tokens.</p></section>
<section><h2>Region mergers up to this frame</h2><svg id="union"></svg><p id="events"></p></section>
<p class="muted">Default view shows the full response window, including EOS and post-stop fills. Uncheck the full-window box to restrict positions to the translation before its final first stop token (retrospective). A boundary of 4 means at least four intervening MASKs separate regions. Entire threshold batches are applied together. Regions retain their lowest birth ID after merging. Raw token text is preserved; invisible-only outputs are flagged.</p>
<script>
const payload=__DATA__,data=payload.reports,NS='http://www.w3.org/2000/svg';
const sample=document.querySelector('#sample'),policy=document.querySelector('#policy'),boundary=document.querySelector('#boundary'),full=document.querySelector('#full'),time=document.querySelector('#time');
let timer=null;const cache=new Map();
for(const [i,id] of [...new Set(data.map(r=>r.sample))].entries()){let o=document.createElement('option');o.value=id;o.textContent=`Example ${i+1} — ${id}`;sample.append(o)}
function selected(){return data.find(r=>r.sample===sample.value&&r.policy===policy.value)}
function color(i){return `hsl(${(i*137.508)%360} 62% 43%)`}
function el(svg,tag,attrs={},label,title){const n=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);if(label!==undefined)n.textContent=label;if(title){let t=document.createElementNS(NS,'title');t.textContent=title;n.append(t)}svg.append(n);return n}
function buildFrames(r,gap,limit){
 const key=`${r.policy}/${r.sample}/${gap}/${limit}`;if(cache.has(key))return cache.get(key);
 let tokens=Array(limit).fill(null),owner={},parent={},births=[],merges=[];
 function find(i){while(parent[i]!==i)i=parent[i];return i}
 const frames=[{step:null,tokens:[...tokens],groups:[],commits:[],births:[],merges:[]}];
 for(const raw of r.frames){
  const commits=raw.commits.filter(c=>c[0]<limit);for(const [p,tid] of commits)tokens[p]=tid;
  const groups=[];for(let p=0;p<limit;p++){if(tokens[p]===null)continue;if(!groups.length||p-groups.at(-1).at(-1)-1>=gap)groups.push([]);groups.at(-1).push(p)}
  const next={},components=[];
  for(const positions of groups){
   let roots=[...new Set(positions.filter(p=>owner[p]!==undefined).map(p=>find(owner[p])))].sort((a,b)=>a-b),root;
   if(!roots.length){root=births.length;parent[root]=root;births.push({id:root,step:raw.step,position:positions[0]})}
   else{root=roots[0];for(const from of roots.slice(1)){parent[from]=root;merges.push({from,to:root,step:raw.step})}}
   for(const p of positions)next[p]=root;
   components.push({root,positions,members:Object.keys(parent).map(Number).filter(i=>find(i)===root)});
  }
  owner=next;frames.push({step:raw.step,tokens:[...tokens],groups:components,commits,births:births.map(b=>({...b})),merges:merges.map(m=>({...m}))});
 }
 cache.set(key,frames);return frames;
}
function draw(){
 const r=selected(),limit=full.checked?r.length:r.answer_end,frames=buildFrames(r,+boundary.value,limit);time.max=frames.length-1;time.value=Math.min(+time.value,+time.max);
 const f=frames[+time.value],now=new Map(f.commits.map(c=>[c[0],c]));
 document.querySelector('#boundary-value').textContent=boundary.value;
 document.querySelector('#source').textContent=r.source;document.querySelector('#reference').textContent=r.reference;
 document.querySelector('#translation').textContent=r.translation;
 const scores=payload.scores[r.policy];document.querySelector('#quality').textContent=`This example chrF++: ${r.chrfpp.toFixed(2)} · Corpus chrF++: ${scores.chrfpp.toFixed(2)} · Corpus BLEU: ${scores.bleu.toFixed(2)} · Translation length: ${r.answer_end} slots`;
 document.querySelector('#warning').textContent=r.visible_translation?'':'No visible translation: '+Array.from(r.translation).map(c=>'U+'+c.codePointAt(0).toString(16).toUpperCase().padStart(4,'0')).join(' ')+' (raw invisible characters preserved).';
 document.querySelector('#clock').textContent=f.step===null?'Before first forward':`After forward ${f.step} · frame ${time.value}/${time.max}`;
 const valid=f.commits.filter(([p,tid])=>p<r.answer_end&&!r.dictionary[tid].special).length;
 const total=f.step===null?0:r.frames[+time.value-1].commits.length,hidden=total-f.commits.length;
 document.querySelector('#stats').textContent=`${f.groups.length} regions · ${f.tokens.filter(t=>t!==null).length}/${limit} displayed slots filled · ${total} commits this forward; ${f.commits.length} shown (${valid} nonspecial answer commits) · ${f.merges.length} mergers so far`+(hidden?` · ${hidden} EOS/post-stop commitments hidden; enable full response window to see them.`:'');
 const space=document.querySelector('#space');space.replaceChildren();const x=p=>40+1120*(p+.5)/Math.max(1,limit),dx=1120/Math.max(1,limit),owners={};
 for(const g of f.groups){for(const p of g.positions)owners[p]=g.root;el(space,'rect',{x:x(g.positions[0])-dx/2,y:45,width:(g.positions.at(-1)-g.positions[0]+1)*dx,height:55,fill:color(g.root),opacity:.15});el(space,'text',{x:x(g.positions[0]),y:30,'font-size':12,fill:color(g.root)},`R${g.root}`)}
 f.tokens.forEach((tid,p)=>{const info=tid===null?null:r.dictionary[tid],c=now.get(p);el(space,'circle',{cx:x(p),cy:72,r:Math.min(6,dx*.4),fill:tid===null?'#cdd3db':color(owners[p]),stroke:c?'#ed8317':'none','stroke-width':2},undefined,`${p}: ${info?JSON.stringify(info.text):'[MASK]'}${c?`; confidence ${c[2].toFixed(4)}`:''}`)});
 const tick=Math.max(1,Math.ceil(limit/12));for(let p=0;p<limit;p+=tick)el(space,'text',{x:x(p),y:122,'font-size':12},String(p));el(space,'text',{x:40,y:148,'font-size':13},'Response token index');
 if(full.checked&&r.answer_end<limit)el(space,'line',{x1:x(r.answer_end)-dx/2,x2:x(r.answer_end)-dx/2,y1:35,y2:105,stroke:'#a34','stroke-dasharray':'4 3'},undefined,'Final first stop boundary');
 const legend=document.querySelector('#legend');legend.replaceChildren();for(const g of f.groups){const n=document.createElement('span');n.style.color=color(g.root);n.textContent=`R${g.root}: birth IDs {${g.members.join(', ')}}`;legend.append(n)}
 const tokens=document.querySelector('#tokens');tokens.replaceChildren();f.tokens.forEach((tid,p)=>{const n=document.createElement('span');n.className='token'+(tid===null?' mask':'')+(now.has(p)?' new':'');n.title=`Position ${p}`;n.textContent=tid===null?'[MASK]':r.dictionary[tid].text;tokens.append(n)});
 const timeline=document.querySelector('#timeline');timeline.replaceChildren();const tx=s=>65+1080*(s+.5)/Math.max(1,r.frames.length),ty=p=>30+270*(p+.5)/Math.max(1,limit);
 el(timeline,'line',{x1:65,x2:1145,y1:300,y2:300,stroke:'#666'});el(timeline,'line',{x1:65,x2:65,y1:30,y2:300,stroke:'#666'});
 for(const raw of r.frames)for(const [p,tid,conf] of raw.commits){if(p>=limit)continue;const info=r.dictionary[tid];el(timeline,'circle',{cx:tx(raw.step),cy:ty(p),r:raw.step===f.step?3.5:2,fill:info.special?'white':raw.step===f.step?'#ed8317':'#3478aa',stroke:info.special?'#9ca8b8':'none'},undefined,`Forward ${raw.step}, position ${p}: ${JSON.stringify(info.text)}; confidence ${conf.toFixed(4)}`)}
 if(f.step!==null)el(timeline,'line',{x1:tx(f.step),x2:tx(f.step),y1:25,y2:300,stroke:'#ed8317','stroke-width':1});
 for(let s=0;s<r.frames.length;s+=Math.max(1,Math.ceil(r.frames.length/10)))el(timeline,'text',{x:tx(s),y:323,'font-size':12},String(s));for(let p=0;p<limit;p+=tick)el(timeline,'text',{x:12,y:ty(p)+4,'font-size':12},String(p));
 el(timeline,'text',{x:530,y:350,'font-size':14},'Actual forward step');el(timeline,'text',{x:5,y:15,'font-size':12},'Position');
 const graph=document.querySelector('#union');graph.replaceChildren();const rows=Math.max(1,Math.ceil(f.births.length/8));graph.setAttribute('viewBox',`0 0 1200 ${rows*110+40}`);const xy=id=>[75+(id%8)*150,60+Math.floor(id/8)*110];
 const defs=el(graph,'defs'),marker=el(defs,'marker',{id:'arrow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:6,markerHeight:6,orient:'auto'});el(marker,'path',{d:'M 0 0 L 10 5 L 0 10 z',fill:'#778'});
 for(const m of f.merges){const [a,b]=xy(m.from),[c,d]=xy(m.to),n=Math.hypot(c-a,d-b),u=(c-a)/n,v=(d-b)/n;el(graph,'line',{x1:a+u*23,y1:b+v*23,x2:c-u*25,y2:d-v*25,stroke:'#778','marker-end':'url(#arrow)'},undefined,`Forward ${m.step}: R${m.from} → R${m.to}`);el(graph,'text',{x:(a+c)/2,y:(b+d)/2-5,'font-size':11},`t${m.step}`)}
 const rootOf={};f.groups.forEach(g=>g.members.forEach(id=>rootOf[id]=g.root));for(const b of f.births){const [x,y]=xy(b.id),root=rootOf[b.id];el(graph,'circle',{cx:x,cy:y,r:23,fill:root===b.id?color(root):'#edf1f6',stroke:color(root),'stroke-width':3},undefined,`R${b.id}, born at forward ${b.step}, position ${b.position}; current root R${root}`);el(graph,'text',{x,y:y+5,'text-anchor':'middle',fill:root===b.id?'white':'#334'},`R${b.id}`)}
 document.querySelector('#events').textContent=f.births.length?`${f.births.length} birth identities; ${f.merges.length} mergers`:'No region births yet.';
}
function stop(){clearInterval(timer);timer=null;document.querySelector('#play').textContent='Play'}
sample.onchange=policy.onchange=()=>{stop();time.value=0;draw()};boundary.oninput=full.onchange=time.oninput=()=>{stop();draw()};
document.querySelector('#play').onclick=()=>{if(timer){stop();return}if(+time.value===+time.max)time.value=0;document.querySelector('#play').textContent='Pause';timer=setInterval(()=>{if(+time.value>=+time.max){stop();return}time.value=+time.value+1;draw()},180)};
draw();
</script></html>'''


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();render(args.run,args.out)
