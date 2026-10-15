"""Offline confidence buckets, entropy strata, and downloaded sleep diagnostics.

No checkpoint, tokenizer, torch, dataset download or generation is used.
Run with --top1 top1.zip --threshold threshold.zip --sleep 'runs (4).zip' --out DIR.
"""
import argparse
import base64
from collections import Counter, defaultdict
import csv
import hashlib
import html
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from confidence_geography.trace_source import Source
from experiments.gsm8k_eval.scoring import score_saved

ENTROPY = ('All entropy', 'H < 1', '1 ≤ H < 2', '2 ≤ H < 3', 'H ≥ 3')
HORIZONS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def buckets(confidence, entropy):
    if not (math.isfinite(confidence) and 0 <= confidence <= 1 and
            math.isfinite(entropy) and entropy >= 0):
        raise ValueError('Invalid confidence or entropy')
    return min(9, int(confidence * 10)), 1 + sum(entropy >= h for h in (1, 2, 3))


def answer_positions(result):
    end = result.get('first_stop_position')
    end = len(result['final_ids']) if end is None else end
    dictionary = result['token_dictionary']
    return {p for p, token in enumerate(result['final_ids'][:end])
            if not dictionary[str(token)]['special']}


def collect_baseline(path, policy):
    """Collect every fresh pre-fill observation, then attach its actual commit step.

    Main scope: eventual nonspecial answer positions before the final first stop.
    The final-answer filter is retrospective and never controls generation.
    Each question is kept as one bootstrap cluster. No first-entry selection.
    """
    source = Source(path)
    observations, sums, questions = [], [], {}
    try:
        for number, name in enumerate(source.names):
            result = source.result(name)
            sid = str(result['sample_id'])
            if sid in questions:
                raise ValueError('Duplicate question ID')
            valid = answer_positions(result)
            commits, rows, header = {}, [], None
            for record in source.records(name):
                if record['type'] == 'header':
                    header = record
                    continue
                if record['type'] != 'step':
                    continue
                step = record['step']
                if header is None or 'positions' not in record:
                    raise ValueError('Need complete full-position baseline traces')
                for position, token in zip(record['commit_positions'], record['commit_ids']):
                    if position in commits or token != result['final_ids'][position]:
                        raise ValueError('Repeated commitment or final token mismatch')
                    commits[position] = step
                for row in record['positions']:
                    p = row['position']
                    if p in valid and row['eligible']:
                        b, e = buckets(row['confidence'], row['entropy'])
                        rows.append((step, p, b, e, row['confidence'], row['entropy']))
            config = header['config']
            protocol = header.get('prompt_protocol', config.get('prompt_protocol'))
            if protocol != 'question_only_chat_v1' or config['policy'] != policy:
                raise ValueError('Unexpected prompt or policy')
            if config['block_length'] != config['length']:
                raise ValueError('This analysis expects unrestricted full-window traces')
            if set(commits) != set(range(len(result['final_ids']))):
                raise ValueError('Incomplete response commitments')
            hist = np.zeros((10, 5, config['length'] + 1), dtype=np.int64)
            # sums[..., 0/1] store confidence/entropy sums; zero-delay observations
            # are tracked separately so the sleep-decision view can exclude them.
            totals = np.zeros((2, 10, 5, 2), dtype=float)
            last_seen = {}
            for step, p, b, e, confidence, entropy in rows:
                delay = commits[p] - step
                if delay < 0 or (p in last_seen and step != last_seen[p] + 1):
                    raise ValueError('Observation after commitment or missing fresh forward')
                last_seen[p] = step
                for band in (0, e):
                    hist[b, band, delay] += 1
                    totals[0, b, band] += (confidence, entropy)
                    if delay > 0:
                        totals[1, b, band] += (confidence, entropy)
            if set(last_seen) != valid or any(last_seen[p] != commits[p] for p in valid):
                raise ValueError('Answer position lacks its commitment observation')
            questions[sid] = dict(question=result['question'], reference=result['reference'],
                                  answer=result['answer'], legacy_correct=result['correct_numeric'],
                                  valid_positions=len(valid), observations=len(rows),
                                  model=config['model'], revision=config['revision'])
            observations.append(hist)
            sums.append(totals)
            if (number + 1) % 10 == 0:
                print(f'{policy}: read {number + 1}/{len(source.names)} questions', flush=True)
    finally:
        source.close()
    if len({a.shape for a in observations}) != 1:
        raise ValueError('Mixed response lengths')
    return np.stack(observations), np.stack(sums), questions


def histogram_quantile(hist, fraction):
    count = int(hist.sum())
    if not count:
        return None
    rank = (count - 1) * fraction
    cumulative = hist.cumsum()
    lo = int(np.searchsorted(cumulative, math.floor(rank) + 1))
    hi = int(np.searchsorted(cumulative, math.ceil(rank) + 1))
    return lo + (hi - lo) * (rank - math.floor(rank))


def summarize(histograms, sums, replicates=1000):
    """Pooled observation statistics; bootstrap resamples entire questions.

    A duration of zero means committed in the SAME forward as the observation.
    The primary 'waiting' view removes those observations because the scheduler
    cannot put a token it just committed to sleep.
    """
    n_questions = len(histograms)
    weights = np.random.default_rng(1729).multinomial(
        n_questions, np.full(n_questions, 1 / n_questions), size=replicates)
    output = {}
    for scope, first_delay, sum_index in (('waiting', 1, 1), ('all', 0, 0)):
        cells = []
        for b in range(10):
            for e in range(5):
                qhist = histograms[:, b, e].copy()
                qhist[:, :first_delay] = 0
                per_question = qhist.sum(axis=1)
                hist = qhist.sum(axis=0)
                n = int(hist.sum())
                cell = dict(confidence_bucket=b, confidence_range=f'{10*b}–{10*(b+1)}%',
                            entropy_band=e, entropy_label=ENTROPY[e], observations=n,
                            questions=int((per_question > 0).sum()), histogram=hist.tolist(),
                            mean=None, median=histogram_quantile(hist, .5),
                            p90=histogram_quantile(hist, .9), mean_confidence=None,
                            mean_entropy=None, within={}, within_ci95={})
                if n:
                    cell['mean'] = float(np.dot(hist, np.arange(len(hist))) / n)
                    cell['mean_confidence'], cell['mean_entropy'] = (
                        sums[:, sum_index, b, e].sum(axis=0) / n).tolist()
                    denominator = weights @ per_question
                    valid = denominator > 0
                    for h in HORIZONS:
                        success = qhist[:, :h+1].sum(axis=1)
                        cell['within'][str(h)] = float(100 * success.sum() / n)
                        bootstrap = 100 * (weights @ success)[valid] / denominator[valid]
                        cell['within_ci95'][str(h)] = np.quantile(bootstrap, [.025, .975]).tolist()
                cells.append(cell)
        output[scope] = cells
    return output


def nearby_reveal_during_sleep(episode, commits, length, radius=4):
    """A reveal at r affects input r+1; exclude reveals just before a fresh wake."""
    return any(episode['start'] <= commits[p] < episode['actual_wake']-1
               for p in range(max(0,episode['position']-radius),
                              min(length,episode['position']+radius+1)))


def sleep_diagnostics(path, baselines):
    """Analyze actual sleep episodes; never pretend assigned-only rows are all rows."""
    source = Source(path)
    groups, per_question = defaultdict(list), {}
    try:
        for name in source.names:
            result = source.result(name)
            records = list(source.records(name))
            header = records[0]
            config = header['config']
            label = f"H{config['high_entropy']:g}_skip{config['high_sleep']}"
            sid = str(result['sample_id'])
            if sid in per_question.setdefault(label, {}):
                raise ValueError('Duplicate sleep question')
            baseline = baselines[sid]
            if (result['question'], result['reference']) != (baseline['question'], baseline['reference']):
                raise ValueError('Sleep and baseline questions/references differ')
            if (config['model'], config['revision']) != (baseline['model'], baseline['revision']):
                raise ValueError('Model mismatch')
            steps = [r for r in records if r['type'] == 'step']
            commits, episodes = {}, []
            sleeping_exposures = nearby = answer_sleep = total_sleep = 0
            pre_stop = result['first_stop_position']
            pre_stop = len(result['final_ids']) if pre_stop is None else pre_stop
            pending = {}
            for record in steps:
                step = record['step']
                sleepers = set(record['sleep_positions'])
                chosen = record['commit_positions']
                if sleepers.intersection(chosen):
                    raise ValueError('A sleeping position was committed')
                for p in list(pending):
                    if p not in sleepers:
                        pending[p]['actual_wake'] = step
                        episodes.append(pending.pop(p))
                total_sleep += len(sleepers)
                answer_sleep += sum(p < pre_stop for p in sleepers)
                for p in chosen:
                    if p in commits:
                        raise ValueError('Duplicate commitment')
                    commits[p] = step
                for p in sleepers:
                    sleeping_exposures += 1
                    nearby += any(abs(p - q) <= 4 for q in chosen)
                for entry in record['assigned']:
                    p = entry['position']
                    if p in pending or p in sleepers or p in commits:
                        raise ValueError('Invalid sleep assignment')
                    pending[p] = dict(entry, start=step, actual_wake=None)
            if pending or len(commits) != len(result['final_ids']):
                raise ValueError('Incomplete wake/commit history')
            wake_delays = Counter(commits[e['position']] - e['actual_wake'] for e in episodes)
            if any(k < 0 for k in wake_delays):
                raise ValueError('Commitment before wake')
            assignments = Counter(e['position'] for e in episodes)
            entry_wait = Counter(commits[e['position']] - e['start'] for e in episodes)
            # A reveal at step r first changes the next forward's input. Only
            # count reveals whose next forward is still asleep, not the final
            # reveal immediately before a fresh wake.
            nearby_episodes = sum(nearby_reveal_during_sleep(e,commits,len(result['final_ids']))
                                  for e in episodes)
            per_question[label][sid] = dict(
                question=result['question'], reference=result['reference'], answer=result['answer'],
                baseline_answer=baseline['answer'], baseline_correct=baseline['legacy_correct'],
                correct=result['correct_numeric'], decode_seconds=result['decode_seconds'],
                model_seconds=result['model_seconds'], sleeping_position_forwards=total_sleep,
                pre_stop_sleeping_position_forwards=answer_sleep,
                post_stop_sleeping_position_forwards=total_sleep-answer_sleep,
                episodes=len(episodes), slept_positions=len(assignments),
                max_sleep_cycles=max(assignments.values(), default=0),
                answer_token_length=pre_stop,
                all_asleep_full_wakes=result['all_asleep_full_wakes'],
                episodes_with_nearby_reveal_before_sleep_forward=nearby_episodes,
                wake_to_commit_histogram=dict(sorted(wake_delays.items())),
                assignment_to_commit_histogram=dict(sorted(entry_wait.items())),
                sleep_exposures_with_commit_within_4=nearby,
                sleep_exposures=sleeping_exposures)
            groups[label].append((result, episodes))
    finally:
        source.close()
    report = {}
    for label, items in groups.items():
        qs = per_question[label]
        if set(qs) != set(baselines):
            raise ValueError('Sleep cohort differs from baseline cohort')
        results = [r for r, _ in items]
        scored = score_saved([r['answer'] for r in results], [r['reference'] for r in results])
        correct = dict(zip([str(r['sample_id']) for r in results], scored['flexible-extract']['correct']))
        for sid in qs:
            qs[sid]['harness_correct']=correct[sid]
        histogram = Counter()
        for q in qs.values():
            histogram.update(q['wake_to_commit_histogram'])
        report[label] = dict(samples=len(qs), harness_accuracy=scored['flexible-extract']['accuracy'],
                            legacy_accuracy=sum(q['correct'] for q in qs.values())/len(qs),
                            decode_seconds=sum(q['decode_seconds'] for q in qs.values()),
                            full_window_tps=sum(r['steps'] for r in results)/sum(q['decode_seconds'] for q in qs.values()),
                            sleep_episodes=sum(q['episodes'] for q in qs.values()),
                            sleeping_position_forwards=sum(q['sleeping_position_forwards'] for q in qs.values()),
                            post_stop_sleeping_position_forwards=sum(q['post_stop_sleeping_position_forwards'] for q in qs.values()),
                            episodes_with_nearby_reveal_before_sleep_forward=sum(q['episodes_with_nearby_reveal_before_sleep_forward'] for q in qs.values()),
                            wake_to_commit_histogram=dict(sorted(histogram.items())),
                            harness_correct_by_id=correct)
    labels = sorted(report)
    if len(labels) == 2:
        a, b = labels
        pairs = Counter(('both_correct' if report[a]['harness_correct_by_id'][sid] and report[b]['harness_correct_by_id'][sid]
                         else 'both_wrong' if not report[a]['harness_correct_by_id'][sid] and not report[b]['harness_correct_by_id'][sid]
                         else a+'_only_correct' if report[a]['harness_correct_by_id'][sid] else b+'_only_correct')
                        for sid in baselines)
        report['paired'] = dict(pairs)
        differences=np.array([int(report[a]['harness_correct_by_id'][sid])-
                              int(report[b]['harness_correct_by_id'][sid]) for sid in sorted(baselines)])
        draws=np.random.default_rng(1729).integers(0,len(differences),size=(1000,len(differences)))
        report['paired']['accuracy_a_minus_b_percentage_points']=float(100*differences.mean())
        report['paired']['ci95_question_bootstrap']=np.quantile(100*differences[draws].mean(axis=1),[.025,.975]).tolist()
    return report, per_question


def plot_readiness(data, out):
    """Generate median-by-confidence bars, near-term CDF heatmaps, per-bucket CDFs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ('#777777', '#207baf', '#e19b27', '#bd573d', '#784b98')
    images = {}
    for policy, populations in data.items():
        for scope, cells in populations.items():
            lookup = {(c['confidence_bucket'], c['entropy_band']): c for c in cells}
            caption = 'Not committed in this forward' if scope == 'waiting' else 'Including same-forward commitments'
            fig, ax = plt.subplots(figsize=(11, 4.8))
            for metric,color in (('median','#207baf'),('mean','#d98025')):
                ax.plot(range(10),[lookup[b,0][metric] for b in range(10)],'o-',color=color,label=metric.title())
            ax.set(xticks=range(10),xticklabels=[f'{10*b}–{10*(b+1)}' for b in range(10)],
                   xlabel='Top-1 confidence (%)',ylabel='Additional forwards until commitment',
                   title=f'{policy.upper()} — confidence buckets, all entropy\n{caption}')
            ax.legend();ax.grid(alpha=.2);fig.tight_layout()
            path=out/f'{policy}_{scope}_overview.png';fig.savefig(path,dpi=160);plt.close(fig)
            images[f'{policy}_{scope}_overview']=path
            fig, ax = plt.subplots(figsize=(12, 5.5))
            for e in range(1, 5):
                y = [lookup[b, e]['median'] if lookup[b, e]['observations'] else np.nan for b in range(10)]
                bars=ax.bar(np.arange(10)+(e-2.5)*.19, y, .19, label=ENTROPY[e], color=colors[e])
                for b,bar in enumerate(bars):
                    if lookup[b,e]['questions']<20 or lookup[b,e]['observations']<100:
                        bar.set_hatch('///');bar.set_edgecolor('#444444')
            ax.set(xticks=range(10), xticklabels=[f'{10*b}–{10*(b+1)}' for b in range(10)],
                   xlabel='Top-1 confidence (%)', ylabel='Median additional forwards until commitment',
                   title=f'{policy.upper()} — commitment timing by confidence and entropy\n{caption}')
            ax.legend(); ax.grid(axis='y', alpha=.2); fig.tight_layout()
            path = out/f'{policy}_{scope}_median.png'; fig.savefig(path, dpi=160); plt.close(fig)
            images[f'{policy}_{scope}_median'] = path
            fig, ax = plt.subplots(figsize=(12, 4.8))
            values = np.array([[lookup[b, e]['within'].get('2', np.nan) for b in range(10)] for e in range(1, 5)])
            image = ax.imshow(values, aspect='auto', vmin=0, vmax=100, cmap='Blues')
            for e in range(4):
                for b in range(10):
                    c = lookup[b, e+1]
                    if c['observations']:
                        ax.text(b, e, f"{values[e,b]:.1f}%\nn={c['observations']:,}", ha='center', va='center',
                                fontsize=7, color='white' if values[e,b] > 55 else '#152838')
            ax.set(xticks=range(10), xticklabels=[f'{10*b}–{10*(b+1)}' for b in range(10)],
                   yticks=range(4), yticklabels=ENTROPY[1:], xlabel='Top-1 confidence (%)',
                   title=f'{policy.upper()} — percentage committed within 2 additional forwards\n{caption}')
            fig.colorbar(image, ax=ax, label='% of observations in the cell'); fig.tight_layout()
            path=out/f'{policy}_{scope}_within2.png'; fig.savefig(path, dpi=160); plt.close(fig)
            images[f'{policy}_{scope}_within2'] = path
            for b in range(10):
                fig, ax = plt.subplots(figsize=(10.5, 5.4))
                for e in range(5):
                    c = lookup[b, e]
                    if not c['observations']:
                        continue
                    histogram = np.array(c['histogram'])
                    ax.step(range(len(histogram)), 100*histogram.cumsum()/histogram.sum(), where='post',
                            color=colors[e], linestyle='--' if e == 0 else '-',
                            label=f"{ENTROPY[e]} (n={c['observations']:,}; {c['questions']} questions)")
                    if e:
                        for h in HORIZONS[:5]:
                            ci = c['within_ci95'][str(h)]; value = c['within'][str(h)]
                            ax.errorbar(h, value, yerr=[[max(0,value-ci[0])],[max(0,ci[1]-value)]],
                                        color=colors[e], capsize=2, alpha=.65)
                ax.set(xscale='symlog', xlim=(0, 256), ylim=(0, 101),
                       xticks=[0,1,2,4,8,16,32,64,128,256],
                       xticklabels=['0','1','2','4','8','16','32','64','128','256'],
                       xlabel='Additional forwards until commitment (0 = this forward)',
                       ylabel='Cumulative % of observations',
                       title=f'{policy.upper()} — confidence {10*b}–{10*(b+1)}%\n{caption}')
                if ax.lines:
                    ax.legend(fontsize=8)
                else:
                    ax.text(.5,.5,'No observations in this population / confidence bucket',
                            transform=ax.transAxes,ha='center')
                ax.grid(alpha=.2); fig.tight_layout()
                path=out/f'{policy}_{scope}_cdf_{b}.png'; fig.savefig(path,dpi=130); plt.close(fig)
                images[f'{policy}_{scope}_cdf_{b}'] = path
    return images


def write_html(data, diagnostics, images, out, sleep_questions=None):
    def embedded(key):
        return 'data:image/png;base64,'+base64.b64encode(images[key].read_bytes()).decode()
    assets = {key: embedded(key) for key in images}
    payload = json.dumps(dict(data=data, images=assets), separators=(',',':'))
    cards = ''.join(f'<li>{html.escape(label)}: harness accuracy {100*r["harness_accuracy"]:.1f}%, '
                    f'{r["full_window_tps"]:.2f} full-window TPS; {r["sleep_episodes"]:,} sleep episodes.</li>'
                    for label,r in diagnostics.items() if label != 'paired')
    examples=[]
    if sleep_questions and len(sleep_questions)==2:
        a,b=sorted(sleep_questions)
        for sid in sorted(sleep_questions[a]):
            qa,qb=sleep_questions[a][sid],sleep_questions[b][sid]
            if qa['harness_correct']==qb['harness_correct']:
                continue
            sections=''.join('<h4>'+html.escape(label)+'</h4><pre style="white-space:pre-wrap">'+html.escape(answer)+'</pre>'
                             for label,answer in [('Reference',qa['reference']),('Historical full baseline',qa['baseline_answer']),
                                                  (a+(' — correct' if qa['harness_correct'] else ' — incorrect'),qa['answer']),
                                                  (b+(' — correct' if qb['harness_correct'] else ' — incorrect'),qb['answer'])])
            examples.append('<details><summary>'+html.escape(sid+' — '+qa['question'])+'</summary>'+sections+'</details>')
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Confidence and commitment timing</title>
<style>body{font:16px system-ui;max-width:1250px;margin:24px auto;padding:16px;color:#203047}img{width:100%;height:auto}section{border:1px solid #ddd;padding:18px;margin:18px 0}select{font:inherit;margin:8px;padding:5px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}.muted{color:#526274;line-height:1.5}</style>
<h1>Confidence, entropy and time to commitment</h1>
<p class="muted">Full baseline traces: all fresh observations of eventual nonspecial answer positions before the final first stop. Positions remain masked at the instant observed. Each observation uses its own confidence and entropy; no first-entry selection. Error bars are 95% question-cluster bootstrap intervals (1,000 replicates). Long-lived positions contribute more observations. This describes saved trajectories, not cache safety or a causal sleep effect.</p>
<label>Policy<select id="policy"><option value="top1">TOP1</option><option value="threshold">THRESHOLD 0.9</option></select></label>
<label>Population<select id="scope"><option value="waiting">Not committed this forward — sleep decision</option><option value="all">All observations — includes commitment forward</option></select></label>
<p class="muted">Delay 0 means committed in the same forward; delay 1 means the next forward. Confidence buckets include their lower endpoint and exclude their upper endpoint, except 100% is included in the last bucket. Ten-point buckets are coarse: compare the actual mean confidence within each entropy band before attributing differences to entropy.</p>
<section><img id="overview" alt="Mean and median by confidence bucket"><img id="median" alt="Median time to commitment"><p class="muted">Hatched median bars have fewer than 20 contributing questions or 100 observations; all counts remain visible in the tables. Blank heatmap cells have no observations.</p><img id="within2" alt="Percentage committed within two forwards"></section>
<section><label>Confidence bucket<select id="bucket">__OPTIONS__</select></label><img id="cdf" alt="Cumulative time to commitment"><table><thead><tr><th>Entropy</th><th>Observations</th><th>Questions</th><th>Mean p</th><th>Median wait</th><th>≤1</th><th>≤2</th><th>≤4</th><th>≤8</th><th>≤16</th></tr></thead><tbody id="rows"></tbody></table><p class="muted">CDF denominators are all observations in the selected cell. All positions have observed commitments, so curves reach 100%. These are model-chosen commitments, not reference-correctness events. The all-entropy curve overlaps its component bands.</p></section>
<section><h2>Newly downloaded sleep runs</h2><ul>__CARDS__</ul><p class="muted">Both new runs share the original sample-file hash and report RTX 4090. Their entropy/confidence observations are saved only at sleep assignment, so they are NOT used for the full confidence-bucket plots above. See sleep_diagnostics.json and sleep_questions.json for actual wake timing and paired outcomes. Historical TOP1 traces are a separate baseline generation run; it is not the latest 54% vanilla output.</p></section>
<section><h2>Questions where the two sleep variants differ in correctness</h2>__EXAMPLES__</section>
<script>const payload=__PAYLOAD__;function draw(){const p=document.querySelector('#policy').value,s=document.querySelector('#scope').value,b=+document.querySelector('#bucket').value;for(const k of ['overview','median','within2'])document.querySelector('#'+k).src=payload.images[p+'_'+s+'_'+k];document.querySelector('#cdf').src=payload.images[p+'_'+s+'_cdf_'+b];const cells=payload.data[p][s].filter(c=>c.confidence_bucket===b);const body=document.querySelector('#rows');body.replaceChildren();for(const c of cells){const tr=document.createElement('tr');const values=[c.entropy_label,c.observations.toLocaleString(),c.questions,c.mean_confidence===null?'—':(100*c.mean_confidence).toFixed(2)+'%',c.median===null?'—':c.median.toFixed(1),...[1,2,4,8,16].map(h=>c.within[h]===undefined?'—':c.within[h].toFixed(2)+'%')];for(const v of values){const td=document.createElement('td');td.textContent=v;tr.append(td)}body.append(tr)}}for(const id of ['policy','scope','bucket'])document.querySelector('#'+id).onchange=draw;draw();</script></html>'''
    options=''.join(f'<option value="{b}"'+(' selected' if b==2 else '')+f'>{10*b}–{10*(b+1)}%</option>' for b in range(10))
    (out/'index.html').write_text(document.replace('__OPTIONS__',options).replace('__CARDS__',cards).replace('__EXAMPLES__',''.join(examples)).replace('__PAYLOAD__',payload),encoding='utf-8')


def generate_readiness_report(top1, threshold, sleep, out):
    """Read saved records, pair observations to commits, summarize, plot and export."""
    out = Path(out); out.mkdir(parents=True,exist_ok=True)
    data, cohorts = {}, {}
    for policy,path in (('top1',top1),('threshold',threshold)):
        hist,sums,questions = collect_baseline(path,policy)
        data[policy] = summarize(hist,sums)
        cohorts[policy] = questions
        np.savez_compressed(out/f'{policy}_question_histograms.npz', histograms=hist, sums=sums,
                            question_ids=np.array(list(questions)))
    if set(cohorts['top1']) != set(cohorts['threshold']):
        raise ValueError('Baseline policy cohorts differ')
    diagnostics, sleep_questions = sleep_diagnostics(sleep,cohorts['top1'])
    definitions = dict(
        scope='Every fresh eligible observation of final nonspecial pre-stop answer positions; retrospective final filtering.',
        primary_population='waiting: positions not committed on the observation forward; all includes zero-delay commitment observations.',
        delay='actual_commit_step - observation_step; 0 = same forward, 1 = next forward.',
        uncertainty='Full-vocabulary entropy in nats; highest non-MASK probability without renormalization.',
        bootstrap='1000 resamples of entire questions; pooled-observation CDF, 95% percentile intervals.',
        weighting='All snapshots pooled; repeated positions are intentional. No first-entry selection. Uncertainty clusters by question.',
        limits='10-point confidence bins do not exactly match confidence, decoding progress or position. Observational association is not entropy incremental predictive value, causal benefit or cache safety.',
        sleep='Actual sleep diagnostics use assignment-only observations, separately from baseline buckets. Wake-to-commit is measured on treatment trajectories, not a counterfactual.',
        files={key:str(Path(value).resolve()) for key,value in [('top1',top1),('threshold',threshold),('sleep',sleep)]})
    for filename,value in [('readiness.json',dict(definitions=definitions,policies=data)),
                           ('sleep_diagnostics.json',diagnostics),('sleep_questions.json',sleep_questions),
                           ('baseline_cohorts.json',cohorts)]:
        (out/filename).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    with (out/'readiness.csv').open('w',newline='') as f:
        fields=['policy','population','confidence_range','entropy_label','observations','questions',
                'mean_confidence','mean_entropy','median','mean','p90',*[f'within_{h}_percent' for h in HORIZONS]]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for policy,scopes in data.items():
            for scope,cells in scopes.items():
                for cell in cells:
                    writer.writerow(dict(policy=policy,population=scope,
                        **{k:cell[k] for k in fields if k in cell},
                        **{f'within_{h}_percent':cell['within'].get(str(h)) for h in HORIZONS}))
    images=plot_readiness(data,out)
    write_html(data,diagnostics,images,out,sleep_questions)
    provenance=dict(analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    bootstrap_seed=1729,question_counts={p:len(q) for p,q in cohorts.items()},
                    baseline_legacy_accuracy={p:sum(q['legacy_correct'] for q in qs.values())/len(qs) for p,qs in cohorts.items()})
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Report:',out/'index.html',flush=True)
    return data,diagnostics


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('top1','threshold','sleep','out'):
        parser.add_argument('--'+flag,type=Path,required=True)
    args=parser.parse_args()
    generate_readiness_report(args.top1,args.threshold,args.sleep,args.out)
