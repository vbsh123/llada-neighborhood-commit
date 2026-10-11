"""CPU-only translation scoring and integrity checks, without model loading."""
import argparse
import gzip
import json
from pathlib import Path

from confidence_geography.run import dump

SCORING_PROTOCOL = 'translation_sacrebleu_chrfpp_bleu_v1'


def score_translation(answer, sample):
    from sacrebleu.metrics import CHRF
    metric = CHRF(word_order=2)
    return {'task': 'translation', 'source_text': sample['source_text'],
            'translation': answer, 'scoring_protocol': SCORING_PROTOCOL,
            'chrfpp': metric.sentence_score(answer, [sample['answer']]).score,
            'chrfpp_signature': str(metric.get_signature()), 'empty_translation': not bool(answer.strip())}


def evaluate(out):
    """Read saved results only; corpus BLEU is not a mean of sentence BLEUs."""
    from sacrebleu.metrics import BLEU, CHRF
    out = Path(out)
    manifest = json.loads((out/'manifest.json').read_text())
    samples = [json.loads(line) for line in (out/'samples.jsonl').read_text().splitlines() if line.strip()]
    scores = {}
    for policy in manifest['config']['policies']:
        results, forward_seconds, histogram = [], 0., {}
        for sample in samples:
            directory = out/policy/'samples'/sample['id']
            result = json.loads((directory/'result.json').read_text())
            if result['sample_id'] != sample['id'] or result['reference'] != sample['answer'] or result['source_text'] != sample['source_text']:
                raise ValueError(f'Mismatched input/result: {directory}')
            steps = 0
            with gzip.open(directory/'trace.jsonl.gz', 'rt', encoding='utf-8') as handle:
                for line in handle:
                    record = json.loads(line)
                    if record['type'] == 'header' and record['config']['policy'] != policy:
                        raise ValueError('Wrong trace policy')
                    if record['type'] != 'step':
                        continue
                    steps += 1
                    forward_seconds += record['forward_seconds']
                    size = len(record['commit_positions'])
                    histogram[size] = histogram.get(size, 0)+1
            if steps != result['steps']:
                raise ValueError('Saved result/trace step counts disagree')
            results.append(result)
        hypotheses = [r['translation'] for r in results]
        references = [[s['answer'] for s in samples]]
        bleu, chrf = BLEU(tokenize='13a'), CHRF(word_order=2)
        b, c = bleu.corpus_score(hypotheses, references), chrf.corpus_score(hypotheses, references)
        item = {'samples': len(samples), 'chrfpp': c.score, 'bleu': b.score,
                'chrfpp_signature': str(chrf.get_signature()), 'bleu_signature': str(bleu.get_signature()),
                'total_forwards': sum(r['steps'] for r in results),
                'mean_forwards': sum(r['steps'] for r in results)/len(results),
                'generation_elapsed_seconds': sum(r['elapsed_seconds'] for r in results),
                'forward_seconds': forward_seconds,
                'commit_batch_histogram': {str(k): v for k,v in sorted(histogram.items())},
                'empty_translations': sum(r['empty_translation'] for r in results),
                'length_limit_count': sum(r['hit_length_limit'] for r in results)}
        scores[policy] = item
        dump(out/policy/'summary.json', item)
        (out/policy/'translations.jsonl').write_text(''.join(json.dumps(
            {'sample_id': s['id'], 'source': s['source_text'], 'reference': s['answer'],
             'translation': r['translation'], 'chrfpp': r['chrfpp']}, ensure_ascii=False)+'\n'
            for s,r in zip(samples,results)), encoding='utf-8')
    report = {'task': 'translation', 'language_pair': 'zho_Hans→eng_Latn', 'policies': scores,
              'limits': 'Ten-example qualitative pilot. Quality scores are not accuracy percentages. '
                        'Generation fills the full response window, including special/post-stop slots. '
                        'Elapsed time includes trace/scoring overhead. References never enter model prompts.'}
    dump(out/'comparison.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    evaluate(parser.parse_args().out)
