import os
import json
import random
import argparse


def load(src):
    rows = []
    for name in ('train.json', 'dev.json'):
        path = os.path.join(src, name)
        if not os.path.exists(path):
            raise SystemExit('missing %s' % path)
        rows += json.load(open(path, encoding='utf-8'))
    seen, out = set(), []
    for r in rows:
        key = ' '.join(r['question'].lower().split())
        if key in seen:
            continue
        seen.add(key)
        if not isinstance(r['answer'], bool):
            raise SystemExit('non-boolean answer for %s: %r' % (r['qid'], r['answer']))
        out.append(r)
    return out


def record(r, source):
    gold = 'yes' if r['answer'] else 'no'
    return {'id': 'sqa_' + r['qid'], 'question': r['question'], 'gold': gold,
            'gsm8k_answer': '\n'.join(r['facts']) + '\n#### ' + gold,
            'decomposition': r['decomposition'], 'facts': r['facts'],
            'term': r['term'], 'description': r['description'], 'source': source}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='folder with train.json and dev.json of the StrategyQA repo')
    ap.add_argument('--n-dev', type=int, default=200)
    ap.add_argument('--n-test', type=int, default=400)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-dev', required=True)
    ap.add_argument('--out-test', required=True)
    a = ap.parse_args()

    rows = sorted(load(a.src), key=lambda r: r['qid'])
    if len(rows) < a.n_dev + a.n_test:
        raise SystemExit('only %d questions, asked for %d' % (len(rows), a.n_dev + a.n_test))
    random.Random(a.seed).shuffle(rows)
    dev = rows[:a.n_dev]
    test = rows[a.n_dev:a.n_dev + a.n_test]
    assert not {r['qid'] for r in dev} & {r['qid'] for r in test}

    for part, out, src in ((dev, a.out_dev, 'StrategyQA train (dev sample)'),
                           (test, a.out_test, 'StrategyQA train (test sample, disjoint from dev)')):
        recs = sorted((record(r, src) for r in part), key=lambda q: q['id'])
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(recs, f, ensure_ascii=False, indent=1)
        yes = sum(q['gold'] == 'yes' for q in recs)
        print('wrote %d questions (%d yes, %d no) -> %s' % (len(recs), yes, len(recs) - yes, out))
    print('pool: %d labelled questions' % len(rows))


if __name__ == '__main__':
    main()
