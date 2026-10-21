import os
import sys
import json
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_math_questions import integer_answer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD_AIME = [os.path.join(ROOT, 'data', 'aime_trials', f)
            for f in ('aime135.json', 'aimeh_solved.json')]


def norm(text):
    return ' '.join(str(text).split())


def strip_answer(a):
    a = str(a).strip()
    if a.startswith('\\boxed{') and a.endswith('}'):
        a = a[len('\\boxed{'):-1]
    return integer_answer(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from datasets import load_dataset

    seen = set()
    pb = load_dataset('Qwen/ProcessBench')
    for split in pb:
        seen.update(norm(r['problem']) for r in pb[split])
    n_pb = len(seen)
    for f in OLD_AIME:
        if not os.path.exists(f):
            raise SystemExit('missing %s (needed to exclude the AIME problems of earlier trials)' % f)
        seen.update(norm(r['question']) for r in json.load(open(f, encoding='utf-8')))
    print('excluding %d ProcessBench problems and %d AIME problems of earlier trials'
          % (n_pb, len(seen) - n_pb))

    rows = list(load_dataset('KbsdJames/Omni-MATH', split='test'))
    need = {'problem', 'answer', 'solution', 'difficulty', 'domain', 'source'}
    if not need <= set(rows[0]):
        raise SystemExit('Omni-MATH fields are %s, expected %s' % (sorted(rows[0]), sorted(need)))
    keep, n_int, n_seen = [], 0, 0
    for i, r in enumerate(rows):
        v = strip_answer(r['answer'])
        if v is None:
            continue
        n_int += 1
        if norm(r['problem']) in seen:
            n_seen += 1
            continue
        keep.append({'id': 'omni_%04d' % i, 'question': r['problem'],
                     'gsm8k_answer': str(r['solution']) + '\n#### %d' % v, 'reference': [],
                     'difficulty': r['difficulty'], 'domain': r['domain'], 'competition': r['source'],
                     'source': 'Omni-MATH (not ProcessBench, not earlier AIME trials)'})
    print('%d problems, %d with an integer answer, %d of those excluded, %d eligible'
          % (len(rows), n_int, n_seen, len(keep)))
    if len(keep) < a.n:
        raise SystemExit('only %d eligible problems, asked for %d' % (len(keep), a.n))
    random.Random(a.seed).shuffle(keep)
    out = sorted(keep[:a.n], key=lambda q: q['id'])
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    by = {}
    for q in out:
        by[q['difficulty']] = by.get(q['difficulty'], 0) + 1
    print('wrote %d -> %s; by difficulty: %s' % (len(out), a.out,
                                                 ', '.join('%s: %d' % kv for kv in sorted(by.items()))))


if __name__ == '__main__':
    main()
