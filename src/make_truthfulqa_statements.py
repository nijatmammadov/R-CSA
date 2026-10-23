import csv
import json
import random
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.n % 2:
        raise SystemExit('--n must be even (half truthful, half misleading)')

    rows = list(csv.DictReader(open(a.csv, encoding='utf-8')))
    items = []
    for i, r in enumerate(rows):
        q, best = r['Question'].strip(), r['Best Answer'].strip()
        wrong = [x.strip() for x in r['Incorrect Answers'].split(';') if x.strip()]
        if q and best and wrong:
            items.append((i, q, best, wrong, r['Category'].strip()))
    if len(items) < a.n:
        raise SystemExit('only %d usable questions' % len(items))
    rng = random.Random(a.seed)
    rng.shuffle(items)
    out = []
    for k, (i, q, best, wrong, cat) in enumerate(items[:a.n]):
        hall = k >= a.n // 2
        out.append({'id': 'tqa_%04d' % i, 'question': q, 'category': cat,
                    'statement': rng.choice(wrong) if hall else best, 'hallucinated': int(hall)})
    out.sort(key=lambda x: x['id'])
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('wrote %d statements (%d truthful, %d misleading) from %d questions -> %s'
          % (len(out), sum(not x['hallucinated'] for x in out), sum(x['hallucinated'] for x in out), len(rows), a.out))


if __name__ == '__main__':
    main()
