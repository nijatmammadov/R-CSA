import re
import json
import random
import argparse

_CALC = re.compile(r'<<([^=<>]*)=([^<>]*)>>')


def reference(gsm8k_answer):
    out = []
    for expr, val in _CALC.findall(gsm8k_answer):
        try:
            out.append({'step': expr.strip(), 'value': float(val.replace(',', '').strip())})
        except ValueError:
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='gsm8k_train_dev_questions.json')
    a = ap.parse_args()
    from datasets import load_dataset
    train = load_dataset('openai/gsm8k', 'main', split='train')
    idx = list(range(len(train)))
    random.Random(a.seed).shuffle(idx)
    qs = []
    for i in sorted(idx[:a.n]):
        r = train[i]
        if '####' not in r['answer']:
            raise SystemExit('train row %d has no #### answer' % i)
        qs.append({'id': 'trn_%04d' % i, 'question': r['question'], 'gsm8k_answer': r['answer'],
                   'reference': reference(r['answer']), 'source': 'gsm8k-train'})
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(qs, f, ensure_ascii=False, indent=1)
    print('wrote %d GSM8K train questions -> %s' % (len(qs), a.out))


if __name__ == '__main__':
    main()
