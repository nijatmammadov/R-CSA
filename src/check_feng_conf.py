import os
import re
import ast
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb_common as pc

EQ = re.compile(r'([\d\s.()+\-*/x×÷]+?)\s*=\s*(-?\d[\d,]*(?:\.\d+)?)\s*\.?\s*$')


def arithmetic(text):
    m = EQ.search(text.replace('$', ''))
    if not m:
        return None
    e = re.sub(r'(?<=[\d)])\s*[xX]\s*(?=[\d(])', '*', m.group(1).strip().replace('×', '*').replace('÷', '/'))
    if not re.search(r'\d\s*[+\-*/]\s*[\d(]', e):
        return None
    try:
        v, w = pc._arith(ast.parse(e, mode='eval')), float(m.group(2).replace(',', ''))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, RecursionError, TypeError):
        return None
    return abs(v - w) < 1e-6 * max(1.0, abs(v)) or round(v, 2) == round(w, 2)


def odd(v):
    x = round(v * 100, 6)
    return x == int(x) and int(x) % 5 != 0 and int(x) not in (1, 33, 66, 67, 99)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    a = ap.parse_args()
    print('| File | Confidences | 0.0 | 0.0 on right arithmetic | 0.0 on wrong arithmetic '
          '| Identical texts, same confidence | Possible echoed numbers |')
    print('|---|---|---|---|---|---|---|')
    for f in a.files:
        with open(f, encoding='utf-8') as fh:
            recs = [r for r in json.load(fh) if r.get('status') == 'ok']
        n = zeros = n_odd = dup = same = 0
        ar = {True: [0, 0], False: [0, 0]}
        for r in recs:
            for step in r['steps']:
                for st in step['statements']:
                    pairs = [(st['text'], st['conf'])] + [(st['probes'][k], st['probe_conf'][k]) for k in st['probes']]
                    seen = {}
                    for t, c in pairs:
                        n += 1
                        zeros += c == 0
                        n_odd += odd(c)
                        ok = arithmetic(t)
                        if ok is not None:
                            ar[ok][0] += 1
                            ar[ok][1] += c == 0
                        key = t.strip().lower().rstrip('.')
                        if key in seen:
                            dup += 1
                            same += seen[key] == c
                        seen[key] = c
        if not n:
            raise SystemExit('%s: no scored statements' % f)
        pct = lambda k, m: '%.0f%% of %d' % (100 * k / m, m) if m else '—'
        print('| %s | %d | %s | %s | %s | %s | %d (%.1f%%) |' % (
            os.path.basename(f), n, pct(zeros, n), pct(ar[True][1], ar[True][0]), pct(ar[False][1], ar[False][0]),
            pct(same, dup), n_odd, 100 * n_odd / n))


if __name__ == '__main__':
    main()
