import os
import re
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb_common as pc

_NUM = re.compile(r'\d[\d,]*(?:\.\d+)?')


def known_values(question, ref):
    base = {float(x.replace(',', '')) for x in _NUM.findall(question)} | {r['value'] for r in ref}
    out = set(base)
    for v in base:
        out |= {v * 100, v / 100, v * 60, v / 60}
    return out


def suggest(steps, known):
    for k, s in enumerate(steps):
        v = pc.step_result(s)
        if v is None:
            continue
        if not any(abs(v - w) <= 1e-6 * max(1.0, abs(w)) or abs(v - w) < 0.011 for w in known):
            return k, v
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solved', required=True)
    ap.add_argument('--questions', required=True)
    a = ap.parse_args()
    qs = {q['id']: q for q in json.load(open(a.questions, encoding='utf-8'))}
    recs = json.load(open(a.solved, encoding='utf-8'))
    n = 0
    for r in recs:
        if r.get('status') != 'ok' or r.get('answer_correct'):
            continue
        q = qs[r['id']]
        k, v = suggest(r['steps'], known_values(q['question'], q.get('reference') or []))
        r['label_auto'] = k
        r['label_auto_note'] = ('step %d concludes %g, not a question number or reference value' % (k, v)
                                if k is not None else 'no suspect value: every step value is in the reference')
        n += 1
    pc.save_json(a.solved, recs)
    print('suggested first wrong steps for %d wrong chains -> label_auto in %s' % (n, a.solved))


if __name__ == '__main__':
    main()
