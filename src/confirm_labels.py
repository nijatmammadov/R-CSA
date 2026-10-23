import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb_common as pc


def parse_set(items):
    out = {}
    for it in items or []:
        if '=' not in it:
            raise SystemExit('--set needs id=label, got %r' % it)
        i, v = it.split('=', 1)
        v = v.strip().lower()
        out[i.strip()] = 'exclude' if v == 'exclude' else int(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('solved', help='pb_solve.py output with label_auto filled in')
    ap.add_argument('--set', action='append', default=[], metavar='ID=LABEL',
                    help='correction: a 0-indexed step, -1 (no wrong step) or exclude; repeatable')
    a = ap.parse_args()

    d = json.load(open(a.solved, encoding='utf-8'))
    fix = parse_set(a.set)
    ids = {r['id'] for r in d}
    unknown = sorted(set(fix) - ids)
    if unknown:
        raise SystemExit('ids not in %s: %s' % (a.solved, unknown))
    ok = [r for r in d if r.get('status') == 'ok']
    missing = [r['id'] for r in ok if 'label_auto' not in r and r['id'] not in fix]
    if missing:
        raise SystemExit('%d chains have no label_auto and no --set: %s' % (len(missing), missing[:10]))

    changed = 0
    for r in ok:
        v = fix.get(r['id'], r.get('label_auto'))
        if v != 'exclude' and not (v == -1 or 0 <= v < len(r['steps'])):
            raise SystemExit('%s: label %r is outside 0..%d' % (r['id'], v, len(r['steps']) - 1))
        changed += r['id'] in fix and v != r.get('label_auto')
        r['label'] = None if v == 'exclude' else v
        r['label_note'] = ('corrected by hand from label_auto %r' % r.get('label_auto')
                           if r['id'] in fix and v != r.get('label_auto') else 'label_auto confirmed by hand')
        if v == 'exclude':
            r['label_note'] += ' (excluded)'
    pc.save_json(a.solved, d)

    lab = [r['label'] for r in ok]
    print('%s: %d chains | %d with a first wrong step, %d clean (-1), %d excluded | %d corrected by hand'
          % (a.solved, len(ok), sum(isinstance(x, int) and x >= 0 for x in lab), sum(x == -1 for x in lab),
             sum(x is None for x in lab), changed))


if __name__ == '__main__':
    main()
