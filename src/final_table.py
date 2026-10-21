import os
import sys
import json
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_pb as S

GRID_CONF = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
GRID_RS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
GRID_DELTA = [-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3]
GRID_TAU = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]
GRID_W = [0.0, 0.5, 1.0, 2.0]


def auroc(pos, neg):
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def fmt(x, d=2):
    return '—' if x is None else ('%.' + str(d) + 'f') % x


def simple_conf(frow, _p=None):
    v = [st['conf'] for st in (frow or {}).get('statements') or []]
    return min(v) if v else None


def feng(frow, w):
    v = [st['sens'] + w * st['var'] for st in (frow or {}).get('statements') or []]
    return min(v) if v else None


def ours_rs(orow, _p=None):
    real = [t for t in S.real_probes(orow) if orow['probe_conf'].get(t) is not None]
    if orow.get('conf') is None or not real:
        return None
    return sum(abs(orow['conf'] - orow['probe_conf'][t]) for t in real) / len(real)


def ours_delta(orow, _p=None):
    real = [t for t in S.real_probes(orow) if orow['probe_conf'].get(t) is not None]
    if orow.get('conf') is None or not real:
        return None
    return orow['conf'] - max(orow['probe_conf'][t] for t in real)


NATURAL = [(-1e-9, None)]


def final_confirmed(r, _p=None):
    if not r or r.get('d') is None:
        return None
    return r['d'] if (r['d'] < 0 and r.get('pair') is not None and r['pair'] < 0.5) else 1.0


def final_screen(r, _p=None):
    return r.get('d') if r else None


def final_ptrue(r, _p=None):
    return r.get('conf') if r else None


METHODS = [
    ('Simple Confidence', 'feng', simple_conf, [(t, None) for t in GRID_CONF]),
    ('Feng (2025)', 'feng', feng, [(t, w) for t in GRID_TAU for w in GRID_W]),
]
FINAL_METHODS = [
    ('**CSS (ours)**', 'final', final_confirmed, NATURAL),
]
ABLATIONS = [
    ('Ablation: full steps, RS score', 'ours', ours_rs, [(t, None) for t in GRID_RS]),
    ('Ablation: full steps, no confirmation', 'ours', ours_delta, [(t, None) for t in GRID_DELTA]),
]
FINAL_ABLATIONS = [
    ('Ablation: P(TRUE) only, no counterfactuals', 'final', final_ptrue, [(t, None) for t in GRID_CONF]),
    ('Ablation: no confirmation', 'final', final_screen, NATURAL),
]


def step_scores(chain, source, fn, param):
    rows = chain[source]
    return [(i, fn(rows.get(i), param)) for i in range(chain['n_steps'])]


def predict(chain, source, fn, t, param):
    for i, v in step_scores(chain, source, fn, param):
        if v is not None and v <= t:
            return i
    return -1


def stats(chains, preds):
    wrong = [c for c in chains if c['label'] != -1]
    cor = [c for c in chains if c['label'] == -1]
    tp = sum(preds[c['id']] != -1 for c in wrong)
    tn = sum(preds[c['id']] == -1 for c in cor)
    fp = len(cor) - tn
    exact = sum(preds[c['id']] == c['label'] for c in wrong)
    near = sum(preds[c['id']] != -1 and abs(preds[c['id']] - c['label']) <= 1 for c in wrong)
    dist = [abs(preds[c['id']] - c['label']) for c in wrong if preds[c['id']] != -1]
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / len(wrong) if wrong else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else 0.0
    a1 = exact / len(wrong) if wrong else 0.0
    a2 = tn / len(cor) if cor else 0.0
    return {'n_wrong': len(wrong), 'n_cor': len(cor), 'flagged_wrong': tp, 'clean': tn,
            'acc': (tp + tn) / len(chains) if chains else None, 'prec': prec, 'rec': rec,
            'spec': a2 if cor else None, 'f1': f1, 'exact': exact, 'near': near,
            'dist': sum(dist) / len(dist) if dist else None,
            'pbf1': 2 * a1 * a2 / (a1 + a2) if a1 + a2 else 0.0}


def chain_auroc(chains, source, fn, param):
    def score(c):
        v = [x for _, x in step_scores(c, source, fn, param) if x is not None]
        return min(v) if v else float('inf')
    return auroc([-score(c) for c in chains if c['label'] != -1],
                 [-score(c) for c in chains if c['label'] == -1])


def step_auroc(chains, source, fn, param):
    pos, neg = [], []
    for c in chains:
        for i, v in step_scores(c, source, fn, param):
            if v is None or (c['label'] != -1 and i > c['label']):
                continue
            (pos if i == c['label'] else neg).append(-v)
    return auroc(pos, neg), len(pos), len(neg)


def split(chains, frac, seed=42):
    dev = set()
    for cls in (True, False):
        ids = sorted(c['id'] for c in chains if (c['label'] != -1) == cls)
        random.Random('%d|%s' % (seed, cls)).shuffle(ids)
        dev |= set(ids[:round(frac * len(ids))])
    return dev


def tune(chains, source, fn, grid):
    best = None
    for t, p in grid:
        st = stats(chains, {c['id']: predict(c, source, fn, t, p) for c in chains})
        key = (st['pbf1'], st['f1'])
        if best is None or key > best[0]:
            best = (key, t, p)
    return best[1], best[2]


def boot(chains, preds, keys, base_preds=None, n=1000, seed=42):
    rng = random.Random(seed)
    wrong = [c for c in chains if c['label'] != -1]
    cor = [c for c in chains if c['label'] == -1]
    vals = {k: [] for k in keys}
    diffs = {k: [] for k in keys}
    for _ in range(n):
        smp = [rng.choice(wrong) for _ in wrong] + [rng.choice(cor) for _ in cor]
        a = stats(smp, preds)
        b = stats(smp, base_preds) if base_preds else None
        for k in keys:
            vals[k].append(a[k] or 0.0)
            if b:
                diffs[k].append((a[k] or 0.0) - (b[k] or 0.0))
    out = {}
    for k in keys:
        v = sorted(vals[k])
        d = sorted(diffs[k]) if diffs[k] else None
        out[k] = ((v[int(0.025 * n)], v[int(0.975 * n) - 1]),
                  (d[int(0.025 * n)], d[int(0.975 * n) - 1], sum(x <= 0 for x in d) / n) if d else None)
    return out


def load_final(final_path, confirm_path):
    from pb_confirm import screen
    runs = {r['id']: r for r in json.load(open(final_path, encoding='utf-8')) if r.get('status') == 'ok'}
    conf = {r['id']: {x['i']: x for x in r['steps']}
            for r in json.load(open(confirm_path, encoding='utf-8')) if r.get('status') == 'ok'}
    out = {}
    for i, r in runs.items():
        if i not in conf:
            continue
        rows = {}
        for row in r['steps']:
            d, _ = screen(row)
            rows[row['i']] = {'d': d, 'conf': row.get('conf'), 'pair': (conf[i].get(row['i']) or {}).get('pair_mean')}
        out[i] = rows
    return out


def load(labels_path, ours_path, feng_path, alt_labels=False, final=None):
    labels = {r['id']: r for r in json.load(open(labels_path, encoding='utf-8'))}
    ours = {r['id']: r for r in json.load(open(ours_path, encoding='utf-8'))}
    fg = {r['id']: r for r in json.load(open(feng_path, encoding='utf-8'))}
    notes, chains = [], []
    unlabelled = [i for i, r in labels.items() if r.get('status') == 'ok' and r.get('label') is None]
    if unlabelled:
        notes.append('%d solved chains have no label yet and are left out: %s'
                     % (len(unlabelled), ', '.join(sorted(unlabelled)[:20])))
    missing = []
    for i, lab in labels.items():
        if lab.get('status') != 'ok' or lab.get('label') is None:
            continue
        o, f = ours.get(i), fg.get(i)
        if (not o or o.get('status') != 'ok' or not f or f.get('status') != 'ok'
                or (final is not None and i not in final)):
            missing.append(i)
            continue
        label = lab.get('label_alt', lab['label']) if alt_labels else lab['label']
        chains.append({'id': i, 'label': label, 'n_steps': len(o['steps']),
                       'ours': {row['i']: row for row in o['steps']},
                       'feng': {row['i']: row for row in f['steps']},
                       'final': (final or {}).get(i, {})})
    if missing:
        notes.append('%d labelled chains lack a finished run of every method and are left out: %s'
                     % (len(missing), ', '.join(sorted(missing)[:20])))
    return chains, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', required=True, help='pb_solve.py output with first-error labels')
    ap.add_argument('--ours', required=True, help='pb_selfprobe.py --probe-style alt --conf ptrue output')
    ap.add_argument('--feng', required=True, help='pb_feng.py output on the same chains')
    ap.add_argument('--final', default=None, help='pb_selfprobe.py --probe-style alt3 (or alt2) --shorten run')
    ap.add_argument('--final-confirm', default=None, help='pb_confirm.py output for --final')
    ap.add_argument('--dev-frac', type=float, default=0.3)
    ap.add_argument('--alt-labels', action='store_true',
                    help="use 'label_alt' where a chain has one: the first step with a wrong NUMBER "
                         "instead of the first step stating anything false (a sensitivity check)")
    ap.add_argument('--ablations', action='store_true', help='also show ablation rows (not methods)')
    ap.add_argument('--md', default=None)
    a = ap.parse_args()

    final = load_final(a.final, a.final_confirm) if a.final else None
    methods = METHODS + (FINAL_METHODS if final else []) + \
        ((ABLATIONS + (FINAL_ABLATIONS if final else [])) if a.ablations else [])
    chains, notes = load(a.labels, a.ours, a.feng, a.alt_labels, final)
    dev_ids = split(chains, a.dev_frac)
    dev = [c for c in chains if c['id'] in dev_ids]
    test = [c for c in chains if c['id'] not in dev_ids]
    nw = sum(c['label'] != -1 for c in chains)
    L = ['# Results: Simple Confidence vs Feng (2025) vs CSS (ours)'
         + (' (alternative labels: first wrong number)' if a.alt_labels else ''), '',
         '%d chains (%d wrong, %d correct). Thresholds chosen on a seeded %d%% dev split (%d chains); '
         'the main table is on the held-out %d chains (%d wrong, %d correct).'
         % (len(chains), nw, len(chains) - nw, round(100 * a.dev_frac), len(dev), len(test),
            sum(c['label'] != -1 for c in test), sum(c['label'] == -1 for c in test)), '']

    res = []
    for name, src, fn, grid in methods:
        t, p = tune(dev, src, fn, grid)
        preds = {c['id']: predict(c, src, fn, t, p) for c in test}
        res.append((name, src, fn, t, p, preds, stats(test, preds)))
    base = next(r for r in res if r[0] == 'Feng (2025)')[5]

    L += ['## Detection and localisation (held-out chains)', '',
          '| Method | Threshold | Accuracy | Precision | Recall | Specificity | F1 | Exact first error | Within ±1 | Mean distance | ProcessBench F1 |',
          '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for name, src, fn, t, p, preds, st in res:
        L.append('| %s | %s | %s | %s | %s | %s | %s | %d/%d | %d/%d | %s | %s |' % (
            name, ('t=%g' % t) + ('' if p is None else ', w=%g' % p), fmt(st['acc']), fmt(st['prec']),
            fmt(st['rec']), fmt(st['spec']), fmt(st['f1']), st['exact'], st['n_wrong'], st['near'],
            st['n_wrong'], fmt(st['dist']), fmt(st['pbf1'])))
    L += ['']

    L += ['## Confusion matrices (held-out chains; positive = wrong chain)', '',
          '| Method | TP (wrong, flagged) | FN (wrong, missed) | FP (correct, flagged) | TN (correct, clear) |',
          '| --- | --- | --- | --- | --- |']
    for name, src, fn, t, p, preds, st in res:
        L.append('| %s | %d | %d | %d | %d |' % (name, st['flagged_wrong'], st['n_wrong'] - st['flagged_wrong'],
                                               st['n_cor'] - st['clean'], st['clean']))
    L += ['']

    nw_t = sum(c['label'] != -1 for c in test)
    L += ['## Same rules at 50% wrong chains (as in Feng\'s setting)' , '',
          'Precision and F1 depend on how many chains are wrong (%d%% here). Recall and specificity do '
          'not, so these columns are what the same thresholds give when half the chains are wrong: '
          'precision = recall / (recall + 1 - specificity). Computed, not measured. Flagging every chain '
          'scores F1 0.67 here.' % round(100 * nw_t / len(test)), '',
          '| Method | Balanced accuracy | Precision | Recall | F1 |', '| --- | --- | --- | --- | --- |']
    for name, src, fn, t, p, preds, st in res:
        r_, s_ = st['rec'] or 0.0, st['spec'] or 0.0
        pr = r_ / (r_ + 1 - s_) if r_ + 1 - s_ > 0 else None
        f = 2 * pr * r_ / (pr + r_) if pr and r_ else 0.0
        L.append('| %s | %s | %s | %s | %s |' % (name, fmt((r_ + s_) / 2), fmt(pr), fmt(r_), fmt(f)))
    L += ['']

    L += ['## 95% CIs and paired difference against Feng (held-out chains)', '',
          '| Method | F1 [CI] | F1 − Feng [CI], P(≤0) | ProcessBench F1 [CI] | PB F1 − Feng [CI], P(≤0) |',
          '| --- | --- | --- | --- | --- |']
    for name, src, fn, t, p, preds, st in res:
        b = boot(test, preds, ['f1', 'pbf1'], None if name == 'Feng (2025)' else base)
        cells = []
        for k in ('f1', 'pbf1'):
            (lo, hi), d = b[k]
            cells.append('%s [%s, %s]' % (fmt(st[k]), fmt(lo), fmt(hi)))
            cells.append('—' if d is None else '%+.2f [%+.2f, %+.2f], %.3f'
                         % ((st[k] - stats(test, base)[k],) + d))
        L.append('| %s | %s |' % (name, ' | '.join(cells)))
    L += ['']

    L += ['## Threshold-free (all %d chains)' % len(chains), '',
          '| Method | Chain AUROC (detection) | Step AUROC (first wrong step vs correct steps) |',
          '| --- | --- | --- |']
    for name, src, fn, t, p, preds, st in res:
        sa, npos, nneg = step_auroc(chains, src, fn, p)
        L.append('| %s | %s | %s (%d / %d steps) |' % (name, fmt(chain_auroc(chains, src, fn, p)),
                                                    fmt(sa), npos, nneg))
    L += ['']
    L += ['- %s' % n for n in notes]
    text = '\n'.join(L)
    print(text)
    if a.md:
        open(a.md, 'w', encoding='utf-8').write(text + '\n')


if __name__ == '__main__':
    main()
