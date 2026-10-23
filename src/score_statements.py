import os
import sys
import json
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from final_table import GRID_W, auroc

FENG_TABLE2 = [('Simple Confidence', 0.720, 0.695, 0.748, 0.721), ('SelfCheckGPT', 0.773, 0.759, 0.789, 0.774),
               ('Counterfactual Probing (Feng)', 0.850, 0.833, 0.800, 0.816)]


def stats(items, flagged):
    tp = sum(1 for r in items if r['hallucinated'] and flagged[r['id']])
    fp = sum(1 for r in items if not r['hallucinated'] and flagged[r['id']])
    fn = sum(1 for r in items if r['hallucinated'] and not flagged[r['id']])
    tn = len(items) - tp - fp - fn
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'acc': (tp + tn) / len(items), 'prec': pr, 'rec': rc,
            'f1': 2 * pr * rc / (pr + rc) if pr + rc else 0.0}


def tune(items, score, params):
    best = None
    for p in params:
        vals = sorted({score(r, p) for r in items})
        for t in vals:
            s = stats(items, {r['id']: score(r, p) <= t for r in items})
            if best is None or s['f1'] > best[0]:
                best = (s['f1'], t, p)
    return best[1], best[2]


def cv(items, score, params, k=5, seed=42):
    pos = sorted(r['id'] for r in items if r['hallucinated'])
    neg = sorted(r['id'] for r in items if not r['hallucinated'])
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    fold = {i: j % k for j, i in enumerate(pos)}
    fold.update({i: j % k for j, i in enumerate(neg)})
    flagged = {}
    for f in range(k):
        train = [r for r in items if fold[r['id']] != f]
        t, p = tune(train, score, params)
        flagged.update({r['id']: score(r, p) <= t for r in items if fold[r['id']] == f})
    return flagged


def auc_ci(items, score, n=1000, seed=42):
    rng = random.Random(seed)
    pos = [score(r) for r in items if r['hallucinated']]
    neg = [score(r) for r in items if not r['hallucinated']]
    a = auroc([-x for x in pos], [-x for x in neg])
    bs = sorted(auroc([-x for x in rng.choices(pos, k=len(pos))], [-x for x in rng.choices(neg, k=len(neg))])
                for _ in range(n))
    return a, bs[int(0.025 * n)], bs[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', help='pb_feng.py --statements output')
    ap.add_argument('--md', default=None)
    a = ap.parse_args()
    items = [r for r in json.load(open(a.run, encoding='utf-8'))
             if r.get('status') == 'ok' and r.get('sens') is not None and r.get('conf') is not None]
    nh = sum(r['hallucinated'] for r in items)
    methods = [('Simple Confidence', lambda r, p=None: r['conf'], [None]),
               ('Feng (2025)', lambda r, w=0.0: r['sens'] + w * r['var'], list(GRID_W))]
    L = ['# Feng (2025) on standalone TruthfulQA statements (replication check)', '',
         '%d statements (%d misleading, %d truthful) from `%s`, model %s.'
         % (len(items), nh, len(items) - nh, os.path.basename(a.run), items[0].get('model') if items else '?'), '',
         '| Method | Protocol | Setting | TP/FN/FP/TN | Acc | Prec | Rec | F1 | AUROC [95% CI] |', '|' + '---|' * 9]
    for name, score, params in methods:
        t, p = tune(items, score, params)
        s = stats(items, {r['id']: score(r, p) <= t for r in items})
        a0 = auc_ci(items, lambda r: score(r, params[0]) if params[0] is not None else score(r))
        setting = 't=%.3f%s' % (t, '' if p is None else ', w=%s' % p)
        L.append('| %s | Feng\'s (in-sample) | %s | %d/%d/%d/%d | %.3f | %.3f | %.3f | %.3f | %.3f [%.3f, %.3f] |'
                 % ((name, setting, s['tp'], s['fn'], s['fp'], s['tn'], s['acc'], s['prec'], s['rec'], s['f1']) + a0))
        c = stats(items, cv(items, score, params))
        L.append('| %s | 5-fold CV | t%s from 4 folds | %d/%d/%d/%d | %.3f | %.3f | %.3f | %.3f | |'
                 % (name, '' if params == [None] else ', w', c['tp'], c['fn'], c['fp'], c['tn'], c['acc'],
                    c['prec'], c['rec'], c['f1']))
    L += ['', "Feng's Table 2 (his datasets; OpenAI API, model not named; section 5.1 gives F1 0.816 on his "
          "TruthfulQA subset):", '',
          '| Method | Acc | Prec | Rec | F1 |', '|---|---|---|---|---|']
    L += ['| %s | %.3f | %.3f | %.3f | %.3f |' % r for r in FENG_TABLE2]
    txt = '\n'.join(L) + '\n'
    print(txt)
    if a.md:
        os.makedirs(os.path.dirname(a.md) or '.', exist_ok=True)
        open(a.md, 'w', encoding='utf-8').write(txt)


if __name__ == '__main__':
    main()
