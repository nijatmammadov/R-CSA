import os
import sys
import json
import math
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_table as F

GRID_CAL = ['raw', 'cf']
GRID_AGG = ['logodds', 'prob']
GRID_LAM = [0.5, 1.0, 2.0]
GRID_M = [0.0, 0.5, 1.0]
GRID_W = [1.0, 2.0, 4.0, 8.0]
GRID_LAM_FULL = [0.0, 0.5, 1.0]
GRID_K = [1.0, 2.0, 4.0, 8.0]
DETECTION_ONLY = {'Simple Confidence', 'Feng (2025)', 'SelfCheckGPT (Manakul et al. 2023)'}


def lg(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def conf(row, cal):
    if not row or row.get('p') is None or (cal == 'cf' and row.get('p_cf') is None):
        return None
    return lg(row['p']) - (lg(row['p_cf']) if cal == 'cf' else 0.0)


def s_ptrue(row, cal):
    return conf(row, cal)


def s_regen(row, _p=None):
    if not row or not row.get('regen'):
        return None
    return row.get('regen_agree', 0) / len(row['regen'])


def delta(row, agg):
    alts = [a['p'] for a in row.get('alts') or [] if a.get('p') is not None]
    if not alts:
        return 0.0
    if agg == 'logodds':
        return lg(row['p']) - sum(lg(x) for x in alts) / len(alts)
    return row['p'] - sum(alts) / len(alts)


def s_css(row, param):
    cal, agg, lam, m = param
    c = conf(row, cal)
    if c is None:
        return None
    return c + lam * delta(row, agg) + m * math.log(row['_n'])


def agreement(row):
    return row.get('regen_agree', 0) / len(row['regen']) if row.get('regen') else 0.5


def s_ptrue_full(row, _p=None):
    return None if not row or row.get('p_full') is None else lg(row['p_full'])


def s_css_full(row, param):
    w, lam, m = param[:3]
    k = param[3] if len(param) > 3 else 0.0
    c = s_ptrue_full(row)
    if c is None:
        return None
    d = delta(row, 'logodds') if lam and row.get('p') is not None else 0.0
    q = k * (1.0 - row['cfq']) if (k and row.get('cfq') is not None) else 0.0
    return c + w * agreement(row) + lam * d + m * math.log(row['_n']) - q


def s_selfcheck(row, _p=None):
    return None if not row or row.get('inconsistency') is None else 1.0 - row['inconsistency']


def s_simple(row, _p=None):
    return F.simple_conf(row)


def s_feng(row, w):
    return F.feng(row, w)


def thresholds(chains, source, fn, param):
    v = sorted({x for c in chains for _, x in F.step_scores(c, source, fn, param) if x is not None})
    if not v:
        return [float('-inf')]
    k = min(60, len(v))
    return [float('-inf')] + [v[int(i * (len(v) - 1) / max(1, k - 1))] for i in range(k)]


def preds_for(chains, source, fn, t, param):
    return {c['id']: F.predict(c, source, fn, t, param) for c in chains}


def tune(chains, source, fn, params, rule='pbf1'):
    best = None
    for p in params:
        for t in thresholds(chains, source, fn, p):
            s = F.stats(chains, preds_for(chains, source, fn, t, p))
            if rule == 'pbf1':
                key = (s['pbf1'], s['f1'] or 0.0)
            else:
                r = float(rule.split(':')[1])
                if s['flagged_wrong'] < r * s['n_wrong'] - 1e-9:
                    continue
                key = (-(s['n_cor'] - s['clean']), s['exact'])
            if best is None or key > best[0]:
                best = (key, t, p)
    if best is None:
        raise SystemExit('no setting reaches %s' % rule)
    return best[1], best[2]


def cv_preds(chains, source, fn, params, rule, k, seed=42):
    import random
    wrong = sorted(c['id'] for c in chains if c['label'] != -1)
    cor = sorted(c['id'] for c in chains if c['label'] == -1)
    rng = random.Random(seed)
    rng.shuffle(wrong)
    rng.shuffle(cor)
    fold = {i: j % k for j, i in enumerate(wrong)}
    fold.update({i: j % k for j, i in enumerate(cor)})
    preds = {}
    for f in range(k):
        train = [c for c in chains if fold[c['id']] != f]
        held = [c for c in chains if fold[c['id']] == f]
        t, p = tune(train, source, fn, params, rule)
        preds.update(preds_for(held, source, fn, t, p))
    return preds


def load(labels_path, css_path, feng_path, critic_path, selfcheck_path=None, ptrue_full_path=None, cfq_path=None):
    labels = {r['id']: r for r in json.load(open(labels_path, encoding='utf-8'))}
    css = {r['id']: r for r in json.load(open(css_path, encoding='utf-8')) if r.get('status') == 'ok'}
    fg = ({r['id']: r for r in json.load(open(feng_path, encoding='utf-8')) if r.get('status') == 'ok'}
          if feng_path else None)
    cr = ({r['id']: r for r in json.load(open(critic_path, encoding='utf-8')) if r.get('status') == 'ok'}
          if critic_path else None)
    sc = ({r['id']: r for r in json.load(open(selfcheck_path, encoding='utf-8')) if r.get('status') == 'ok'}
          if selfcheck_path else None)
    pf = ({r['id']: r for r in json.load(open(ptrue_full_path, encoding='utf-8')) if r.get('status') == 'ok'}
          if ptrue_full_path else None)
    cq = ({r['id']: r for r in json.load(open(cfq_path, encoding='utf-8')) if r.get('status') == 'ok'}
          if cfq_path else None)
    chains, missing, mismatched = [], [], []
    for i, lab in labels.items():
        if lab.get('status') != 'ok' or lab.get('label') is None:
            continue
        if i not in css or (fg is not None and i not in fg) or (cr is not None and i not in cr) \
                or (sc is not None and i not in sc) or (pf is not None and i not in pf) \
                or (cq is not None and i not in cq):
            missing.append(i)
            continue
        n_lab = len(lab.get('steps') or [])
        if len(css[i]['steps']) != n_lab or (fg is not None and len(fg[i]['steps']) != n_lab) \
                or (sc is not None and len(sc[i]['steps']) != n_lab) \
                or (pf is not None and len(pf[i]['steps']) != n_lab) \
                or (cq is not None and len(cq[i]['steps']) != n_lab):
            mismatched.append(i)
            continue
        n = len(css[i]['steps'])
        rows = {}
        for row in css[i]['steps']:
            row = dict(row)
            row['_n'] = n
            if pf is not None:
                row['p_full'] = pf[i]['steps'][row['i']]['p_full']
            if cq is not None:
                row['cfq'] = cq[i]['steps'][row['i']]['cfq']
            rows[row['i']] = row
        c = {'id': i, 'label': lab['label'], 'n_steps': n, 'css': rows}
        if fg is not None:
            c['feng'] = {row['i']: row for row in fg[i]['steps']}
        if sc is not None:
            c['selfcheck'] = {row['i']: row for row in sc[i]['steps']}
        if cr is not None:
            pred = cr[i].get('prediction')
            c['critic'] = -2 if pred is None else pred
        chains.append(c)
    if mismatched:
        raise SystemExit('%d chains have a different number of steps in some method file than in %s '
                         '(run on another version of the chains): %s' % (len(mismatched), labels_path, mismatched[:10]))
    return chains, missing, css


def methods(have_feng, have_critic, have_selfcheck=False, have_full=False, ablations=False, have_cfq=False):
    out = []
    if have_feng:
        out += [('Simple Confidence', 'feng', s_simple, [None]),
                ('Feng (2025)', 'feng', s_feng, list(F.GRID_W))]
    if have_critic:
        out += [('ProcessBench critic (baseline)', 'critic', None, [None])]
    if have_selfcheck:
        out += [('SelfCheckGPT (Manakul et al. 2023)', 'selfcheck', s_selfcheck, [None])]
    v2 = [(c, a, l, m) for c in GRID_CAL for a in GRID_AGG for l in GRID_LAM for m in GRID_M]
    if not have_full:
        return out + [('P(TRUE) alone (baseline)', 'css', s_ptrue, list(GRID_CAL)),
                      ('Regeneration agreement (baseline)', 'css', s_regen, [None]),
                      ('**R-CSA (ours)**', 'css', s_css, v2)]
    out += [('P(TRUE) alone (baseline)', 'css', s_ptrue_full, [None]),
            ('Regeneration agreement (baseline)', 'css', s_regen, [None]),
            ('**R-CSA (ours)**', 'css', s_css_full,
             [(w, l, m, k) for w in GRID_W for l in GRID_LAM_FULL for m in GRID_M for k in (GRID_K if have_cfq else [0.0])])]
    if ablations and have_cfq:
        out += [('Ablation: CSS without the counterfactual question test', 'css', s_css_full,
                 [(w, l, m, 0.0) for w in GRID_W for l in GRID_LAM_FULL for m in GRID_M])]
    if ablations:
        out += [('Ablation: CSS without regeneration agreement', 'css', s_css_full,
                 [(0.0, l, m) for l in GRID_LAM for m in GRID_M]),
                ('Ablation: css.py v2 score (lettered judge, short steps)', 'css', s_css, v2)]
    return out


def coverage(css, chains):
    ids = {c['id'] for c in chains}
    lab = {c['id']: c['label'] for c in chains}
    steps = [(r['id'], s) for r in css.values() if r['id'] in ids for s in r['steps']]
    err = [s for i, s in steps if lab[i] == s['i']]
    src = Counter(a['source'] for _, s in steps for a in s['alts'])
    typ = Counter(a['type'] for _, s in steps for a in s['alts'])
    drop = Counter()
    for _, s in steps:
        drop.update(s.get('dropped') or {})
    beat = sum(1 for s in err if any(a['p'] > s['p'] for a in s['alts']))
    return [
        'Steps: %d. Shortened: %d (%.0f%%). With at least one scored step probe: %d (%.0f%%).'
        % (len(steps), sum(s['shortened'] for _, s in steps), 100 * sum(s['shortened'] for _, s in steps) / len(steps),
           sum(bool(s['alts']) for _, s in steps), 100 * sum(bool(s['alts']) for _, s in steps) / len(steps)),
        'First-error steps: %d. With a step probe: %d. With a probe the model scores above the step: %d.'
        % (len(err), sum(bool(s['alts']) for s in err), beat),
        'Step probes by source: %s. By type: %s. Dropped in selection: %s.' % (dict(src), dict(typ), dict(drop)),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', required=True)
    ap.add_argument('--css', required=True, help='css.py output')
    ap.add_argument('--feng', default=None, help='pb_feng.py output on the same chains')
    ap.add_argument('--critic', default=None, help='pb_critic.py output on the same chains')
    ap.add_argument('--selfcheck', default=None, help='pb_selfcheck.py output on the same chains')
    ap.add_argument('--ptrue-full', default=None, help='pb_ptrue_full.py output on the same chains')
    ap.add_argument('--ablations', action='store_true', help='also score the ablation rows')
    ap.add_argument('--cfq', default=None, help='css_cfq.py output on the same chains')
    ap.add_argument('--weights-from', default=None,
                    help='frozen file of another model: keep each method\'s weights from it and tune only the '
                         'threshold t on these chains (per-model calibration)')
    ap.add_argument('--rule', default='pbf1', help="dev selection rule: 'pbf1' or 'recall:0.8' (fewest false "
                                                    "alarms among settings catching >= 80%% of wrong chains)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze', help='DEV: tune here and write the chosen settings to this file')
    g.add_argument('--frozen', help='TEST: apply the settings in this file, tune nothing')
    g.add_argument('--cv', type=int, help='K: no separate test set; each of K folds is predicted with the '
                                          'threshold tuned on the other folds (needs --weights-from)')
    ap.add_argument('--md', default=None)
    a = ap.parse_args()

    chains, missing, css = load(a.labels, a.css, a.feng, a.critic, a.selfcheck, a.ptrue_full, a.cfq)
    if not chains:
        raise SystemExit('no labelled chains with a finished run of every method')
    nw = sum(c['label'] != -1 for c in chains)
    frozen = json.load(open(a.frozen, encoding='utf-8')) if a.frozen else {}
    if a.cv is not None and (not a.weights_from or a.cv < 2):
        raise SystemExit('--cv K needs K >= 2 and --weights-from (weights fixed, only the threshold is tuned)')
    if a.weights_from and a.frozen:
        raise SystemExit('--weights-from tunes thresholds; it cannot be combined with --frozen')
    fixed = json.load(open(a.weights_from, encoding='utf-8')) if a.weights_from else {}
    chosen, res, skipped = {}, [], []
    for name, src, fn, params in methods(a.feng is not None, a.critic is not None, a.selfcheck is not None,
                                     a.ptrue_full is not None, a.ablations, a.cfq is not None):
        if fn and not any(x is not None for c in chains for _, x in F.step_scores(c, src, fn, params[0])):
            skipped.append(name)
            continue
        if src == 'critic':
            preds, t, p = {c['id']: c['critic'] for c in chains}, None, None
        elif a.cv:
            if name not in fixed:
                raise SystemExit('%s missing from %s' % (name, a.weights_from))
            w = fixed[name]['param']
            p = tuple(w) if isinstance(w, list) else w
            preds, t = cv_preds(chains, src, fn, [p], a.rule, a.cv), 'cv'
        else:
            if a.frozen:
                if name not in frozen:
                    raise SystemExit('%s missing from %s' % (name, a.frozen))
                t, p = frozen[name]['t'], frozen[name]['param']
                p = tuple(p) if isinstance(p, list) else p
            else:
                if a.weights_from:
                    if name not in fixed:
                        raise SystemExit('%s missing from %s' % (name, a.weights_from))
                    w = fixed[name]['param']
                    params = [tuple(w) if isinstance(w, list) else w]
                t, p = tune(chains, src, fn, params, a.rule)
            preds = preds_for(chains, src, fn, t, p)
            chosen[name] = {'t': t, 'param': list(p) if isinstance(p, tuple) else p}
        s = F.stats(chains, preds)
        ca = F.chain_auroc(chains, src, fn, p) if fn else None
        sa = F.step_auroc(chains, src, fn, p)[0] if fn else None
        res.append((name, t, p, preds, s, ca, sa))
    if a.freeze:
        with open(a.freeze, 'w', encoding='utf-8') as f:
            json.dump(chosen, f, indent=1)

    mode = ('DEV: %s tuned on these same chains (in-sample, rule %s) and frozen to %s'
            % ('thresholds (weights from %s)' % a.weights_from if a.weights_from else 'settings', a.rule, a.freeze)
            if a.freeze else 'TEST: frozen settings from %s, nothing tuned' % a.frozen if a.frozen else
            '%d-fold cross-validation: each fold predicted with the threshold tuned (rule %s) on the other folds; '
            'weights from %s, not tuned; AUROCs use no threshold' % (a.cv, a.rule, a.weights_from))
    L = ['# R-CSA (ours) vs baselines', '', '%d chains (%d wrong, %d correct). %s.' % (len(chains), nw, len(chains) - nw, mode)]
    if missing:
        L.append('%d labelled chains lack a finished run of every method and are left out.' % len(missing))
    if skipped:
        L.append('Not scored, no value on these chains: %s.' % ', '.join(skipped))
    L += ['', '## Results', '',
          '| Method | Setting | TP/FN/FP/TN | Acc | Prec | Rec | Spec | F1 | Bal. acc | Exact | ±1 | Dist | PB F1 | Chain AUROC | Step AUROC |',
          '|' + '---|' * 15]
    for name, t, p, preds, s, ca, sa in res:
        setting = '—' if t is None else ('t: %d-fold CV%s' % (a.cv, '' if p is None else ', %s' % (p,))
                                         if t == 'cv' else 't=%.2f%s' % (t, '' if p is None else ', %s' % (p,)))
        loc = ['—'] * 4 if name in DETECTION_ONLY else \
            ['%d/%d' % (s['exact'], nw), '%d/%d' % (s['near'], nw), F.fmt(s['dist']), F.fmt(s['pbf1'])]
        L.append('| %s | %s | %d/%d/%d/%d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % tuple([
            name, setting, s['flagged_wrong'], s['n_wrong'] - s['flagged_wrong'], s['n_cor'] - s['clean'], s['clean'],
            F.fmt(s['acc']), F.fmt(s['prec']), F.fmt(s['rec']), F.fmt(s['spec']), F.fmt(s['f1']),
            F.fmt((s['rec'] + s['spec']) / 2)] + loc + [F.fmt(ca), F.fmt(sa)]))
    ours = next(r for r in res if r[0] == '**R-CSA (ours)**')
    L += ['', '## R-CSA (ours) against each baseline: difference, bootstrap 95% CI (1,000, seed 42)', '',
          'ProcessBench F1 for methods that localize; detection F1 for published methods that do not '
          '(%s), which are run as published and compared only on detection.' % ', '.join(sorted(DETECTION_ONLY)), '']
    for name, t, p, preds, s, ca, sa in res:
        if name == ours[0]:
            continue
        k = 'f1' if name in DETECTION_ONLY else 'pbf1'
        b = F.boot(chains, ours[3], [k], preds)[k][1]
        L.append('- vs %s (%s): %+.2f [%+.2f, %+.2f], one-sided p = %.3f'
                 % (name, 'detection F1' if k == 'f1' else 'PB F1', (ours[4][k] or 0) - (s[k] or 0), b[0], b[1], b[2]))
    L += ['', '## Coverage', ''] + ['- ' + x for x in coverage(css, chains)]
    txt = '\n'.join(L) + '\n'
    print(txt)
    if a.md:
        open(a.md, 'w', encoding='utf-8').write(txt)


if __name__ == '__main__':
    main()
