import os
import sys
import json
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pb_common import not_a_probe, not_an_alternative, type_matches

MIN_PROBES = 2
LABELS = {}
CONSENSUS = None
SUBSET_ORDER = ['gsm8k', 'math', 'olympiadbench', 'omnimath']
GRID_TAU = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
GRID_CONF = [0.1, 0.3, 0.5, 0.7, 0.9]
GRID_SHARE = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def real_probes(row):
    style = row.get('style')
    rule = not_an_alternative if style in ('alt', 'alt2', 'alt3') else not_a_probe
    out = [t for t, p in row['probes'].items() if not rule(row['text'], p)]
    if style == 'alt2':
        out = [t for t in out if type_matches(t, row['text'], row['probes'][t],
                                              row.get('prior_values') or (), row.get('time_values') or ())]
    return out


def diffs(row, feng=False):
    c = row['conf_feng'] if feng else row['conf']
    pc = row['probe_conf_feng'] if feng else row['probe_conf']
    return [abs(c - pc[t]) for t in real_probes(row) if pc.get(t) is not None]


def first(rec, test):
    for row in rec['steps']:
        if test(row):
            return row['i']
    return -1


def _method(agg, feng=False, gate=True):
    def build(tau, conf=None):
        def t(row):
            d = diffs(row, feng)
            if len(d) < MIN_PROBES or agg(d) > tau:
                return False
            return (not gate) or (row['conf_feng'] if feng else row['conf']) <= conf
        return lambda rec: first(rec, t)
    return build


def _feng(tau, w):
    def t(row):
        d = diffs(row, feng=True)
        if len(d) < MIN_PROBES:
            return False
        vals = [row['conf_feng']] + [row['probe_conf_feng'][p] for p in real_probes(row)
                                     if row['probe_conf_feng'].get(p) is not None]
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        return (sum(d) / len(d)) + w * var <= tau
    return lambda rec: first(rec, t)


def _feng_exact(tau, w):
    def t(row):
        return any(st['sens'] + w * st['var'] <= tau for st in row.get('feng_exact') or [])
    return lambda rec: first(rec, t)


def _mean(d):
    return sum(d) / len(d)


def _conf(field):
    def build(conf):
        return lambda rec: first(rec, lambda row: row.get(field) is not None and row[field] <= conf)
    return build


def _contrastive(key):
    def build(theta):
        def t(row):
            c = row.get(key)
            return c is not None and c['n_options'] >= MIN_PROBES + 1 and c['share'] <= theta
        return lambda rec: first(rec, t)
    return build


FENG_GRID = [(t, w) for t in GRID_TAU + [0.5, 0.6] for w in (0.0, 0.5, 1.0, 2.0)]


def families(has_feng, has_unv, contrastive_keys, has_rederive=False, has_feng_exact=False,
             has_rollout=False):
    pairs = [(a, b) for a in GRID_TAU for b in GRID_CONF]
    f = {
        'directional': (_directional, (0.0,), [(t,) for t in GRID_DIR]),
        'method': (_method(_mean), (0.10, 0.50), pairs),
        'method_min': (_method(min), (0.10, 0.50), pairs),
        'no_probe': (_conf('conf'), (0.50,), [(c,) for c in GRID_CONF]),
    }
    if has_unv:
        f['no_probe_unverified'] = (_conf('conf_unverified'), (0.50,), [(c,) for c in GRID_CONF])
    for key in contrastive_keys:
        f[key] = (_contrastive(key), (0.50,), [(s,) for s in GRID_SHARE])
    if has_rederive:
        f['rederive'] = (_rederive, (0.50,), [(s,) for s in GRID_SHARE])
        f['method_min_or_rederive'] = (_either, (0.10, 0.50, 0.50),
                                       [(a, b, c) for a in GRID_TAU for b in GRID_CONF for c in GRID_SHARE])
    if has_rederive and contrastive_keys:
        ck = 'contrastive_logprob' if 'contrastive_logprob' in contrastive_keys else contrastive_keys[0]
        f['full'] = (_full(ck, True, True, True), (0.10, 0.50, 0.50, 0.50), FULL_GRID)
        f['full_no_min'] = (_full(ck, False, True, True), (0.10, 0.50, 0.50, 0.50), FULL_GRID)
        f['full_no_contrastive'] = (_full(ck, True, False, True), (0.10, 0.50, 0.50, 0.50), FULL_GRID)
        f['full_no_rederive'] = (_full(ck, True, True, False), (0.10, 0.50, 0.50, 0.50), FULL_GRID)
    if has_feng:
        f['feng_same_rule'] = (_method(_mean, feng=True), (0.10, 0.50), pairs)
        f['feng_ported'] = (_method(_mean, feng=True, gate=False), (0.10,), [(t,) for t in GRID_TAU])
        f['feng_our_probes'] = (_feng, (0.10, 1.0), FENG_GRID)
    if has_feng_exact:
        f['feng'] = (_feng_exact, (0.10, 1.0), FENG_GRID)
    if has_rollout:
        f['rollout'] = (_rollout, (0.30,), [(d,) for d in (0.01, 0.30, 0.60, 0.99)])
        f['rollout_value'] = (_rollout_value, (0.34,), [(t,) for t in (0.0, 0.34, 0.5, 0.67)])
        f['feng_outcome'] = (_outcome_sens, (0.10, 0.34),
                             [(a, b) for a in (0.0, 0.10, 0.34, 0.5) for b in (0.0, 0.34, 0.67)])
    return f


def refilter_contrastive(row, c):
    if not c or not c.get('dist'):
        return c
    copies = [t for t in row['probes'] if t not in set(real_probes(row))]
    if not copies:
        return c
    shares = []
    for dist in c['dist']:
        tot = sum(dist.values())
        shares.append((dist.get('original', 0) + sum(dist.get(t, 0) for t in copies)) / tot if tot else 0.0)
    n_copies = sum(1 for t in copies if t in c['dist'][0])
    return dict(c, share=sum(shares) / len(shares), shares=shares, n_options=c['n_options'] - n_copies)


def refilter_votes(row, rd):
    if not rd:
        return rd
    real = set(real_probes(row))
    votes = ['original' if v in row['probes'] and v not in real else v for v in rd['votes']]
    if votes == rd['votes']:
        return rd
    addressed = [v for v in votes if v != 'unaddressed']
    support = sum(v == 'original' for v in addressed) / len(addressed) if addressed else None
    return dict(rd, votes=votes, support=support, n_addressed=len(addressed))


GRID_DIR = [-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5]


def directional(row):
    pc = [row['probe_conf'][t] for t in real_probes(row) if row['probe_conf'].get(t) is not None]
    if len(pc) < MIN_PROBES or row.get('conf') is None:
        return None
    return row['conf'] - max(pc)


def _directional(tau):
    def t(row):
        d = directional(row)
        return d is not None and d <= tau
    return lambda rec: first(rec, t)


def _step_features(rec, row, ck):
    d = diffs(row)
    n = max(len(rec['steps']) - 1, 1)
    c = row.get(ck) if ck else None
    rd = row.get('rederive')
    return {
        'conf': row.get('conf'),
        'sens_mean': sum(d) / len(d) if d else None,
        'sens_min': min(d) if d else None,
        'directional': directional(row),
        'contrastive': c['share'] if c and c.get('n_options', 0) >= MIN_PROBES + 1 else None,
        'rederive': rd['support'] if rd and rd.get('support') is not None and rd['n_addressed'] >= 2 else None,
        'position': row['i'] / n,
        'rollout_gap': rollout_gap(row),
        'rollout_value': (row.get('rollout') or {}).get('orig'),
    }


class Learned:
    def __init__(self, dev_recs, dev_groups, ck, l2=1.0, iters=3000, lr=0.1):
        self.ck = ck
        X, y = [], []
        for r in dev_recs:
            for row in r['steps']:
                if r['label'] != -1 and row['i'] > r['label']:
                    continue
                X.append(_step_features(r, row, ck))
                y.append(1 if row['i'] == r['label'] else 0)
        names = [k for k in X[0] if any(x[k] is not None for x in X)] if X else []
        self.names = names
        self.mean = {k: sum(x[k] for x in X if x[k] is not None) / max(1, sum(x[k] is not None for x in X))
                     for k in names}
        rows = [[(x[k] if x[k] is not None else self.mean[k]) for k in names] for x in X]
        self.sd = {}
        for j, k in enumerate(names):
            v = [r[j] for r in rows]
            m = sum(v) / len(v)
            self.sd[k] = (sum((a - m) ** 2 for a in v) / len(v)) ** 0.5 or 1.0
            self.mean[k] = m
        Z = [[(r[j] - self.mean[k]) / self.sd[k] for j, k in enumerate(names)] for r in rows]
        npos = sum(y) or 1
        wpos = (len(y) - npos) / npos if npos else 1.0
        self.w, self.b = [0.0] * len(names), 0.0
        import math
        for _ in range(iters):
            gw, gb = [0.0] * len(names), 0.0
            for z, t in zip(Z, y):
                p = 1 / (1 + math.exp(-(self.b + sum(a * b for a, b in zip(self.w, z)))))
                e = (p - t) * (wpos if t else 1.0)
                gb += e
                for j in range(len(names)):
                    gw[j] += e * z[j]
            n = len(Z) or 1
            self.b -= lr * gb / n
            self.w = [w - lr * (g / n + l2 * w / n) for w, g in zip(self.w, gw)]
        best = None
        for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            _, f1 = evaluate({r['id']: self.predict(r, th) for r in dev_recs}, dev_groups)
            f1 = -1 if f1 is None else f1
            if best is None or (f1, -abs(th - 0.5)) > (best[0], -abs(best[1] - 0.5)):
                best = (f1, th)
        self.theta = best[1] if best else 0.5

    def prob(self, rec, row):
        import math
        x = _step_features(rec, row, self.ck)
        z = [((x[k] if x[k] is not None else self.mean[k]) - self.mean[k]) / self.sd[k] for k in self.names]
        return 1 / (1 + math.exp(-(self.b + sum(a * b for a, b in zip(self.w, z)))))

    def predict(self, rec, theta=None):
        th = self.theta if theta is None else theta
        return first(rec, lambda row: self.prob(rec, row) >= th)

    def describe(self):
        return 'theta %.1f; standardised weights: %s' % (
            self.theta, ', '.join('%s %+.2f' % (k, w) for k, w in sorted(
                zip(self.names, self.w), key=lambda kw: -abs(kw[1]))))


def rollout_gap(row):
    ro = row.get('rollout')
    if not ro or ro['orig'] is None:
        return None
    ph = [h for t, h in ro['probes'].items() if h is not None and t in set(real_probes(row))]
    return (max(ph) - ro['orig']) if ph else None


def _rollout(delta):
    def t(row):
        g = rollout_gap(row)
        return g is not None and g >= delta
    return lambda rec: first(rec, t)


def _outcome_sens(tau, theta):
    def t(row):
        ro = row.get('rollout')
        if not ro or ro['orig'] is None:
            return False
        real = set(real_probes(row))
        ph = [h for p, h in ro['probes'].items() if p in real and h is not None]
        if not ph:
            return False
        return sum(abs(ro['orig'] - h) for h in ph) / len(ph) <= tau and ro['orig'] <= theta
    return lambda rec: first(rec, t)


def _rollout_value(theta):
    def t(row):
        ro = row.get('rollout')
        return bool(ro) and ro['orig'] is not None and ro['orig'] <= theta
    return lambda rec: first(rec, t)


def _rederive_test(theta):
    def t(row):
        r = row.get('rederive')
        return r is not None and r['support'] is not None and r['n_addressed'] >= 2 and r['support'] <= theta
    return t


def _rederive(theta):
    t = _rederive_test(theta)
    return lambda rec: first(rec, t)


def _either(tau, conf, theta):
    rd = _rederive_test(theta)

    def t(row):
        d = diffs(row)
        orig = len(d) >= MIN_PROBES and min(d) <= tau and row['conf'] <= conf
        return orig or rd(row)
    return lambda rec: first(rec, t)


FULL_GRID = [(a, b, c, d) for a in GRID_TAU for b in GRID_CONF for c in GRID_SHARE for d in GRID_SHARE]


def _full(ck, use_min, use_con, use_rd):
    def build(tau, conf, share_c, share_r):
        rd = _rederive_test(share_r)

        def t(row):
            if use_min:
                d = diffs(row)
                if len(d) >= MIN_PROBES and min(d) <= tau and row['conf'] <= conf:
                    return True
            if use_con:
                c = row.get(ck)
                if c is not None and c['n_options'] >= MIN_PROBES + 1 and c['share'] <= share_c:
                    return True
            return bool(use_rd and rd(row))
        return lambda rec: first(rec, t)
    return build


def stats(items):
    err = [(l, p) for l, p in items if l != -1]
    cor = [(l, p) for l, p in items if l == -1]
    acc_e = sum(p == l for l, p in err) / len(err) if err else None
    acc_c = sum(p == -1 for l, p in cor) / len(cor) if cor else None
    f1 = (2 * acc_e * acc_c / (acc_e + acc_c)
          if acc_e is not None and acc_c is not None and acc_e + acc_c > 0 else
          (0.0 if acc_e is not None and acc_c is not None else None))
    tp = sum(p != -1 for l, p in err)
    fp = sum(p != -1 for l, p in cor)
    near = sum(p != -1 and p is not None and abs(p - l) <= 1 for l, p in err)
    return {'n': len(items), 'n_err': len(err), 'acc_err': acc_e, 'acc_cor': acc_c, 'f1': f1,
            'prec': tp / (tp + fp) if tp + fp else None,
            'rec': tp / len(err) if err else None,
            'spec': 1 - fp / len(cor) if cor else None,
            'within1': near / len(err) if err else None}


def avg_f1(by_subset):
    vals = [s['f1'] for s in by_subset.values() if s['f1'] is not None]
    return sum(vals) / len(vals) if vals else None


def evaluate(preds, groups):
    per = {s: stats([(l, preds[i]) for i, l in g]) for s, g in groups.items()}
    return per, avg_f1(per)


def bootstrap(groups, preds_by_method, base, n=1000, seed=42):
    rng = random.Random(seed)
    dist = {m: [] for m in preds_by_method}
    for _ in range(n):
        sample = {s: [g[rng.randrange(len(g))] for _ in g] for s, g in groups.items()}
        for m, p in preds_by_method.items():
            dist[m].append(evaluate(p, sample)[1])
    out = {}
    for m, d in dist.items():
        ok = sorted(x for x in d if x is not None)
        ci = (ok[int(0.025 * len(ok))], ok[int(0.975 * len(ok)) - 1]) if ok else (None, None)
        diff = [x - y for x, y in zip(dist[base], d) if x is not None and y is not None]
        p = None
        if m != base and diff:
            p = min(1.0, 2 * min(sum(x <= 0 for x in diff), sum(x >= 0 for x in diff)) / len(diff))
        dd = sorted(diff)
        out[m] = {'ci': ci, 'diff_vs_base': (sum(diff) / len(diff)) if diff and m != base else None,
                  'diff_ci': (dd[int(0.025 * len(dd))], dd[int(0.975 * len(dd)) - 1])
                  if dd and m != base else None, 'p': p}
    return out


def fmt(x, d=3):
    return '—' if x is None else ('%.*f' % (d, x))


def load(run_path, critic_path, contrastive_paths, rederive_path=None, feng_path=None, rollout_path=None):
    recs = json.load(open(run_path, encoding='utf-8'))
    ok = {r['id']: r for r in recs if r.get('status') == 'ok'}
    notes = []
    bad = [r['id'] for r in recs if r.get('status') != 'ok']
    if bad:
        notes.append('%d excluded as run errors: %s' % (len(bad), ', '.join(bad[:20])))

    has_unv, ckeys = False, []
    for cp in contrastive_paths:
        crecs = {r['id']: r for r in json.load(open(cp, encoding='utf-8'))}
        mode = next(iter(crecs.values()))['mode'] if crecs else 'x'
        key = 'contrastive_%s' % mode
        ckeys.append(key)
        drop = [i for i in ok if crecs.get(i, {}).get('status') != 'ok']
        if drop:
            notes.append('%d excluded because %s has no completed record: %s'
                         % (len(drop), os.path.basename(cp), ', '.join(drop[:20])))
        for i in drop:
            ok.pop(i)
        for i, r in ok.items():
            by_i = {s['i']: s for s in crecs[i]['steps']}
            for row in r['steps']:
                extra = by_i.get(row['i'], {})
                row[key] = refilter_contrastive(row, extra.get('contrastive'))
                if extra.get('conf_unverified') is not None:
                    row['conf_unverified'] = extra['conf_unverified']
                    has_unv = True

    has_rd = False
    if rederive_path:
        rrecs = {r['id']: r for r in json.load(open(rederive_path, encoding='utf-8'))}
        drop = [i for i in ok if rrecs.get(i, {}).get('status') != 'ok']
        if drop:
            notes.append('%d excluded because %s has no completed record: %s'
                         % (len(drop), os.path.basename(rederive_path), ', '.join(drop[:20])))
        for i in drop:
            ok.pop(i)
        for i, r in ok.items():
            by_i = {s['i']: s for s in rrecs[i]['steps']}
            for row in r['steps']:
                row['rederive'] = refilter_votes(row, by_i.get(row['i'], {}).get('rederive'))
        has_rd = True

    has_fx = False
    if feng_path:
        frecs = {r['id']: r for r in json.load(open(feng_path, encoding='utf-8'))}
        drop = [i for i in ok if frecs.get(i, {}).get('status') != 'ok']
        if drop:
            notes.append('%d excluded because %s has no completed record: %s'
                         % (len(drop), os.path.basename(feng_path), ', '.join(drop[:20])))
        for i in drop:
            ok.pop(i)
        for i, r in ok.items():
            by_i = {s['i']: s for s in frecs[i]['steps']}
            for row in r['steps']:
                row['feng_exact'] = by_i.get(row['i'], {}).get('statements')
        has_fx = True

    has_ro = False
    if rollout_path:
        orecs = {r['id']: r for r in json.load(open(rollout_path, encoding='utf-8'))}
        drop = [i for i in ok if orecs.get(i, {}).get('status') != 'ok']
        if drop:
            notes.append('%d excluded because %s has no completed record: %s'
                         % (len(drop), os.path.basename(rollout_path), ', '.join(drop[:20])))
        for i in drop:
            ok.pop(i)
        def cons_of(i):
            return CONSENSUS.get(i) if CONSENSUS is not None else orecs[i].get('majority')

        def hit(v, m):
            ans = v.get('answers') or []
            if m is None or not ans:
                return None
            return sum(a is not None and abs(a - m) < 1e-6 for a in ans) / len(ans)
        nocons = [i for i in ok if cons_of(i) is None]
        if nocons:
            notes.append('%d chains have no consensus answer (rollouts never flag them): %s'
                         % (len(nocons), ', '.join(nocons[:20])))
        for i, r in ok.items():
            by_i = {s['i']: s for s in orecs[i]['steps']}
            for row in r['steps']:
                br = by_i.get(row['i'], {}).get('branches', {})
                if 'original' in br:
                    m = cons_of(i)
                    row['rollout'] = {'orig': hit(br['original'], m),
                                      'probes': {t: hit(v, m) for t, v in br.items() if t != 'original'}}
        has_ro = True

    crit = None
    if critic_path:
        crit = {r['id']: r for r in json.load(open(critic_path, encoding='utf-8'))}
        drop = [i for i in ok if crit.get(i, {}).get('status') != 'ok']
        if drop:
            notes.append('%d excluded because the critic run has no completed record' % len(drop))
        for i in drop:
            ok.pop(i)

    recs = list(ok.values())
    for r in recs:
        if r['id'] in LABELS:
            r['label'] = LABELS[r['id']]
    unlabelled = [r['id'] for r in recs if r.get('label') is None]
    if unlabelled:
        raise SystemExit('no first-error label yet for: %s' % ', '.join(unlabelled))
    has_feng = bool(recs) and all(r['steps'] and r['steps'][0].get('conf_feng') is not None for r in recs)
    groups = {}
    for r in recs:
        groups.setdefault(r['subset'], []).append((r['id'], r['label']))
    groups = {s: groups[s] for s in SUBSET_ORDER if s in groups}
    fams = families(has_feng, has_unv, ckeys, has_rd, has_fx, has_ro)
    return recs, groups, fams, crit, notes


def predictions(recs, fams, params_by_family, crit):
    preds = {name: {r['id']: build(*params_by_family[name])(r) for r in recs}
             for name, (build, _, _) in fams.items()}
    if crit is not None:
        preds['critic'] = {r['id']: crit[r['id']]['prediction'] for r in recs}
    return preds


def table(preds, groups, n_boot, label_params=None):
    boot = bootstrap(groups, preds, 'method', n=n_boot)
    cols = ['Rule'] + (['params'] if label_params else []) + ['%s F1 (n, err)' % s for s in groups] \
        + ['Avg F1 [95% CI]', 'Δ method − rule [95% CI], p']
    head = '| ' + ' | '.join(cols) + ' |'
    lines = [head, '|' + ' --- |' * (len(groups) + 3 + (1 if label_params else 0))]
    for m, p in preds.items():
        per, avg = evaluate(p, groups)
        cells = ['%s (%d, %d)' % (fmt(per[s]['f1']), per[s]['n'], per[s]['n_err']) for s in groups]
        b = boot[m]
        ci = '%s [%s, %s]' % (fmt(avg), fmt(b['ci'][0]), fmt(b['ci'][1]))
        pv = ('<%.3f' % (1 / n_boot)) if b['p'] == 0 else fmt(b['p'])
        dv = '—' if m == 'method' or b['diff_vs_base'] is None else '%+.3f [%+.3f, %+.3f], p=%s' % (
            b['diff_vs_base'], b['diff_ci'][0], b['diff_ci'][1], pv)
        pcol = (' %s |' % label_params.get(m, '—')) if label_params else ''
        lines.append('| %s |%s %s | %s | %s |' % (m, pcol, ' | '.join(cells), ci, dv))
    return lines


def detail(preds, groups):
    lines = ['| Rule | acc on wrong chains (exact step) | acc on correct chains | ±1 step | precision | recall | specificity |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for m, p in preds.items():
        s = stats([(l, p[i]) for g in groups.values() for i, l in g])
        lines.append('| %s | %s | %s | %s | %s | %s | %s |' % (
            m, fmt(s['acc_err']), fmt(s['acc_cor']), fmt(s['within1']),
            fmt(s['prec']), fmt(s['rec']), fmt(s['spec'])))
    return lines


def split_dev(groups, frac, seed=42):
    dev = set()
    for sp, g in groups.items():
        rng = random.Random('%d|dev|%s' % (seed, sp))
        for cls in (True, False):
            ids = sorted(i for i, l in g if (l != -1) == cls)
            rng.shuffle(ids)
            dev.update(ids[:round(len(ids) * frac)])
    return dev


def report(run_path, critic_path, contrastive_paths, n_boot, tune_on, rederive_path=None, dev_frac=None,
           feng_path=None, rollout_path=None):
    recs, groups, fams, crit, notes = load(run_path, critic_path, contrastive_paths, rederive_path,
                                           feng_path, rollout_path)
    model = recs[0]['model'] if recs else '?'
    srcs = [os.path.basename(run_path)] + ([os.path.basename(critic_path)] if critic_path else []) \
        + [os.path.basename(c) for c in contrastive_paths] \
        + ([os.path.basename(rederive_path)] if rederive_path else []) \
        + ([os.path.basename(feng_path)] if feng_path else []) \
        + ([os.path.basename(rollout_path)] if rollout_path else [])
    lines = ['## %s' % model, '', 'Sources: %s. %d chains scored, identical for every rule.'
             % (', '.join('`%s`' % s for s in srcs), len(recs)), '']
    lines += ['- %s' % n for n in notes] + ([''] if notes else [])
    if not recs:
        return '\n'.join(lines + ['Nothing to score.'])

    frozen = {name: f[1] for name, f in fams.items()}
    preds = predictions(recs, fams, frozen, crit)
    lines += ['### Frozen thresholds (from AIME, nothing tuned here)', '']
    lines += table(preds, groups, n_boot)
    lines += ['', 'Δ is method minus the rule: positive = the method is better.', '']
    lines += detail(preds, groups) + ['', 'Pooled over subsets. Detection = fired anywhere in the chain.', '']

    if dev_frac:
        dev_ids = split_dev(groups, dev_frac)
        dev = {s: [x for x in g if x[0] in dev_ids] for s, g in groups.items()}
        test = {s: [x for x in g if x[0] not in dev_ids] for s, g in groups.items()}
        dev = {s: g for s, g in dev.items() if g}
        test = {s: g for s, g in test.items() if g}
        is_dev = lambda r: r['id'] in dev_ids
        what = 'a seeded %d%% dev split (%d chains), scored on the other %d chains' % (
            round(100 * dev_frac), len(dev_ids), len(recs) - len(dev_ids))
        lines += ['Dev chains: %s' % ', '.join(sorted(dev_ids)), '']
    elif tune_on:
        dev = {tune_on: groups[tune_on]} if tune_on in groups else {}
        test = {s: g for s, g in groups.items() if s != tune_on}
        is_dev = lambda r: r['subset'] == tune_on
        what = '%s, scored on the other subsets' % tune_on
    if (dev_frac or tune_on):
        if not dev:
            lines += ['Cannot tune: no dev chains.', '']
        else:
            chosen = {}
            for name, (build, fz, grid) in fams.items():
                best = None
                for prm in grid:
                    _, f1 = evaluate({r['id']: build(*prm)(r) for r in recs if is_dev(r)}, dev)
                    f1 = -1 if f1 is None else f1
                    dist = sum(abs(a - b) for a, b in zip(prm, fz))
                    if best is None or (f1, -dist) > (best[0], -best[1]):
                        best = (f1, dist, prm)
                chosen[name] = best[2]
            lines += ['### Thresholds tuned on %s (held out)' % what, '']
            if not test:
                lines += ['No held-out chains to test on.', '']
            else:
                tp = predictions([r for r in recs if not is_dev(r)], fams, chosen, crit)
                labels = {n: ', '.join('%g' % x for x in chosen[n]) for n in chosen}
                dev_recs = [r for r in recs if is_dev(r)]
                if any(r['label'] != -1 for r in dev_recs) and any(r['label'] == -1 for r in dev_recs):
                    ck = next((k for k in ('contrastive_logprob', 'contrastive_verbal')
                               if any(row.get(k) for r in recs for row in r['steps'])), None)
                    lm = Learned(dev_recs, dev, ck)
                    tp['learned'] = {r['id']: lm.predict(r) for r in recs if not is_dev(r)}
                    labels['learned'] = 'theta %.1f' % lm.theta
                    learned_note = 'learned: ' + lm.describe()
                else:
                    learned_note = 'learned: not fitted (the dev split needs both wrong and correct chains)'
                lines += table(tp, test, n_boot, labels)
                lines += ['', learned_note]
                lines += ['', 'params: (tau, Conf max) for method rules, (Conf max) for no-probe rules, '
                          '(share max) for contrastive and rederive; (tau, Conf max, share max) for method_min_or_rederive; '
                          '(tau, Conf max, contrastive share max, rederive support max) for full*. '
                          'The critic has no threshold.', '']

    n_real = [len(real_probes(row)) for r in recs for row in r['steps']]
    lines += ['Probe health: %.2f real probes per step; %d%% of steps have >= %d real probes.'
              % (sum(n_real) / len(n_real), round(100 * sum(x >= MIN_PROBES for x in n_real) / len(n_real)),
                 MIN_PROBES), '']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--critic', nargs='*', default=[], help='one critic file per --runs file, same order')
    ap.add_argument('--contrastive', nargs='*', default=[],
                    help='one entry per --runs file, same order; several files for one run joined by commas')
    ap.add_argument('--rederive', nargs='*', default=[], help='one pb_rederive.py file per --runs file, same order')
    ap.add_argument('--feng', nargs='*', default=[], help='one pb_feng.py file per --runs file, same order')
    ap.add_argument('--rollout', nargs='*', default=[], help='one pb_rollout.py file per --runs file, same order')
    ap.add_argument('--tune-on', default=None, help='subset used to choose thresholds; scored on the others')
    ap.add_argument('--dev-frac', type=float, default=None,
                    help='choose thresholds on a seeded stratified fraction of chains, score on the rest')
    ap.add_argument('--labels', nargs='*', default=[],
                    help='files whose "label" fields are the ground truth (e.g. a pb_solve.py file '
                         'after a person wrote the first-error steps); they override the labels '
                         'copied into the runs, so labels can be fixed after the runs')
    ap.add_argument('--consensus', default=None,
                    help='pb_cfconsensus.py output; rollout rows are scored against its consensus')
    ap.add_argument('--md', default=None)
    ap.add_argument('--boot', type=int, default=1000)
    a = ap.parse_args()
    for name, lst in (('--critic', a.critic), ('--contrastive', a.contrastive), ('--rederive', a.rederive),
                      ('--feng', a.feng), ('--rollout', a.rollout)):
        if lst and len(lst) != len(a.runs):
            raise SystemExit('%s must list one entry per --runs file, in the same order' % name)
    if a.consensus:
        global CONSENSUS
        CONSENSUS = {r['id']: r.get('consensus') for r in json.load(open(a.consensus, encoding='utf-8'))
                     if r.get('status') == 'ok'}
    for lf in a.labels:
        for r in json.load(open(lf, encoding='utf-8')):
            if 'label' in r:
                LABELS[r['id']] = r['label']
    parts = ['# ProcessBench results', '']
    for k, rp in enumerate(a.runs):
        cps = [c for c in a.contrastive[k].split(',') if c] if a.contrastive else []
        parts.append(report(rp, a.critic[k] if a.critic else None, cps, a.boot, a.tune_on,
                            a.rederive[k] if a.rederive else None, a.dev_frac,
                            a.feng[k] if a.feng else None,
                            a.rollout[k] if a.rollout else None))
    text = '\n'.join(parts)
    print(text)
    if a.md:
        open(a.md, 'w', encoding='utf-8').write(text + '\n')


if __name__ == '__main__':
    main()
