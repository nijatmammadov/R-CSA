import os
import sys
import math
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_table as F
from css_eval import load, lg, s_css_full, s_ptrue_full, tune, preds_for, cv_preds

R, D = 'results/test/', 'results/dev/'
MATH = [('Llama-3.1-8B / GSM8K', 'llama31-8b', 'data/dev/dev_llama31-8b_solved.json', D + 'dev_llama31-8b',
         R + 'test_llama31-8b_solved.json', R + 'test_llama31-8b'),
        ('Mistral-7B / GSM8K', 'mistral7b', 'data/dev/dev_mistral7b_solved.json', D + 'dev_mistral7b',
         R + 'test_mistral7b_solved.json', R + 'test_mistral7b'),
        ('Qwen2.5-7B / MATH-L5', 'qwen25-7b_math5', 'data/dev/dev_qwen25-7b_math5_le40_solved.json',
         D + 'dev_qwen25-7b_math5', R + 'test_qwen25-7b_math5_le40_solved.json', R + 'test_qwen25-7b_math5')]
SQA = [('Llama-3.1-8B / StrategyQA', 'llama31-8b'), ('Mistral-7B / StrategyQA', 'mistral7b'),
       ('Qwen2.5-7B / StrategyQA', 'qwen25-7b')]
SYN = ('Llama-3.1-8B / Track A synthetic', 'data/tracka/tracka_llama31-8b_inj.json', 'results/tracka/tracka_llama31-8b')
RULE = 'recall:0.8'
RCSA = '**R-CSA (ours)**'
PTRUE = 'P(TRUE) alone (baseline)'
SAT = 0.999


def s_rs(row, _p=None):
    if not row or row.get('p') is None:
        return None
    alts = [a['p'] for a in row.get('alts') or [] if a.get('p') is not None]
    return sum(abs(row['p'] - x) for x in alts) / len(alts) if alts else None


def s_conf(row, _p=None):
    return None if not row or row.get('p') is None else lg(row['p'])


METHODS = [('R-CSA', s_css_full), ('P(TRUE)', s_ptrue_full), ('RS (plan)', s_rs), ('Conf(step) alone', s_conf)]


def opt(path):
    return path if os.path.exists(path) else None


def load_with(labels, prefix, css_path, selfcheck=True):
    chains, _, _ = load(labels, css_path, opt(prefix + '_feng.json'), opt(prefix + '_critic.json'),
                        opt(prefix + '_selfcheck.json') if selfcheck else None,
                        ptrue_full_path=prefix + '_ptrue_full.json', cfq_path=opt(prefix + '_cfq.json'))
    return chains


def load_math(labels, prefix, css_suffix, selfcheck=True):
    return load_with(labels, prefix, prefix + css_suffix, selfcheck)


def load_sqa(tag):
    p = D + 'dev_%s_sqa' % tag
    return load_with('data/dev/dev_%s_sqa2_solved.json' % tag, p, p + '_css.json')


def scope(chains, fn, param=None):
    out = []
    for c in chains:
        for i, v in F.step_scores(c, 'css', fn, param):
            if v is None or (c['label'] != -1 and i > c['label']):
                continue
            out.append((v, int(i == c['label'])))
    return out


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(xs, ys):
    a, b = ranks(xs), ranks(ys)
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    return cov / math.sqrt(va * vb) if va and vb else float('nan')


def chain_rss_auroc(chains):
    def rss(c):
        v = [x for _, x in F.step_scores(c, 'css', s_rs, None) if x is not None]
        return sum(v) / len(v) if v else float('inf')
    return F.auroc([-rss(c) for c in chains if c['label'] != -1], [-rss(c) for c in chains if c['label'] == -1])


def chain_spearman(chains, fn, param):
    xs, ys = [], []
    for c in chains:
        v = [x for _, x in F.step_scores(c, 'css', fn, param) if x is not None]
        if v:
            xs.append(min(v))
            ys.append(int(c['label'] == -1))
    return spearman(xs, ys)


def f(x, d=2):
    return '—' if x is None or (isinstance(x, float) and math.isnan(x)) else ('%.*f' % (d, x))


def thresholded(name, chains, dev, frozen, cv):
    out = {}
    for m, fn in METHODS:
        param = [1.0, 1.0, 0.0, 2.0] if m == 'R-CSA' else None
        if cv:
            out[m] = (cv_preds(chains, 'css', fn, [param], RULE, 5), '5-fold CV')
        elif m in ('R-CSA', 'P(TRUE)'):
            t = frozen[RCSA if m == 'R-CSA' else PTRUE]['t']
            out[m] = (preds_for(chains, 'css', fn, t, param), 'frozen t=%.2f' % t)
        else:
            t, _ = tune(dev, 'css', fn, [param], RULE)
            out[m] = (preds_for(chains, 'css', fn, t, param), 'dev t=%.3f' % t)
    return out


def section_a(rows):
    L = ['## (a) The plan\'s relative sensitivity RS beside R-CSA', '',
         'RS_k = mean |Conf(step) - Conf(probe)| over the step\'s probes; low RS = flagged. RSS = mean RS over '
         'the chain. Conf(step) alone is the lettered judge\'s confidence in the step without the probes\' '
         'contrast. AUROCs use no threshold; F1 and PB F1 use the setting shown.', '',
         '| Setting | Method | Step AUROC | Chain AUROC (min) | Chain AUROC (RSS) | t | TP/FN/FP/TN | Exact | F1 | PB F1 |',
         '|' + '---|' * 10]
    for name, chains, th in rows:
        for m, fn in METHODS:
            param = [1.0, 1.0, 0.0, 2.0] if m == 'R-CSA' else None
            sa = F.step_auroc(chains, 'css', fn, param)[0]
            ca = F.chain_auroc(chains, 'css', fn, param)
            preds, setting = th[m]
            s = F.stats(chains, preds)
            L.append('| %s | %s | %s | %s | %s | %s | %d/%d/%d/%d | %d | %s | %s |'
                     % (name, m, f(sa), f(ca), f(chain_rss_auroc(chains)) if m == 'RS (plan)' else '', setting,
                        s['flagged_wrong'], s['n_wrong'] - s['flagged_wrong'], s['n_cor'] - s['clean'], s['clean'],
                        s['exact'], f(s['f1']), f(s['pbf1'])))
    return L


def section_b(rows):
    L = ['## (b) Per probe type', '',
         '"Beats": share of steps with at least one probe of that type whose Conf is above the step\'s '
         '(first-error steps vs correct steps up to the first error). AUROC: step sensitivity '
         'logit Conf(step) - mean logit Conf(probe) from that type\'s probes only (low = error). '
         'Typed probes and re-derivation probes are listed apart; a re-derivation probe\'s type is the '
         'type of its difference from the step.', '',
         '| Setting | Probe | Steps with it (err/cor) | Beats: error steps | Beats: correct steps | AUROC |',
         '|' + '---|' * 6]
    for name, chains, _ in rows:
        groups = sorted({(a.get('source', 'edit'), a['type']) for c in chains for r in c['css'].values()
                         for a in r.get('alts') or [] if a.get('p') is not None})
        for src, typ in groups:
            pos, neg, bp, bn = [], [], 0, 0
            for c in chains:
                for i, r in c['css'].items():
                    if r.get('p') is None or (c['label'] != -1 and i > c['label']):
                        continue
                    ps = [a['p'] for a in r.get('alts') or []
                          if a.get('p') is not None and a.get('source', 'edit') == src and a['type'] == typ]
                    if not ps:
                        continue
                    sens = lg(r['p']) - sum(lg(x) for x in ps) / len(ps)
                    beats = any(x > r['p'] for x in ps)
                    if i == c['label']:
                        pos.append(-sens)
                        bp += beats
                    else:
                        neg.append(-sens)
                        bn += beats
            if len(pos) < 3:
                continue
            label = ('typed: ' if src == 'edit' else 're-derivation: ') + typ
            L.append('| %s | %s | %d/%d | %.0f%% | %.0f%% | %s |'
                     % (name, label, len(pos), len(neg), 100 * bp / len(pos), 100 * bn / len(neg) if neg else 0,
                        f(F.auroc(pos, neg))))
    return L


def section_c(rows):
    L = ['## (c) Spearman rank correlation', '',
         'Step: score vs correctness (1 = correct step, 0 = first-error step), steps up to the first error. '
         'Chain: lowest step score vs chain correct (1) or wrong (0). Positive = higher score on correct.', '',
         '| Setting | ' + ' | '.join('%s step | %s chain' % (m, m) for m, _ in METHODS) + ' |',
         '|' + '---|' * (1 + 2 * len(METHODS))]
    for name, chains, _ in rows:
        cells = []
        for m, fn in METHODS:
            param = [1.0, 1.0, 0.0, 2.0] if m == 'R-CSA' else None
            sc = scope(chains, fn, param)
            cells += [f(spearman([v for v, _ in sc], [1 - e for _, e in sc])), f(chain_spearman(chains, fn, param))]
        L.append('| %s | %s |' % (name, ' | '.join(cells)))
    return L


def section_d(rows):
    L = ['## (d) Failure modes of R-CSA at its threshold', '',
         'Wrong chains: missed (no step flagged), early (flagged before the labelled error), exact, late. '
         'Correct chains: false alarms. "Saturated" = P(TRUE) of the labelled error step >= %s, where the '
         'confidence term cannot separate it from correct steps.' % SAT, '',
         '| Setting | Wrong | Missed | Early | Exact | Late | Correct | False alarms | Missed with saturated P(TRUE) | Error steps saturated | Correct steps saturated |',
         '|' + '---|' * 11]
    for name, chains, th in rows:
        preds = th['R-CSA'][0]
        wrong = [c for c in chains if c['label'] != -1]
        cor = [c for c in chains if c['label'] == -1]
        miss = [c for c in wrong if preds[c['id']] == -1]
        early = sum(0 <= preds[c['id']] < c['label'] for c in wrong)
        exact = sum(preds[c['id']] == c['label'] for c in wrong)
        late = sum(preds[c['id']] > c['label'] for c in wrong)

        def sat(c, i):
            p = c['css'].get(i, {}).get('p_full')
            return p is not None and p >= SAT
        err_sat = sum(sat(c, c['label']) for c in wrong)
        cor_steps = [(c, i) for c in chains for i in range(c['n_steps']) if c['label'] == -1 or i < c['label']]
        cor_sat = sum(sat(c, i) for c, i in cor_steps)
        L.append('| %s | %d | %d | %d | %d | %d | %d | %d | %d/%d | %.0f%% | %.0f%% |'
                 % (name, len(wrong), len(miss), early, exact, late, len(cor),
                    sum(preds[c['id']] != -1 for c in cor), sum(sat(c, c['label']) for c in miss), len(miss),
                    100 * err_sat / len(wrong), 100 * cor_sat / max(1, len(cor_steps))))
    return L


def section_e(chains, th):
    solved = {r['id']: r for r in json.load(open(SYN[1], encoding='utf-8'))}
    L = ['## (e) Track A: injected errors, per type', '',
         'Llama-3.1-8B judges its own correct chains, one of which in two has one error injected at a known '
         'step (src/inject_errors.py); thresholds frozen on the GSM8K dev set, nothing tuned here.', '',
         '| Method | Error type | Chains | Detected | Exact | ±1 | Early | Late |', '|' + '---|' * 8]
    for m in ('R-CSA', 'P(TRUE)', 'RS (plan)', 'Conf(step) alone'):
        preds = th[m][0]
        for typ in ('arithmetic', 'operand', 'operation', None):
            cs = [c for c in chains if c['label'] != -1 and (typ is None or solved[c['id']]['injected']['type'] == typ)]
            if not cs:
                continue
            det = sum(preds[c['id']] != -1 for c in cs)
            L.append('| %s | %s | %d | %d (%.0f%%) | %d | %d | %d | %d |'
                     % (m, typ or 'all', len(cs), det, 100 * det / len(cs), sum(preds[c['id']] == c['label'] for c in cs),
                        sum(preds[c['id']] != -1 and abs(preds[c['id']] - c['label']) <= 1 for c in cs),
                        sum(0 <= preds[c['id']] < c['label'] for c in cs), sum(preds[c['id']] > c['label'] for c in cs)))
        cl = [c for c in chains if c['label'] == -1]
        L.append('| %s | clean chains: false alarms | %d | %d (%.0f%%) | | | | |'
                 % (m, len(cl), sum(preds[c['id']] != -1 for c in cl),
                    100 * sum(preds[c['id']] != -1 for c in cl) / max(1, len(cl))))
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--md', default=None)
    a = ap.parse_args()
    rows = []
    for name, tag, dev_labels, dev_prefix, test_labels, test_prefix in MATH:
        dev = load_math(dev_labels, dev_prefix, '_css2.json', selfcheck=False)
        test = load_math(test_labels, test_prefix, '_css.json')
        frozen = json.load(open(D + 'css_frozen_%s.json' % tag, encoding='utf-8'))
        rows.append((name + ' (test)', test, thresholded(name, test, dev, frozen, cv=False)))
    for name, tag in SQA:
        chains = load_sqa(tag)
        rows.append((name + ' (dev, CV)', chains, thresholded(name, chains, None, None, cv=True)))
    L = ['# Analyses from the research plan', '', 'Generated by `src/plan_analyses.py` from saved runs; no API calls.', '']
    for sec in (section_a, section_b, section_c, section_d):
        L += sec(rows) + ['']
    if os.path.exists(SYN[1]) and os.path.exists(SYN[2] + '_css.json'):
        syn = load_with(SYN[1], SYN[2], SYN[2] + '_css.json')
        dev = load_math(MATH[0][2], MATH[0][3], '_css2.json', selfcheck=False)
        th = thresholded(SYN[0], syn, dev, json.load(open(D + 'css_frozen_llama31-8b.json', encoding='utf-8')), cv=False)
        sa = section_a([(SYN[0], syn, th)])
        sa[0] = '## (a) on Track A: RS beside R-CSA'
        L += sa + [''] + section_e(syn, th) + ['']
    txt = '\n'.join(L)
    print(txt)
    if a.md:
        os.makedirs(os.path.dirname(a.md) or '.', exist_ok=True)
        open(a.md, 'w', encoding='utf-8').write(txt)


if __name__ == '__main__':
    main()
