import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb_common as pc
from pb_solve import SOLVE
from pb_selfcheck import PROMPT as SC_PROMPT
from pb_selfprobe import ALT3_TYPES, PRIOR_LABELS

T = {k: len(pc.load_prompt(f)[0]) for k, f in {
    'judge': 'css_judge.md', 'regen': 'css_regenerate.md', 'short': 'step_shorten.md',
    'probe': 'counterfactuals_alt3.md', 'ptrue': 'step_ptrue.md', 'critic': 'pb_critique_template.txt',
    'f_ext': 'feng_extract.md', 'f_probe': 'feng_probe.md', 'f_conf': 'feng_confidence.md'}.items()}
F_PROBE_FILL = 250
ONE = 4


def load(path):
    with open(path, encoding='utf-8') as f:
        return {r['id']: r for r in json.load(f) if r.get('status') == 'ok'}


def prior_len(steps, k, label=''):
    return len(label) + sum(len(s) + 6 for s in steps[:k])


def rcsa(css, cfq, n_regen):
    g = s = i = o = 0
    q = len(css['problem'])
    short = [r['text'] for r in css['steps']]
    full = [r['text_full'] for r in css['steps']]
    for k, r in enumerate(css['steps']):
        pj = T['judge'] + q + prior_len(short, k)
        g += 2
        i += T['short'] + len(full[k]) + T['probe'] + q + prior_len(short, k) + len(short[k])
        o += len(short[k]) + len(ALT3_TYPES) * (len(short[k]) + 15)
        s += 3
        i += pj + len(short[k]) + T['ptrue'] + q + prior_len(full, k, PRIOR_LABELS['accepted']) + len(full[k])
        o += 2 * ONE
        if r['shortened'] and pc.probe_parts(r['text']):
            g += n_regen
            qty = r['text'].split('=')[0]
            i += n_regen * (T['regen'] + q + prior_len(short, k) + 2 * len(qty))
            o += n_regen * len(r['text'])
        s += len(r['alts'])
        i += sum(pj + len(a['text']) for a in r['alts'])
        o += ONE * len(r['alts'])
    for iv in (cfq or {}).get('interventions', []):
        if 'raw' in iv:
            g += 1
            i += len(SOLVE) + len(iv['question'])
            o += len(iv['raw'] or '')
    return g, s, i, o


def selfcheck(r, solved):
    n, steps = r['n_samples_used'], solved['steps']
    sample = len(solved.get('raw') or '\n'.join(steps))
    i = n * (len(SOLVE) + len(solved['problem'])) + n * sum(len(SC_PROMPT) + sample + len(st) for st in steps)
    o = n * sample + n * len(steps) * ONE
    return n, n * len(steps), i, o


def critic(r, solved):
    return 1, 0, T['critic'] + len(solved['problem']) + sum(len(x) + 20 for x in solved['steps']), len(r.get('critique') or '')


def feng(r, full):
    g = s = i = o = 0
    for st in r['steps']:
        sts = st['statements']
        texts = [x['text'] for x in sts] + list(st.get('dropped') or [])
        g += 1
        i += T['f_ext'] + len(st['text'])
        o += sum(len(t) + 6 for t in texts) + 20
        s += len(texts)
        i += sum(T['f_conf'] + len(t) for t in texts)
        o += 3 * ONE * len(texts)
        if full:
            probes = [p for x in sts for p in x['probes'].values()]
            g += 4 * len(texts)
            i += sum(4 * (T['f_probe'] + F_PROBE_FILL + len(t)) for t in texts)
            o += sum(len(p) for p in probes)
            s += 4 * len(texts)
            i += sum(T['f_conf'] + len(p) for p in probes)
            o += 3 * ONE * len(probes)
    return g, s, i, o


def solved_path(pre):
    for p in (pre + '_solved.json', pre + '_le40_solved.json'):
        if os.path.exists(p):
            return p
    raise SystemExit('no solved file for %s' % pre)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('prefixes', nargs='+', help='e.g. results/test/test_llama31-8b (reads <prefix>_css.json, ...)')
    a = ap.parse_args()
    print('| Run | Chains | Steps/chain | Method | Generation calls | Scoring calls | Total calls '
          '| ≈ input tokens | ≈ output tokens |')
    print('|---|---|---|---|---|---|---|---|---|')
    for pre in a.prefixes:
        css_path = pre + '_css.json' if os.path.exists(pre + '_css.json') else pre + '_css2.json'
        files = {'css': css_path, 'cfq': pre + '_cfq.json', 'sc': pre + '_selfcheck.json',
                 'critic': pre + '_critic.json', 'feng': pre + '_feng.json', 'solved': solved_path(pre)}
        missing = [p for p in files.values() if not os.path.exists(p)]
        if missing:
            raise SystemExit('missing: %s' % ', '.join(missing))
        d = {k: load(p) for k, p in files.items() if k != 'solved'}
        with open(files['solved'], encoding='utf-8') as f:
            d['solved'] = {r['id']: r for r in json.load(f)}
        ids = sorted(set.intersection(*(set(v) for v in d.values())))
        if not ids:
            raise SystemExit('%s: no chain is ok in every file' % pre)
        n_regen = d['css'][ids[0]]['config']['n_regen']
        rows = {
            'R-CSA (ours)': [rcsa(d['css'][x], d['cfq'][x], n_regen) for x in ids],
            'SelfCheckGPT': [selfcheck(d['sc'][x], d['solved'][x]) for x in ids],
            'ProcessBench critic': [critic(d['critic'][x], d['solved'][x]) for x in ids],
            'Feng (2025)': [feng(d['feng'][x], True) for x in ids],
            'Verbal Confidence': [feng(d['feng'][x], False) for x in ids],
        }
        steps = sum(len(d['css'][x]['steps']) for x in ids) / len(ids)
        for name, v in rows.items():
            g, s, i, o = (sum(t[j] for t in v) / len(v) for j in range(4))
            print('| %s | %d | %.1f | %s | %.0f | %.0f | %.0f | %s | %s |' % (
                os.path.basename(pre), len(ids), steps, name, g, s, g + s,
                '{:,.0f}'.format(i / 4), '{:,.0f}'.format(o / 4)))


if __name__ == '__main__':
    main()
