import json
import math
import subprocess
import sys

import css_fact as CF
import css_eval as E
from conftest import ROOT, SRC

STEPS = ['Genghis Khan had about 16 children.',
         'Julius Caesar had three children.',
         'Therefore, 16 is greater than 3, so more people descend from Genghis Khan.']
CLAIMS = {STEPS[0]: 'Genghis Khan had about 16 children.',
          STEPS[1]: 'Claim: Julius Caesar had three children.',
          STEPS[2]: ''}
PROBES = {
    'Genghis Khan had about 16 children.': {
        'factual': 'Kublai Khan had about 16 children.', 'temporal': 'NONE',
        'quantitative': 'Genghis Khan had about 4 children.',
        'logical': 'Genghis Khan had about 16 children.',
        'causal': 'NONE'},
    'Julius Caesar had three children.': {
        'factual': 'Augustus had three children.', 'temporal': 'NONE',
        'quantitative': 'Julius Caesar had ten children.',
        'logical': 'Julius Caesar had no children.',
        'causal': 'Caesar had.'},
}


class Fake:
    model = 'fake'

    def __init__(self):
        self.tokens = {'in': 0, 'out': 0}
        self.errors = {}

    def ask(self, system, user, max_tokens=0, temperature=0):
        if user.startswith('Rewrite one step of a reasoning chain'):
            return CLAIMS[user.split('Step: "')[-1].rstrip().rstrip('"')]
        if 'COUNTERFACTUAL' in user:
            cur = user.rstrip().split('Current step: ')[-1]
            return json.dumps(PROBES.get(cur, {t: 'NONE' for t in CF.FACT_TYPES}))
        raise AssertionError('unexpected prompt: ' + user[:60])

    def ask_logprobs(self, system, user, top=20, max_tokens=1):
        step = user.split('Step: ')[-1].split('\n')[0]
        p = {'N/A': 0.5, 'Julius Caesar had three children.': 0.4, 'Augustus had three children.': 0.7}.get(step, 0.6)
        return None, [(' A', math.log(p)), (' B', math.log(1 - p))]


def run_chain():
    css = CF.FactCSS(Fake(), Fake(), inner=3)
    return css.run({'id': 'sqa_1', 'subset': 'gsm8k', 'generator': 'x', 'label': 1,
                    'problem': 'Are more people today related to Genghis Khan than Julius Caesar?',
                    'steps': STEPS})


def test_fact_css_run_with_fake_model():
    o = run_chain()
    assert o['status'] == 'ok', o
    s0, s1, s2 = o['steps']
    assert s1['text'] == 'Julius Caesar had three children.'
    assert s2['shortened'] is False and s2['text'] == STEPS[2]
    assert s2['alts'] == []
    assert [a['type'] for a in s0['alts']] == ['factual', 'quantitative']
    assert s0['dropped'] == {'same claim': 1}
    assert [a['type'] for a in s1['alts']] == ['factual', 'quantitative', 'logical']
    assert s1['dropped'] == {'truncated': 1}
    assert all(r['regen'] == [] and 'regen_agree' not in r for r in o['steps'])
    assert abs(s1['p'] - 0.4) < 1e-9 and abs(s1['p_cf'] - 0.5) < 1e-9
    d = E.delta(s1, 'logodds')
    want = E.lg(0.4) - (E.lg(0.7) + E.lg(0.6) + E.lg(0.6)) / 3
    assert abs(d - want) < 1e-9


def test_css_eval_scores_fact_output(tmp_path):
    o = run_chain()
    o2 = json.loads(json.dumps(o))
    o2['id'], o2['label'] = 'sqa_2', -1
    labels = [{'id': x['id'], 'status': 'ok', 'label': x['label'], 'steps': STEPS} for x in (o, o2)]
    ptrue = [{'id': x['id'], 'status': 'ok', 'steps': [{'i': k, 'p_full': p} for k, p in enumerate((0.9, 0.2, 0.8))]}
             for x in (o, o2)]
    for name, obj in (('lab.json', labels), ('css.json', [o, o2]), ('pf.json', ptrue)):
        (tmp_path / name).write_text(json.dumps(obj), encoding='utf-8')
    r = subprocess.run([sys.executable, '%s/css_eval.py' % SRC, '--labels', str(tmp_path / 'lab.json'),
                        '--css', str(tmp_path / 'css.json'), '--ptrue-full', str(tmp_path / 'pf.json'),
                        '--weights-from', '%s/results/dev/css_frozen_llama31-8b.json' % ROOT,
                        '--freeze', str(tmp_path / 'frozen.json')], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert 'Not scored, no value on these chains: Regeneration agreement (baseline).' in r.stdout
    frozen = json.loads((tmp_path / 'frozen.json').read_text())
    assert frozen['**R-CSA (ours)**']['param'] == [1.0, 1.0, 0.0, 2.0]


def test_cv_preds_predicts_every_chain_out_of_fold():
    chains = []
    for j in range(10):
        wrong = j < 4
        ps = (0.9, 0.1 if wrong else 0.8, 0.85)
        chains.append({'id': 'c%d' % j, 'label': 1 if wrong else -1, 'n_steps': 3,
                       'css': {k: {'i': k, 'p_full': p} for k, p in enumerate(ps)}})
    preds = E.cv_preds(chains, 'css', E.s_ptrue_full, [None], 'pbf1', k=2)
    assert set(preds) == {c['id'] for c in chains}
    assert all(preds[c['id']] == (1 if c['label'] == 1 else -1) for c in chains)
