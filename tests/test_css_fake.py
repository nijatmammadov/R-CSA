import itertools
import json
import math

import css as C

SHORT = {'Ana works 6 x 5 = 30 hours.': 'hours worked = 6 * 5 = 30',
         'She earns 30 x 12 = 365 dollars.': 'pay = 30 * 12 = 365',
         'So she has a lot of money.': None}
TYPES = ['logical', 'factual', 'temporal', 'quantitative', 'causal', 'reading']


class Fake:
    model = 'fake'

    def __init__(self):
        self.tokens = {'in': 0, 'out': 0}
        self.errors = {}
        self.regen = itertools.cycle(['pay = 30 * 12 = 360', 'pay = 30 * 12 = 360',
                                      'pay = 6 * 12 = 72', 'Sure! pay = 30 * 12 = 365'])

    def ask(self, system, user, max_tokens=0, temperature=0):
        if user.startswith('Rewrite one step'):
            step = user.split('Step: "')[-1].rstrip().rstrip('"')
            return ('Rewrite: ' + SHORT[step]) if SHORT[step] else 'I cannot.'
        if 'COUNTERFACTUAL' in user:
            cur = user.rstrip().split('Current step: ')[-1]
            if cur.startswith('hours worked'):
                return json.dumps({'logical': 'hours worked = 6 + 5 = 11', 'factual': 'hours worked = 12 * 5 = 61',
                                   'temporal': 'NONE', 'quantitative': 'NONE', 'causal': 'NONE', 'reading': 'NONE'})
            if cur.startswith('pay'):
                return json.dumps({'logical': 'pay = 30 / 12 = 2.5', 'factual': 'NONE', 'temporal': 'pay = 6 * 12 = 72',
                                   'quantitative': 'pay = 30 * 12 = 360', 'causal': 'NONE', 'reading': 'NONE'})
            return json.dumps({t: 'NONE' for t in TYPES})
        if user.startswith('Here is a math word problem'):
            if 'computes: hours worked' in user.lower():
                return 'hours worked = 6 * 5 = 30'
            return next(self.regen)
        raise AssertionError('unexpected prompt: ' + user[:60])

    def ask_logprobs(self, system, user, top=20, max_tokens=1):
        step = user.split('Step: ')[-1].split('\n')[0]
        p = {'N/A': 0.5, 'pay = 30 * 12 = 365': 0.3, 'pay = 30 * 12 = 360': 0.8}.get(step, 0.6)
        return None, [(' A', math.log(p)), (' B', math.log(1 - p))]


def test_css_run_with_fake_model():
    css = C.CSS(Fake(), Fake(), inner=3, n_regen=4)
    o = css.run({'id': 'c1', 'subset': 'gsm8k', 'generator': 'x', 'label': 1,
                 'problem': 'Ana works 6 hours a day for 5 days and is paid $12 an hour. How much does she earn?',
                 'steps': list(SHORT)})
    assert o['status'] == 'ok', o
    s0, s1, s2 = o['steps']
    assert s0['arith_ok'] is True and s1['arith_ok'] is False
    assert s2['shortened'] is False and s2['alts'] == []
    assert s0['dropped'].get('own arithmetic wrong') == 1
    assert [a['type'] for a in s0['alts']] == ['logical']
    assert s1['regen_agree'] == 1
    assert {a['text'] for a in s1['alts']} == {'pay = 30 * 12 = 360', 'pay = 6 * 12 = 72', 'pay = 30 / 12 = 2.5'}
    assert s1['dropped'].get('duplicate value') == 3
    assert abs(s1['p'] - 0.3) < 1e-9 and abs(s1['p_cf'] - 0.5) < 1e-9
