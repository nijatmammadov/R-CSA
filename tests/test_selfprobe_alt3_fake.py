import json
import math

import pb_selfprobe as P
import score_pb as S
from pb_confirm import screen

SHORT = {'Ana works 6 x 5 = 30 hours.': 'hours worked = 6 * 5 = 30',
         'She earns 30 x 12 = 360 dollars.': 'pay = 30 * 12 = 360'}


class Fake:
    model = 'fake'

    def __init__(self):
        self.tokens = {'in': 0, 'out': 0}

    def ask(self, system, user, max_tokens=0, temperature=0):
        if user.startswith('Rewrite one step'):
            step = user.split('Step: "')[-1].rstrip().rstrip('"')
            return 'Rewrite: ' + SHORT[step]
        if 'COUNTERFACTUAL' in user:
            if user.rstrip().endswith('hours worked = 6 * 5 = 30'):
                return json.dumps({'logical': 'hours worked = 6 + 5 = 11', 'factual': 'hours worked = 12 * 5 = 60',
                                   'temporal': 'NONE', 'quantitative': 'NONE', 'causal': 'NONE',
                                   'reading': 'hours = 6 = 6'})
            return json.dumps({'logical': 'pay = 30 / 12 = 2.5', 'factual': 'NONE', 'temporal': 'pay = 6 * 12 = 72',
                               'quantitative': 'pay = 30 * 12 = 350', 'causal': 'NONE', 'reading': 'pay = 30 * 12 = 360'})
        if user.startswith('Below is a math word problem'):
            keys = [l.split('"')[1] for l in user.split('Alternatives:')[-1].splitlines() if l.strip().startswith('"')]
            m = {'logical': 'Ana works 6 + 5 = 11 hours.', 'factual': 'Ana works 12 x 5 = 61 hours.',
                 'temporal': 'She earns 6 x 12 = 72 dollars.', 'quantitative': 'She earns 30 x 12 = 350 dollars.'}
            if 'pay = 30 / 12' in user:
                m['logical'] = 'She earns 30 / 12 = 2.5 dollars.'
            return json.dumps({k: m[k] for k in keys})
        raise AssertionError('unexpected prompt: ' + user[:80])

    def ask_logprobs(self, system, user, top=20, max_tokens=1):
        p = 0.2 if ('6 + 5 = 11' in user.split('Step:')[-1]) else 0.6
        return None, [('TRUE', math.log(p)), ('FALSE', math.log(1 - p))]


def test_alt3_runner_with_fake_model():
    r = P.Runner(Fake(), feng=False, chunk=4, inner=2, conf_mode='ptrue', probe_style='alt3',
                 shorten=True, render_full=True)
    o = r.run({'id': 't1', 'subset': 'gsm8k', 'generator': 'x', 'label': -1,
               'problem': 'Ana works 6 hours a day for 5 days and is paid $12 an hour. How much does she earn?',
               'steps': list(SHORT)})
    assert o['status'] == 'ok', o
    s0, s1 = o['steps']
    assert s0['probe_type'] == {'logical': 'logical', 'factual': 'temporal'}
    assert 'reading' in s0['not_single_edit']
    assert list(s0['probes']) == ['logical'] and 'factual' in s0['render_failed']
    assert s1['probe_type'] == {'logical': 'logical', 'temporal': 'temporal', 'quantitative': 'quantitative'}
    assert 'reading' in s1['copies'] and s1['text'] == 'She earns 30 x 12 = 360 dollars.'
    assert S.real_probes(s1) == ['logical', 'temporal', 'quantitative']
    assert screen(s1) == (0.0, 'logical')
