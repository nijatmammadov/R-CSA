import json
import os

import pytest

import css_eval as E
import final_table as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = os.path.join(ROOT, 'results', 'dev')
LABELS = os.path.join(ROOT, 'data', 'dev', 'dev_llama31-8b_solved.json')
FROZEN = os.path.join(DEV, 'css_frozen_llama31-8b.json')


@pytest.fixture(scope='module')
def dev():
    chains, missing, _ = E.load(LABELS, os.path.join(DEV, 'dev_llama31-8b_css2.json'),
                                os.path.join(DEV, 'dev_llama31-8b_feng.json'),
                                os.path.join(DEV, 'dev_llama31-8b_critic.json'),
                                None,
                                os.path.join(DEV, 'dev_llama31-8b_ptrue_full.json'),
                                os.path.join(DEV, 'dev_llama31-8b_cfq.json'))
    frozen = json.load(open(FROZEN, encoding='utf-8'))
    return chains, missing, frozen


def confusion(s):
    return (s['flagged_wrong'], s['n_wrong'] - s['flagged_wrong'], s['n_cor'] - s['clean'], s['clean'])


def stats_for(chains, frozen, name, fn):
    t, p = frozen[name]['t'], frozen[name]['param']
    p = tuple(p) if isinstance(p, list) else p
    return F.stats(chains, E.preds_for(chains, 'css', fn, t, p))


def test_dev_set_shape(dev):
    chains, missing, _ = dev
    assert len(chains) == 199 and not missing
    assert sum(c['label'] != -1 for c in chains) == 19


def test_frozen_setting(dev):
    _, _, frozen = dev
    r = frozen['**R-CSA (ours)**']
    assert r['param'] == [1.0, 1.0, 0.0, 2.0] and abs(r['t'] - (-2.6346)) < 1e-4


def test_rcsa_confusion_counts(dev):
    chains, _, frozen = dev
    s = stats_for(chains, frozen, '**R-CSA (ours)**', E.s_css_full)
    assert confusion(s) == (16, 3, 22, 158)
    assert s['exact'] == 7 and round(s['pbf1'], 2) == 0.52


def test_ptrue_alone_baseline(dev):
    chains, _, frozen = dev
    s = stats_for(chains, frozen, 'P(TRUE) alone (baseline)', E.s_ptrue_full)
    assert confusion(s) == (16, 3, 71, 109)
