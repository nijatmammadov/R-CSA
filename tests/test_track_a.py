import random

import inject_errors as ie
from plan_analyses import spearman, s_rs
from score_statements import stats


def first(step):
    return ie.expressions(step)[0]


def test_checkable_and_right():
    assert ie.expressions('She pays 12 x 3 = 36 dollars.')
    assert not ie.expressions('Total 88 + 22 + 70 = 180.')
    assert not ie.all_arithmetic_right(['He has 5 + 4 = 10 apples.'])
    assert ie.expressions('Each gets 110 / 6 = 18.33 kg.')


def test_operation_swaps_and_recomputes():
    s = 'She has 40 + 15 = 55 stickers.'
    new = ie.inject(s, first(s), 'operation', [40, 15], random.Random(0))
    assert new == 'She has 40 - 15 = 25 stickers.'


def test_operation_rejects_implausible():
    s = 'Then 50 - 45 = 5 left.'
    assert ie.inject(s, first(s), 'operation', [50, 45], random.Random(0)) is None


def test_arithmetic_changes_only_result():
    s = 'He earns $20 x 4 = $80 in total.'
    new = ie.inject(s, first(s), 'arithmetic', [20, 4], random.Random(1))
    m = ie.EXPR.search(new)
    assert m.group('a') == '$20' and m.group('b') == '4' and m.group('c') != '$80' and not ie.right(m)


def test_operand_uses_question_number_and_stays_consistent():
    s = 'She buys 6 x 5 = 30 pens.'
    new = ie.inject(s, first(s), 'operand', [6, 5, 7], random.Random(0))
    m = ie.EXPR.search(new)
    assert ie.right(m) and '7' in (m.group('a'), m.group('b'))


def test_spearman():
    assert abs(spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1) < 1e-12
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-12
    assert abs(spearman([1, 1, 2, 2], [0, 0, 1, 1]) - 1) < 1e-12


def test_rs():
    assert abs(s_rs({'p': 0.9, 'alts': [{'p': 0.1}, {'p': 0.5}]}) - 0.6) < 1e-12
    assert s_rs({'p': 0.9, 'alts': []}) is None


def test_statement_stats():
    items = [{'id': 1, 'hallucinated': 1}, {'id': 2, 'hallucinated': 0}]
    s = stats(items, {1: True, 2: False})
    assert s['f1'] == 1.0 and s['acc'] == 1.0
