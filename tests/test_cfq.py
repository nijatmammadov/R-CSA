import re

import css_cfq as Q

CHAIN = ['hours worked = 6 * 5 = 30', 'pay = 30 * 12 = 360', 'money kept = 360 - 15 = 345']


def test_propagate_through_the_chain():
    assert Q.propagate(CHAIN, 6.0, 9.0) == [(30.0, 45.0), (360.0, 540.0), (345.0, 525.0)]


def test_chained_equality_uses_first_expression():
    step = 'total = 2 * (30 + 60) = 2 * 90 = 180'
    assert Q.step_expr(step) == ('2 * (30 + 60)', 180.0)
    assert Q.propagate([step], 30.0, 45.0) == [(180.0, 210.0)]


def test_arithmetic_slip_step_is_none():
    assert Q.propagate(['hours worked = 6 * 5 = 31'], 6.0, 9.0) == [None]


def test_question_numbers_skip_repeated_numbers_and_clock_times():
    q = 'At 9:30 Tom buys 3 apples and 3 pears for 12 dollars and pays with a 20 dollar bill.'
    assert [n[1] for n in Q.question_numbers(q)] == [12.0, 20.0]


def test_bump_and_counterfactual_question():
    assert Q.bump(6.0) == 9.0 and Q.bump(8.0) == 12.0 and Q.bump(2.5) == 3.75
    q = 'Ana works 6 hours a day.'
    assert Q.counterfactual_question(q, Q.question_numbers(q)[0]) == ('Ana works 9 hours a day.', 9.0)


QUESTION = 'Ana works 6 hours a day for 5 days and is paid $12 an hour. How much does she earn?'


class FakeSolver:
    model = 'fake'
    tokens = {'in': 0, 'out': 0}

    def ask(self, system, user, max_tokens=0, temperature=0):
        h, d, r = (float(x) for x in re.findall(r'\d+(?:\.\d+)?', user)[:3])
        return ('Step 1: hours worked = %g * %g = %g\nStep 2: pay = %g * %g = %g\nAnswer: %g'
                % (h, d, h * d, h * d, r, h * d * r, h * d * r))


def _run(short):
    rec = {'id': 't', 'label': None, 'problem': QUESTION}
    return Q.run(FakeSolver(), rec, short, max_int=3, dry=False)


def test_misread_step_fails():
    out = _run(['hours worked = 6 + 5 = 11', 'pay = 11 * 12 = 132'])
    cfq = [s['cfq'] for s in out['steps']]
    assert cfq[0] == 0.0 and cfq[1] == 0.0


def test_correct_step_passes():
    out = _run(['hours worked = 6 * 5 = 30', 'pay = 30 * 12 = 360'])
    assert [s['cfq'] for s in out['steps']] == [1.0, 1.0]
    assert len(out['interventions']) == 3


def test_dry_run_counts_coverage_without_calls():
    class NoCalls:
        def ask(self, *a, **k):
            raise AssertionError('dry run must not call the model')
    out = Q.run(NoCalls(), {'id': 't', 'label': None, 'problem': QUESTION}, CHAIN[:2], max_int=3, dry=True)
    assert [s['informative'] for s in out['steps']] == [2, 3] and all(s['cfq'] is None for s in out['steps'])
