import pb_common as pc

QUESTION = ('Ana works 6 hours a day for 5 days and is paid $12 an hour. '
            'She spends $15 on lunch. How much money does she keep?')
TIME = pc.time_numbers(QUESTION)
PRIOR = [30.0, 360.0]


def test_time_numbers():
    assert TIME == {'6', '5'}


def test_operator_change_is_logical():
    assert pc.alternative_type('hours worked = 6 * 5 = 30', 'hours worked = 6 + 5 = 11', [], TIME) == 'logical'


def test_operand_order_change_is_logical():
    assert pc.alternative_type('share = 7 / 3 = 2.33', 'share = 3 / 7 = 0.43', [], TIME) == 'logical'


def test_same_expression_new_value_is_quantitative():
    assert pc.alternative_type('hours worked = 6 * 5 = 30', 'hours worked = 6 * 5 = 35', [], TIME) == 'quantitative'


def test_swapped_operand_from_earlier_step_is_causal():
    assert pc.alternative_type('money kept = 360 - 15 = 345', 'money kept = 360 - 30 = 330', PRIOR, TIME) == 'causal'


def test_time_number_operand_is_temporal():
    assert pc.alternative_type('pay = 30 * 12 = 360', 'pay = 30 * 5 = 150', PRIOR, TIME) == 'temporal'


def test_other_operand_is_factual():
    assert pc.alternative_type('pay = 30 * 12 = 360', 'pay = 30 * 10 = 300', PRIOR, TIME) == 'factual'


def test_added_omitted_operand_is_factual():
    assert pc.alternative_type('money kept = 360 - 15 = 345', 'money kept = 360 - 15 - 20 = 325',
                               PRIOR, TIME) == 'factual'


def test_quantity_name_change_is_none():
    assert pc.alternative_type('money kept = 360 - 15 = 345', 'money spent = 360 - 20 = 340', PRIOR, TIME) is None


def test_two_operands_changed_is_none():
    assert pc.alternative_type('pay = 30 * 12 = 360', 'pay = 18 * 10 = 180', PRIOR, TIME) is None
