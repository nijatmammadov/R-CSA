import pytest

import pb_common as pc


@pytest.mark.parametrize('step, expected', [
    ('hours worked = 6 * 5 = 30', True),
    ('hours worked = 6 x 5 = 31', False),
    ('share = 7 / 3 = 2.33', True),
    ('share = 7 / 3 = 2.4', False),
    ('total cost = $1,200 * 2 = $2,400', True),
    ('total = 80 + 60 + 110 + 140 = 290', False),
    ('total = 80 + 60 + 110 + 140 = 390', True),
])
def test_checkable(step, expected):
    assert pc.arithmetic_ok(step) is expected


@pytest.mark.parametrize('step', [
    'end time = 9:30 + 2 = 11:30',
    'discount = 20% * 50 = 10',
    'length = 3 m * 4 = 12 m',
    'hours worked = 30',
    'no equals sign here',
])
def test_uncheckable_is_none(step):
    assert pc.arithmetic_ok(step) is None
