import pytest

import pb_selfcheck


@pytest.mark.parametrize('reply, score', [
    ('Yes', 0.0),
    ('No', 1.0),
    ('Not sure', 0.5),
    ('**Yes**', 0.0),
    ('yes, it is supported', 0.0),
    ('', 0.5),
    (None, 0.5),
])
def test_verdict(reply, score):
    assert pb_selfcheck.verdict(reply) == score
