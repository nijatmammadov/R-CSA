import math

import pb_common as pc
import css as C
import pb_selfprobe as P


def _tops(*pairs):
    return [(pc.clean_token(t), math.log(p)) for t, p in pairs]


def test_clean_token_maps_space_markers():
    assert pc.clean_token('▁A') == ' A'
    assert pc.clean_token('ĠB') == ' B'
    assert pc.clean_token('Ċ') == '\n'
    assert pc.clean_token(' A') == ' A'
    assert pc.clean_token(None) == ''


def test_letter_prob_reads_sentencepiece_tokens():
    p = C.letter_prob(_tops(('▁A', 0.6), ('▁B', 0.3), ('▁The', 0.1)))
    assert abs(p - 0.6 / 0.9) < 1e-9


def test_letter_prob_counts_both_spellings():
    p = C.letter_prob(_tops(('A', 0.5), ('▁A', 0.2), ('▁B', 0.3)))
    assert abs(p - 0.7) < 1e-9


def test_ptrue_reads_byte_level_tokens():
    p = P.ptrue_of(_tops(('ĠTRUE', 0.8), ('ĠFALSE', 0.2)))
    assert abs(p - 0.8) < 1e-9
