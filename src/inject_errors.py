import re
import json
import random
import argparse

NUM = r'\$?\d[\d,]*(?:\.\d+)?'
UNIT = r'(?:\s+[A-Za-z%][A-Za-z/%]*){0,3}'
EXPR = re.compile(r'(?P<a>%s)(?P<ua>%s)\s*(?P<op>[x×*+\-−/÷])\s*(?P<b>%s)(?P<ub>%s)\s*=\s*(?P<c>%s)'
                  % (NUM, UNIT, NUM, UNIT, NUM))
QNUM = re.compile(r'(?<![\w.])\d[\d,]*(?:\.\d+)?')
TYPES = ['arithmetic', 'operand', 'operation']
SWAP = {'+': '-', '-': '+', '−': '+'}


def plausible(new, old):
    return new > 0 and abs(new - old) > 1e-9 and old / 3 <= new <= old * 3 and abs(new * 100 - round(new * 100)) < 1e-9


def val(s):
    return float(s.replace('$', '').replace(',', ''))


def apply(a, op, b):
    if op in '+':
        return a + b
    if op in '-−':
        return a - b
    if op in 'x×*':
        return a * b
    return a / b if b else None


def fmt(x, like):
    s = ('%d' % round(x)) if abs(x - round(x)) < 1e-9 else ('%.2f' % x).rstrip('0').rstrip('.')
    return ('$' if like.startswith('$') else '') + s


_OPS = 'x\u00d7*+-\u2212/\u00f7('


def checkable(step, m):
    before = step[:m.start()].rstrip()
    after = step[m.end():].lstrip()
    return not (before and before[-1] in _OPS) and not (after and after[0] in _OPS.replace('(', ''))


def right(m):
    r = apply(val(m.group('a')), m.group('op'), val(m.group('b')))
    c = m.group('c')
    dec = len(c.split('.')[1]) if '.' in c else 0
    return r is not None and abs(round(r, dec) - val(c)) < 1e-9


def expressions(step):
    return [m for m in EXPR.finditer(step) if checkable(step, m) and right(m)]


def all_arithmetic_right(steps):
    return all(right(m) for s in steps for m in EXPR.finditer(s) if checkable(s, m))


def splice(step, m, repl):
    parts, last = [], m.start()
    for g in ('a', 'ua', 'op', 'b', 'ub', 'c'):
        st, en = m.span(g)
        parts.append(step[last:st])
        parts.append(repl.get(g, m.group(g)))
        last = en
    return step[:m.start()] + ''.join(parts) + step[last:m.end()] + step[m.end():]


def inject(step, m, kind, qnums, rng):
    a, op, b, c = val(m.group('a')), m.group('op'), val(m.group('b')), val(m.group('c'))
    if kind == 'arithmetic':
        d = max(1.0, round(abs(c) * rng.choice([0.1, 0.2, 0.25, 0.5])))
        new = c + d * rng.choice([-1, 1])
        if new <= 0 or new == c:
            new = c + d
        return splice(step, m, {'c': fmt(new, m.group('c'))})
    if kind == 'operand':
        cands = sorted({x for x in qnums if x > 0 and x not in (a, b)})
        rng.shuffle(cands)
        side = rng.choice(['a', 'b'])
        for x in cands:
            na, nb = (x, b) if side == 'a' else (a, x)
            r = apply(na, op, nb)
            if r is not None and plausible(r, c):
                return splice(step, m, {side: fmt(x, m.group(side)), 'c': fmt(r, m.group('c'))})
        return None
    if kind == 'operation':
        nop = SWAP.get(op)
        r = apply(a, nop, b) if nop else None
        if r is None or not plausible(r, c):
            return None
        return splice(step, m, {'op': nop, 'c': fmt(r, m.group('c'))})
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solved', required=True)
    ap.add_argument('--questions', required=True)
    ap.add_argument('--n-injected', type=int, default=100)
    ap.add_argument('--n-clean', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    qs = {q['id']: q for q in json.load(open(a.questions, encoding='utf-8'))}
    chains = [r for r in json.load(open(a.solved, encoding='utf-8')) if r.get('status') == 'ok']
    usable = [r for r in chains if r['answer_correct'] and len(r['steps']) >= 2 and all_arithmetic_right(r['steps'])
              and any(expressions(s) for s in r['steps'][:-1])]
    print('%d chains, %d right answers, %d usable (all arithmetic right, an expression before the last step)'
          % (len(chains), sum(r['answer_correct'] for r in chains), len(usable)))
    if len(usable) < a.n_injected + a.n_clean:
        raise SystemExit('need %d usable chains, have %d: solve more questions' % (a.n_injected + a.n_clean, len(usable)))
    rng = random.Random(a.seed)
    usable.sort(key=lambda r: r['id'])
    rng.shuffle(usable)

    out, kinds, spare = [], {}, []
    pool = list(usable)
    for n in range(a.n_injected):
        want = TYPES[n % len(TYPES)]
        for idx, r in enumerate(pool):
            qnums = [val(x) for x in QNUM.findall(qs[r['id']]['question'])]
            sites = [(k, m) for k, s in enumerate(r['steps'][:-1]) for m in expressions(s)]
            rng.shuffle(sites)
            hit = None
            for k, m in sites:
                new = inject(r['steps'][k], m, want, qnums, rng)
                if new and new != r['steps'][k]:
                    hit = (k, new)
                    break
            if hit:
                break
        else:
            raise SystemExit('no usable chain left that takes a %r error (%d injected so far)' % (want, n))
        pool.pop(idx)
        k, new = hit
        rec = dict(r)
        rec['steps'] = r['steps'][:k] + [new] + r['steps'][k + 1:]
        rec['raw'] = '\n'.join('Step %d: %s' % (j + 1, s) for j, s in enumerate(rec['steps']))
        rec.update(label=k, label_note='injected error (by construction)',
                   injected={'type': want, 'step': k, 'before': r['steps'][k], 'after': new})
        kinds[want] = kinds.get(want, 0) + 1
        out.append(rec)
    if len(pool) < a.n_clean:
        raise SystemExit('only %d usable chains left for the %d clean ones: solve more questions' % (len(pool), a.n_clean))
    for r in pool[:a.n_clean]:
        rec = dict(r)
        rec.update(label=-1, label_note='clean: right answer and all arithmetic right', injected=None)
        out.append(rec)
    out.sort(key=lambda r: r['id'])
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False)
    pos = [r['injected']['step'] / max(1, len(r['steps']) - 1) for r in out if r['injected']]
    print('wrote %d chains -> %s | injected %d %s, clean %d | injected step at %.0f%% of the chain on average'
          % (len(out), a.out, len(pos), kinds, sum(r['label'] == -1 for r in out), 100 * sum(pos) / len(pos)))


if __name__ == '__main__':
    main()
