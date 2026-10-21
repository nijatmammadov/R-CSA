import os
import re
import ast
import sys
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb_common as pc
from pb_solve import SOLVE, parse

_QNUM = re.compile(r'(?<![\d.,:])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![\d:]|,\d{3}\b)')


def fval(s):
    return float(s.replace(',', ''))


def close(a, b):
    return abs(a - b) <= max(1e-6, 0.006 * max(1.0, abs(b)))


def question_numbers(question):
    hits = [(m.group(1), fval(m.group(1)), m.start(1)) for m in _QNUM.finditer(question)]
    count = {}
    for _, v, _ in hits:
        count[v] = count.get(v, 0) + 1
    return [h for h in hits if count[h[1]] == 1]


def bump(v):
    if float(v).is_integer():
        return float(int(v) + max(1, round(abs(v) / 2)))
    return round(v * 1.5, 4)


def fmt(v):
    return str(int(v)) if float(v).is_integer() else ('%g' % v)


def counterfactual_question(question, num):
    text, v, start = num
    v2 = bump(v)
    return question[:start] + fmt(v2) + question[start + len(text):], v2


def _eval_expr(expr):
    e = expr.replace('$', '').replace('×', '*').replace('÷', '/').replace('^', '**')
    e = re.sub(r'(?<=\d),(?=\d{3}\b)', '', e)
    e = re.sub(r'(?<=[\d)])\s*[xX]\s*(?=[\d(])', '*', e)
    if re.search(r'[A-Za-z_:%]', e):
        return None
    try:
        return pc._arith(ast.parse(e, mode='eval'))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, RecursionError):
        return None


def step_expr(txt):
    segs = [x.strip() for x in (txt or '').split('=')]
    if len(segs) < 2:
        return None
    v0 = pc.step_result(txt)
    if v0 is None:
        return None
    exprs = segs[1:-1]
    for e in exprs:
        r = _eval_expr(e)
        if r is not None and close(r, v0):
            return e, v0
    return (exprs[0] if exprs else ''), v0


def propagate(short_steps, q, q2):
    out, known = [], []
    for txt in short_steps:
        se = step_expr(txt) if txt else None
        if not se:
            out.append(None)
            continue
        expr, v0 = se
        if not expr:
            new = q2 if close(v0, q) else next((n for o, n in reversed(known) if close(v0, o)), v0)
            out.append((v0, new))
            known.append((v0, new))
            continue

        def sub(m):
            x = fval(m.group(0))
            for o, n in reversed(known):
                if close(x, o):
                    return repr(n)
            if close(x, q):
                return repr(q2)
            if close(x, q / 100):
                return repr(q2 / 100)
            return m.group(0)
        e = re.sub(r'\d[\d,]*(?:\.\d+)?', sub, expr.replace('$', ''))
        orig = _eval_expr(expr)
        new = _eval_expr(e)
        if orig is None or new is None or not close(orig, v0):
            out.append(None)
            continue
        out.append((v0, new))
        known.append((v0, new))
    return out


def solution_values(raw):
    steps, ans = parse(raw)
    vals = set()
    for s in steps + ([ans] if ans else []):
        r = pc.step_result(s)
        if r is not None:
            vals.add(r)
        for m in re.finditer(r'=\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)', s or ''):
            vals.add(fval(m.group(1)))
    return vals


def plan(rec, short):
    used = set()
    for txt in short:
        se = step_expr(txt) if txt else None
        if se:
            for n in re.findall(r'\d[\d,]*(?:\.\d+)?', (se[0] or str(se[1])).replace('$', '')):
                used.add(fval(n))
    keep = []
    for num in question_numbers(rec['problem']):
        if any(close(num[1], u) or close(num[1] / 100, u) for u in used):
            keep.append(num)
    return keep


def run(asker, rec, short, max_int, dry):
    out = {'id': rec['id'], 'label': rec.get('label'), 'interventions': []}
    nums = plan(rec, short)[:max_int]
    steps = [{'i': k, 'informative': 0, 'passes': 0} for k in range(len(short))]
    for num in nums:
        qcf, q2 = counterfactual_question(rec['problem'], num)
        pred = propagate(short, num[1], q2)
        iv = {'q': num[1], 'q2': q2, 'question': qcf,
              'predicted': [None if p is None else p[1] for p in pred]}
        informative = [k for k, p in enumerate(pred) if p is not None and not close(p[1], p[0])]
        iv['informative'] = informative
        if not dry and informative:
            raw = asker.ask(SOLVE, qcf, max_tokens=1024, temperature=0.0)
            vals = solution_values(raw)
            iv.update(raw=raw, values=sorted(vals))
            for k in informative:
                ok = any(close(pred[k][1], v) for v in vals)
                steps[k]['informative'] += 1
                steps[k]['passes'] += int(ok)
        elif dry:
            for k in informative:
                steps[k]['informative'] += 1
        out['interventions'].append(iv)
    for s in steps:
        s['cfq'] = (s['passes'] / s['informative']) if (s['informative'] and not dry) else None
    out.update(steps=steps, status='ok', config={'max_int': max_int, 'bump': 'q + max(1, round(q/2))'})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--api', default='openrouter', choices=sorted(pc.APIS))
    ap.add_argument('--generator', required=True)
    ap.add_argument('--model', default=None)
    ap.add_argument('--from-json', required=True, help='pb_solve.py output')
    ap.add_argument('--css', required=True, help='css.py output on the same chains (for the shortened steps)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-int', type=int, default=3, help='interventions per question')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--or-provider', default=None)
    ap.add_argument('--quant', default=None)
    ap.add_argument('--price', default=None)
    ap.add_argument('--budget', type=float, default=None)
    ap.add_argument('--dry-run', action='store_true', help='coverage only: no API calls, nothing saved')
    a = ap.parse_args()
    pin, pout = ([float(x) for x in a.price.split(',')] if a.price else (None, None))
    data = {r['id']: r for r in pc.load_processbench(['gsm8k'], a.generator, a.from_json)}
    css = {r['id']: r for r in json.load(open(a.css, encoding='utf-8')) if r.get('status') == 'ok'}
    ids = [i for i in data if i in css]
    short = {i: [s['text'] if s.get('shortened') else None for s in css[i]['steps']] for i in ids}

    if a.dry_run:
        res = [run(None, data[i], short[i], a.max_int, True) for i in ids]
        st = [s for r in res for s in r['steps']]
        err = [r['steps'][data[r['id']]['label']] for r in res if data[r['id']]['label'] not in (-1, None)]
        print('chains %d | interventions %d (%.1f per chain) | steps reached by >=1 intervention %d/%d (%.0f%%) | '
              'first-error steps reached %d/%d' % (
                  len(res), sum(len(r['interventions']) for r in res), sum(len(r['interventions']) for r in res) / len(res),
                  sum(s['informative'] > 0 for s in st), len(st), 100 * sum(s['informative'] > 0 for s in st) / len(st),
                  sum(s['informative'] > 0 for s in err), len(err)))
        return

    asker = pc.Asker(pc.make_client(a.api), pc.model_for(a.api, a.generator, a.model),
                     pc.provider_body(a.api, a.or_provider, a.quant))
    done = {r['id']: r for r in json.load(open(a.out, encoding='utf-8'))} if os.path.exists(a.out) else {}
    todo = [i for i in ids if done.get(i, {}).get('status') != 'ok']
    print('CSS counterfactual question test | %s | %d chains, %d to run' % (asker.model, len(ids), len(todo)), flush=True)
    t0, stopped = time.time(), None

    def one(i):
        try:
            return run(asker, data[i], short[i], a.max_int, False)
        except pc.APIGaveUp as e:
            return {'id': i, 'status': 'error', 'error': str(e)[:300]}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(one, i) for i in todo]
        for f in as_completed(futs):
            r = f.result()
            done[r['id']] = r
            pc.save_json(a.out, list(done.values()))
            print('  %-14s %-5s | %.0fs | %s' % (r['id'], r['status'], time.time() - t0,
                                                pc.spend(asker.tokens, pin, pout)), flush=True)
            if a.budget is not None and pin is not None and not stopped and \
                    asker.tokens['in'] / 1e6 * pin + asker.tokens['out'] / 1e6 * pout >= a.budget:
                stopped = 'budget reached; re-run to continue'
                for g in futs:
                    g.cancel()
    pc.save_json(a.out, list(done.values()))
    ok = sum(r.get('status') == 'ok' for r in done.values())
    print('saved %d ok, %d error | api errors %s | %s' % (ok, len(done) - ok, asker.errors or '{}',
                                                         pc.spend(asker.tokens, pin, pout)), flush=True)
    if stopped:
        print('STOPPED: %s' % stopped)
        sys.exit(2)


if __name__ == '__main__':
    main()
