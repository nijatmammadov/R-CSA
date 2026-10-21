import os
import re
import sys
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb_common as pc
from pb_solve import SOLVE

PROMPT = ('Context: {context}\n\nSentence: {sentence}\n\n'
          'Is the sentence supported by the context above? Answer Yes or No.\n\nAnswer: ')
SAMPLE_T = 1.0


def verdict(text):
    t = (text or '').strip().lower().lstrip('*"\' ')
    if re.match(r'yes\b', t):
        return 0.0
    if re.match(r'no\b', t):
        return 1.0
    return 0.5


def run(asker, rec, n, inner):
    out = {k: rec[k] for k in ('id', 'subset', 'generator', 'label')}
    out.update(model=asker.model, config={'n_samples': n, 'sample_temperature': SAMPLE_T, 'prompt': PROMPT})
    steps = [' '.join(s.split()) for s in rec['steps']]
    try:
        with ThreadPoolExecutor(max_workers=inner) as ex:
            samples = list(ex.map(lambda _: asker.ask(SOLVE, rec['problem'], max_tokens=1024,
                                                      temperature=SAMPLE_T), range(n)))
        samples = [s for s in samples if s and s.strip()]
        if not samples:
            raise pc.APIGaveUp('no non-empty samples')
        jobs = [(k, j) for k in range(len(steps)) for j in range(len(samples))]
        with ThreadPoolExecutor(max_workers=inner) as ex:
            answers = list(ex.map(lambda kj: asker.ask(None, PROMPT.format(context=samples[kj[1]], sentence=steps[kj[0]]),
                                                       max_tokens=5, temperature=0.0), jobs))
    except pc.APIGaveUp as e:
        out.update(status='error', error=str(e)[:500])
        return out
    rows = [{'i': k, 'verdicts': []} for k in range(len(steps))]
    for (k, _), a in zip(jobs, answers):
        rows[k]['verdicts'].append(verdict(a))
    for r in rows:
        r['inconsistency'] = sum(r['verdicts']) / len(r['verdicts'])
    out.update(steps=rows, n_samples_used=len(samples),
               unreadable=sum(v == 0.5 for r in rows for v in r['verdicts']), status='ok')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--api', required=True, choices=sorted(pc.APIS))
    ap.add_argument('--generator', required=True)
    ap.add_argument('--model', default=None)
    ap.add_argument('--from-json', required=True, help='pb_solve.py output')
    ap.add_argument('--subsets', default='gsm8k')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--n', type=int, default=20, help='stochastic samples per chain (paper: 20)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--inner', type=int, default=8)
    ap.add_argument('--or-provider', default=None)
    ap.add_argument('--quant', default=None)
    ap.add_argument('--max-attempts', type=int, default=3)
    ap.add_argument('--price', default=None, help='USD per million tokens IN,OUT')
    ap.add_argument('--budget', type=float, default=None, help='USD; stop after this much (needs --price)')
    a = ap.parse_args()
    pin, pout = ([float(x) for x in a.price.split(',')] if a.price else (None, None))
    if a.budget is not None and pin is None:
        raise SystemExit('--budget needs --price')
    model = pc.model_for(a.api, a.generator, a.model)
    data = pc.load_processbench([s.strip() for s in a.subsets.split(',')], a.generator, a.from_json, a.limit)
    if not data:
        raise SystemExit('no chains for %s in %s' % (a.generator, a.from_json))
    asker = pc.Asker(pc.make_client(a.api), model, pc.provider_body(a.api, a.or_provider, a.quant))
    done = {r['id']: r for r in json.load(open(a.out, encoding='utf-8'))} if os.path.exists(a.out) else {}
    done = {i: r for i, r in done.items() if r.get('config', {}).get('n_samples') == a.n
            and r.get('config', {}).get('prompt') == PROMPT}
    todo = [r for r in data if done.get(r['id'], {}).get('status') != 'ok'
            and done.get(r['id'], {}).get('attempts', 0) < a.max_attempts]
    print('SelfCheckGPT (prompt, step level) | %s | %d chains, %d to run | N=%d'
          % (model, len(data), len(todo), a.n), flush=True)
    t0, stopped = time.time(), None
    for st in range(0, len(todo), a.workers * 2):
        batch = todo[st:st + a.workers * 2]
        try:
            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                futs = [ex.submit(run, asker, r, a.n, a.inner) for r in batch]
                for f in as_completed(futs):
                    r = f.result()
                    r['attempts'] = done.get(r['id'], {}).get('attempts', 0) + 1
                    done[r['id']] = r
                    pc.save_json(a.out, list(done.values()))
                    print('  %-14s %-5s | %.0fs | %s' % (r['id'], r['status'], time.time() - t0,
                                                        pc.spend(asker.tokens, pin, pout)), flush=True)
                    if a.budget is not None and asker.tokens['in'] / 1e6 * pin + asker.tokens['out'] / 1e6 * pout >= a.budget \
                            and not stopped:
                        stopped = 'budget reached; re-run to continue'
                        for g in futs:
                            g.cancel()
        except pc.Fatal as e:
            stopped = str(e)
        if stopped:
            break
        recent = [done[r['id']] for r in batch if r['id'] in done]
        if len(recent) >= 4 and sum(r['status'] == 'error' for r in recent) > len(recent) / 2:
            stopped = 'more than half of the last batch failed'
            break
    pc.save_json(a.out, list(done.values()))
    ok = [r for r in done.values() if r.get('status') == 'ok']
    print('saved %d ok, %d error | unreadable verdicts %d | api errors %s | %s'
          % (len(ok), sum(r.get('status') == 'error' for r in done.values()),
             sum(r.get('unreadable', 0) for r in ok), asker.errors or '{}', pc.spend(asker.tokens, pin, pout)), flush=True)
    if stopped:
        print('STOPPED: %s' % stopped, flush=True)
        sys.exit(2)
    print('DONE -> %s' % a.out, flush=True)


if __name__ == '__main__':
    main()
