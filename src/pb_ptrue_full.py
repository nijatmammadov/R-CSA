import os
import sys
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb_common as pc
from pb_selfprobe import Runner, StepFailed


def run(runner, rec, inner):
    out = {k: rec[k] for k in ('id', 'subset', 'generator', 'label')}
    out.update(model=runner.a.model, config={'ptrue_sha': runner.ptrue_sha, 'prior_label': runner.prior_label,
                                             'text': 'full'})
    steps = [' '.join(s.split()) for s in rec['steps']]

    def one(k):
        try:
            return k, runner.ptrue(rec['problem'], steps[:k], steps[k]), None
        except (StepFailed, pc.APIGaveUp) as e:
            return k, None, str(e)[:200]
    with ThreadPoolExecutor(max_workers=inner) as ex:
        res = list(ex.map(one, range(len(steps))))
    errs = [e for _, _, e in res if e]
    out['steps'] = [{'i': k, 'p_full': p} for k, p, _ in res]
    if errs:
        out.update(status='error', error='; '.join(errs)[:1000])
    else:
        out['status'] = 'ok'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--api', required=True, choices=sorted(pc.APIS))
    ap.add_argument('--generator', required=True)
    ap.add_argument('--model', default=None)
    ap.add_argument('--from-json', required=True, help='pb_solve.py output')
    ap.add_argument('--subsets', default='gsm8k')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--inner', type=int, default=6)
    ap.add_argument('--or-provider', default=None)
    ap.add_argument('--quant', default=None)
    ap.add_argument('--max-attempts', type=int, default=3)
    ap.add_argument('--price', default=None, help='USD per million tokens IN,OUT')
    a = ap.parse_args()
    pin, pout = ([float(x) for x in a.price.split(',')] if a.price else (None, None))
    model = pc.model_for(a.api, a.generator, a.model)
    data = pc.load_processbench([s.strip() for s in a.subsets.split(',')], a.generator, a.from_json)
    if not data:
        raise SystemExit('no chains for %s in %s' % (a.generator, a.from_json))
    asker = pc.Asker(pc.make_client(a.api), model, pc.provider_body(a.api, a.or_provider, a.quant))
    try:
        asker.ask_logprobs(None, 'Is 2 + 2 = 4? Answer TRUE or FALSE.', top=5, max_tokens=1)
    except pc.NoLogprobs as e:
        raise SystemExit('needs token logprobs: %s' % e)
    runner = Runner(asker, feng=False, chunk=4, inner=a.inner, prior_label='accepted', conf_mode='ptrue')
    cfg = {'ptrue_sha': runner.ptrue_sha, 'prior_label': 'accepted', 'text': 'full'}
    done = {r['id']: r for r in json.load(open(a.out, encoding='utf-8'))} if os.path.exists(a.out) else {}
    done = {i: r for i, r in done.items() if r.get('config') == cfg}
    todo = [r for r in data if done.get(r['id'], {}).get('status') != 'ok'
            and done.get(r['id'], {}).get('attempts', 0) < a.max_attempts]
    print('P(TRUE) on full steps | %s | %d chains, %d to run' % (model, len(data), len(todo)), flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run, runner, r, a.inner) for r in todo]
        for f in as_completed(futs):
            r = f.result()
            r['attempts'] = done.get(r['id'], {}).get('attempts', 0) + 1
            done[r['id']] = r
            pc.save_json(a.out, list(done.values()))
            print('  %-14s %-5s | %.0fs | %s' % (r['id'], r['status'], time.time() - t0,
                                                pc.spend(asker.tokens, pin, pout)), flush=True)
    pc.save_json(a.out, list(done.values()))
    ok = sum(r.get('status') == 'ok' for r in done.values())
    print('saved %d ok, %d error | api errors %s | %s' % (ok, len(done) - ok, asker.errors or '{}',
                                                         pc.spend(asker.tokens, pin, pout)), flush=True)
    if ok < len(done):
        print('Re-run the same command to retry the chains in error.')


if __name__ == '__main__':
    main()
