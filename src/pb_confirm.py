import os
import sys
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb_common as pc
import score_pb as S
from pb_conf_variants import Runner as Variants, Failed


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def screen(row):
    real = [t for t in S.real_probes(row) if row['probe_conf'].get(t) is not None]
    if row.get('conf') is None or not real:
        return None, None
    win = max(real, key=lambda t: row['probe_conf'][t])
    return row['conf'] - row['probe_conf'][win], win


class Runner(Variants):
    def run(self, rec):
        out = {'id': rec['id'], 'model': rec['model'],
               'config': {'wordings': [w[1] for w in self.words], 'pair_sha': self.pair_sha,
                          'prior_label': self.prior_label}}
        steps = [s['text'] for s in rec['steps']]
        rows, jobs = [], []
        for k, row in enumerate(rec['steps']):
            d, win = screen(row)
            r = {'i': k, 'screen': d, 'winner': win}
            rows.append(r)
            if d is None or d >= 0:
                continue
            alt = row['probes'][win]
            r.update(conf2={'step': [], 'alt': []}, pair={})
            for w, _sha in self.words:
                jobs.append((k, 'conf2', 'step', lambda w=w, k=k: self.ptrue(w, rec['problem'], steps[:k], steps[k])))
                jobs.append((k, 'conf2', 'alt', lambda w=w, k=k, alt=alt: self.ptrue(w, rec['problem'], steps[:k], alt)))
            jobs.append((k, 'pair', 'ab', lambda k=k, alt=alt: self.prefer(rec['problem'], steps[:k], steps[k], alt)))
            jobs.append((k, 'pair', 'ba', lambda k=k, alt=alt: 1 - self.prefer(rec['problem'], steps[:k], alt, steps[k])))

        def go(j):
            try:
                return j, j[3](), None
            except (Failed, pc.APIGaveUp) as e:
                return j, None, str(e)[:200]

        errs = []
        if jobs:
            with ThreadPoolExecutor(max_workers=self.inner) as ex:
                for (k, kind, slot, _), v, err in ex.map(go, jobs):
                    if err:
                        errs.append(err)
                    elif kind == 'conf2':
                        rows[k]['conf2'][slot].append(v)
                    else:
                        rows[k]['pair'][slot] = v
        for r in rows:
            if r.get('screen') is None or r['screen'] >= 0:
                r['delta2'] = r['pair_mean'] = None
                continue
            c2 = r.get('conf2') or {}
            ms, ma = mean(c2.get('step') or []), mean(c2.get('alt') or [])
            r['delta2'] = (ms - ma) if (ms is not None and ma is not None) else None
            r['pair_mean'] = mean(list((r.get('pair') or {}).values()))
        out['steps'] = rows
        out['n_screened'] = sum(1 for r in rows if r.get('screen') is not None and r['screen'] < 0)
        out['dropped'] = len(errs)
        out['status'] = 'error' if any('gave up' in e.lower() for e in errs) else 'ok'
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='pb_selfprobe.py --probe-style alt --conf ptrue output')
    ap.add_argument('--api', required=True, choices=sorted(pc.APIS))
    ap.add_argument('--out', default=None)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--inner', type=int, default=8)
    ap.add_argument('--or-provider', default=None)
    ap.add_argument('--quant', default=None)
    ap.add_argument('--price', default=None)
    ap.add_argument('--budget', type=float, default=None)
    a = ap.parse_args()

    src = [r for r in json.load(open(a.input, encoding='utf-8')) if r.get('status') == 'ok']
    cfg = src[0].get('config', {}) if src else {}
    if not src or cfg.get('probe_style') not in ('alt', 'alt2', 'alt3') or cfg.get('conf_mode') != 'ptrue':
        raise SystemExit('--input must be a pb_selfprobe.py --probe-style alt --conf ptrue run')
    model = src[0]['model']
    out_path = a.out or a.input.replace('.json', '') + '_confirm.json'
    if os.path.abspath(out_path) == os.path.abspath(a.input):
        raise SystemExit('refusing to write over the input file')
    pin, pout = ([float(x) for x in a.price.split(',')] if a.price else (None, None))
    if a.budget is not None and pin is None:
        raise SystemExit('--budget needs --price')

    asker = pc.Asker(pc.make_client(a.api), model, pc.provider_body(a.api, a.or_provider, a.quant))
    try:
        asker.ask_logprobs(None, 'Is 2 + 2 = 4? Answer TRUE or FALSE.', top=5, max_tokens=1)
    except pc.NoLogprobs as e:
        raise SystemExit('needs token logprobs: %s (pin a provider with --or-provider)' % e)
    runner = Runner(asker, a.inner, cfg.get('prior_label', 'accepted'))
    done = {r['id']: r for r in json.load(open(out_path, encoding='utf-8'))} if os.path.exists(out_path) else {}
    done = {i: r for i, r in done.items() if r.get('config') == {
        'wordings': [w[1] for w in runner.words], 'pair_sha': runner.pair_sha,
        'prior_label': cfg.get('prior_label', 'accepted')}}
    todo = [r for r in src if done.get(r['id'], {}).get('status') != 'ok']
    print('Confirmation of screened steps | %s | %d chains, %d to run' % (model, len(src), len(todo)), flush=True)

    stopped = None
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(runner.run, r) for r in todo]
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result()
            done[r['id']] = r
            pc.save_json(out_path, list(done.values()))
            print('  %-12s %-5s screened %d dropped %d | %d/%d | %s'
                  % (r['id'], r['status'], r['n_screened'], r['dropped'], n, len(todo),
                     pc.spend(asker.tokens, pin, pout)), flush=True)
            if a.budget is not None and asker.tokens['in'] / 1e6 * pin + asker.tokens['out'] / 1e6 * pout >= a.budget:
                stopped = 'budget reached'
                for g in futs:
                    g.cancel()
                break
    pc.save_json(out_path, list(done.values()))
    print('saved %d | screened steps %d | first-token failures %d | %s'
          % (len(done), sum(r.get('n_screened', 0) for r in done.values()), runner.fail,
             pc.spend(asker.tokens, pin, pout)), flush=True)
    if stopped:
        print('STOPPED: %s. Re-run to continue.' % stopped)
        sys.exit(2)
    print('DONE -> %s' % out_path)


if __name__ == '__main__':
    main()
