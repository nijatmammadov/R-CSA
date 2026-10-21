import os
import sys
import json
import math
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb_common as pc
import score_pb as S
from pb_selfprobe import PRIOR_LABELS, ptrue_of

WORDINGS = ['step_ptrue_v2.md', 'step_ptrue_v3.md']


class Failed(Exception):
    pass


class Runner:
    def __init__(self, asker, inner, prior_label):
        self.a, self.inner, self.prior_label = asker, inner, prior_label
        self.words = [pc.load_prompt(w) for w in WORDINGS]
        self.pair, self.pair_sha = pc.load_prompt('step_pair.md')
        self.fail = 0
        self.lock = threading.Lock()

    def _prior(self, prior):
        head = PRIOR_LABELS[self.prior_label]
        return (head + '\n' + ''.join('  %d. %s\n' % (j, s) for j, s in enumerate(prior, 1))).rstrip() \
            if prior else head + ' (none)'

    def _first_token(self, user, read):
        for _ in range(2):
            _, tops = self.a.ask_logprobs(None, user, top=20, max_tokens=1)
            v = read(tops)
            if v is not None:
                return v
            with self.lock:
                self.fail += 1
        raise Failed('no usable first token')

    def ptrue(self, tmpl, problem, prior, step):
        user = tmpl.replace('<<QUESTION>>', problem).replace('<<PRIOR>>', self._prior(prior)).replace('<<STEP>>', step)
        return self._first_token(user, ptrue_of)

    def prefer(self, problem, prior, a, b):
        user = (self.pair.replace('<<QUESTION>>', problem).replace('<<PRIOR>>', self._prior(prior))
                .replace('<<A>>', a).replace('<<B>>', b))

        def read(tops):
            pa = pb = 0.0
            for tok, lp in tops:
                t = tok.strip().upper().strip('.):')
                if t == 'A':
                    pa += math.exp(lp)
                elif t == 'B':
                    pb += math.exp(lp)
            return pa / (pa + pb) if pa + pb > 0 else None
        return self._first_token(user, read)

    def run(self, rec):
        out = {'id': rec['id'], 'model': rec['model'], 'config': {
            'wordings': [w[1] for w in self.words], 'pair_sha': self.pair_sha, 'prior_label': self.prior_label}}
        steps = [s['text'] for s in rec['steps']]
        jobs = []
        rows = []
        for k, row in enumerate(rec['steps']):
            real = S.real_probes(row)
            rows.append({'i': k, 'multi': {'original': [row['conf']]}, 'pair': {}})
            texts = [('original', row['text'])] + [(t, row['probes'][t]) for t in real]
            for name, txt in texts:
                if name != 'original':
                    rows[k]['multi'][name] = [row['probe_conf'].get(t) for t in [name]]
                for w, _sha in self.words:
                    jobs.append((k, 'multi', name, None, lambda w=w, txt=txt, k=k: self.ptrue(w, rec['problem'], steps[:k], txt)))
            for t in real:
                alt = row['probes'][t]
                jobs.append((k, 'pair', t, 'ab', lambda alt=alt, k=k: self.prefer(rec['problem'], steps[:k], steps[k], alt)))
                jobs.append((k, 'pair', t, 'ba', lambda alt=alt, k=k: 1 - self.prefer(rec['problem'], steps[:k], alt, steps[k])))

        def go(j):
            try:
                return j, j[4](), None
            except (Failed, pc.APIGaveUp) as e:
                return j, None, str(e)[:200]
        errs = []
        with ThreadPoolExecutor(max_workers=self.inner) as ex:
            for (k, kind, name, order, _), v, err in ex.map(go, jobs):
                if err:
                    errs.append(err)
                    continue
                if kind == 'multi':
                    rows[k]['multi'][name].append(v)
                else:
                    rows[k]['pair'].setdefault(name, {})[order] = v
        out['steps'] = rows
        out['status'] = 'error' if any('gave up' in e.lower() for e in errs) else 'ok'
        out['dropped'] = len(errs)
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
    if not src or src[0].get('config', {}).get('probe_style') not in ('alt', 'alt2', 'alt3') or src[0]['config'].get('conf_mode') != 'ptrue':
        raise SystemExit('--input must be a pb_selfprobe.py --probe-style alt --conf ptrue run')
    model = src[0]['model']
    out_path = a.out or a.input.replace('.json', '') + '_variants.json'
    pin, pout = ([float(x) for x in a.price.split(',')] if a.price else (None, None))
    asker = pc.Asker(pc.make_client(a.api), model, pc.provider_body(a.api, a.or_provider, a.quant))
    try:
        asker.ask_logprobs(None, 'Is 2 + 2 = 4? Answer TRUE or FALSE.', top=5, max_tokens=1)
    except pc.NoLogprobs as e:
        raise SystemExit('needs token logprobs: %s (pin a provider with --or-provider)' % e)
    runner = Runner(asker, a.inner, src[0]['config'].get('prior_label', 'accepted'))
    done = {r['id']: r for r in json.load(open(out_path, encoding='utf-8'))} if os.path.exists(out_path) else {}
    todo = [r for r in src if done.get(r['id'], {}).get('status') != 'ok']
    print('P(TRUE) variants | %s | %d chains, %d to run' % (model, len(src), len(todo)), flush=True)
    stopped = None
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(runner.run, r) for r in todo]
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result()
            done[r['id']] = r
            pc.save_json(out_path, list(done.values()))
            print('  %-10s %-5s dropped %d | %d/%d | %s' % (r['id'], r['status'], r['dropped'], n, len(todo),
                                                         pc.spend(asker.tokens, pin, pout)), flush=True)
            if a.budget is not None and pin is not None and \
                    asker.tokens['in'] / 1e6 * pin + asker.tokens['out'] / 1e6 * pout >= a.budget:
                stopped = 'budget reached'
                for g in futs:
                    g.cancel()
                break
    pc.save_json(out_path, list(done.values()))
    print('saved %d | first-token failures %d | %s' % (len(done), runner.fail, pc.spend(asker.tokens, pin, pout)))
    if stopped:
        print('STOPPED: %s. Re-run to continue.' % stopped)
        sys.exit(2)


if __name__ == '__main__':
    main()
