import os
import sys
import json
import math
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb_common as pc
from pb_selfprobe import Runner as ProbeRunner
from probe_filter import is_copy, is_vacuous

REGEN_T = 0.7


class Failed(Exception):
    pass


def letter_prob(tops):
    a = b = 0.0
    for tok, lp in tops:
        t = tok.strip().upper().strip('().:*')
        if t == 'A':
            a += math.exp(lp)
        elif t == 'B':
            b += math.exp(lp)
    return a / (a + b) if a + b > 0 else None


class CSS:
    def __init__(self, asker, judge, inner, n_regen):
        self.a, self.j, self.inner, self.n_regen = asker, judge, inner, n_regen
        self.judge_prompt, self.judge_sha = pc.load_prompt('css_judge.md')
        self.regen_prompt, self.regen_sha = pc.load_prompt('css_regenerate.md')
        self.probe = ProbeRunner(asker, feng=False, chunk=4, inner=inner, conf_mode='ptrue',
                                 probe_style='alt3', shorten=True, judge=judge)
        self.lock = threading.Lock()
        self.unread = 0

    def config(self):
        return {'version': 2, 'judge_sha': self.judge_sha, 'regen_sha': self.regen_sha, 'probe_sha': self.probe.probe_sha,
                'shorten_sha': self.probe.short_sha, 'n_regen': self.n_regen, 'regen_temperature': REGEN_T,
                'prior_label': 'neutral'}

    @staticmethod
    def _prior(prior):
        return '\n'.join('  %d. %s' % (j, s) for j, s in enumerate(prior, 1)) if prior else '  (none)'

    def judge(self, problem, prior, step):
        user = (self.judge_prompt.replace('<<QUESTION>>', problem).replace('<<PRIOR>>', self._prior(prior))
                .replace('<<STEP>>', step))
        for _ in range(2):
            _, tops = self.j.ask_logprobs(None, user, top=20, max_tokens=1)
            p = letter_prob(tops)
            if p is not None:
                return p
            with self.lock:
                self.unread += 1
        raise Failed('no A/B among the top logprobs (top: %s)' % ', '.join('%r' % t for t, _ in tops[:5]))

    def judge_cf(self, problem, prior):
        try:
            return self.judge(problem, prior, 'N/A')
        except (Failed, pc.NoLogprobs):
            return None

    def regenerate(self, problem, prior, quantity):
        user = (self.regen_prompt.replace('<<QUESTION>>', problem).replace('<<PRIOR>>', self._prior(prior))
                .replace('<<QUANTITY>>', quantity))
        txt = self.a.ask(None, user, max_tokens=120, temperature=REGEN_T)
        for line in (txt or '').splitlines():
            line = ' '.join(line.strip().strip('`*').split())
            parts = pc.probe_parts(line)
            if parts and pc.step_result(line) is not None:
                expr, val = parts[1], parts[2]
                return ('%s = %s = %s' % (quantity, expr, val)) if expr else ('%s = %s' % (quantity, val))
        return None

    def run(self, rec):
        out = {k: rec[k] for k in ('id', 'subset', 'generator', 'label', 'problem')}
        full = [' '.join(s.split()) for s in rec['steps']]
        out.update(n_steps=len(full), model=self.a.model, config=self.config())
        try:
            short = self.probe.shorten_steps(full)
            steps = [s or f for s, f in zip(short, full)]
            edits, _ = self.probe.alt_probes_for(rec['problem'], steps)
        except pc.APIGaveUp as e:
            out.update(status='error', error=str(e)[:500])
            return out
        time_values = sorted(pc.time_numbers(rec['problem']))
        rows, jobs = [], []
        for k, txt in enumerate(steps):
            shortened = short[k] is not None
            row = {'i': k, 'text': txt, 'text_full': full[k], 'shortened': shortened,
                   'arith_ok': pc.arithmetic_ok(txt) if shortened else None,
                   'p': None, 'p_cf': None, 'alts': [], 'dropped': {}, 'regen': []}
            rows.append(row)
            jobs.append((k, 'p', None, lambda k=k: self.judge(rec['problem'], steps[:k], steps[k])))
            jobs.append((k, 'p_cf', None, lambda k=k: self.judge_cf(rec['problem'], steps[:k])))
            if not shortened:
                continue
            q = txt.split('=')[0].strip() if pc.probe_parts(txt) else None
            if q and self.n_regen:
                for r in range(self.n_regen):
                    jobs.append((k, 'regen', r, lambda k=k, q=q: self.regenerate(rec['problem'], steps[:k], q)))

        def go(job):
            try:
                return job, job[3](), None
            except (Failed, pc.APIGaveUp) as e:
                return job, None, e

        errs = []
        with ThreadPoolExecutor(max_workers=self.inner) as ex:
            for (k, slot, r, _), v, e in ex.map(go, jobs):
                if e is not None:
                    if slot == 'regen' and isinstance(e, Failed):
                        continue
                    errs.append('step %d %s: %s' % (k, slot, str(e)[:200]))
                elif slot == 'regen':
                    if v:
                        rows[k]['regen'].append(v)
                else:
                    rows[k][slot] = v
        if errs:
            out.update(status='error', error='; '.join(errs)[:1500])
            return out

        jobs = []
        for k, row in enumerate(rows):
            if not row['shortened']:
                continue
            txt, v0 = row['text'], pc.step_result(row['text'])
            prior_values = [pc.step_result(x) for x in steps[:k]]
            seen = set()
            cands = [('regen', None, t) for t in row['regen']] + \
                    [('edit', s, ' '.join((edits.get(k) or {}).get(s, '').split())) for s in self.probe.alt_types]
            for src, slot, p in cands:
                if not p or p.upper() == 'NONE':
                    continue
                v = pc.step_result(p)
                if src == 'regen' and v is not None and v0 is not None and abs(v - v0) < 1e-6:
                    continue
                why = None
                if p == txt or is_copy(txt, p) or is_vacuous(txt, p) or pc.not_an_alternative(txt, p):
                    why = 'same claim'
                elif pc.arithmetic_ok(p) is False:
                    why = 'own arithmetic wrong'
                elif v is not None and round(v, 6) in seen:
                    why = 'duplicate value'
                if why:
                    row['dropped'][why] = row['dropped'].get(why, 0) + 1
                    continue
                seen.add(round(v, 6) if v is not None else p)
                row['alts'].append({'text': p, 'source': src, 'slot': slot,
                                    'type': pc.alternative_type(txt, p, prior_values, time_values) or 'mixed',
                                    'p': None})
            row['regen_agree'] = sum(1 for t in row['regen'] if v0 is not None and pc.step_result(t) is not None
                                     and abs(pc.step_result(t) - v0) < 1e-6)
            for j, alt in enumerate(row['alts']):
                jobs.append((k, j, lambda k=k, t=alt['text']: self.judge(rec['problem'], steps[:k], t)))

        def go2(job):
            try:
                return job, job[2](), None
            except (Failed, pc.APIGaveUp) as e:
                return job, None, e

        unscored = []
        with ThreadPoolExecutor(max_workers=self.inner) as ex:
            for (k, j, _), v, e in ex.map(go2, jobs):
                if e is None:
                    rows[k]['alts'][j]['p'] = v
                elif isinstance(e, Failed):
                    unscored.append('step %d alt %d' % (k, j))
                else:
                    errs.append('step %d alt %d: %s' % (k, j, str(e)[:200]))
        for row in rows:
            row['alts'] = [x for x in row['alts'] if x['p'] is not None]
        out.update(steps=rows, unscored=unscored)
        if errs:
            out.update(status='error', error='; '.join(errs)[:1500])
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
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--regen', type=int, default=4, help='regenerations per step (0 = edits only)')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--inner', type=int, default=6)
    ap.add_argument('--or-provider', default=None, help='provider(s) for generation')
    ap.add_argument('--judge-provider', default=None, help='provider(s) for the logprob calls, if different')
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
    judge = (pc.Asker(pc.make_client(a.api), model, pc.provider_body(a.api, a.judge_provider, a.quant))
             if a.judge_provider else asker)
    try:
        judge.ask_logprobs(None, 'Is 2 + 2 = 4? (A) Yes (B) No. Answer with a single letter.', top=5, max_tokens=1)
    except pc.NoLogprobs as e:
        raise SystemExit('needs token logprobs: %s (check_logprobs.py lists providers that return them)' % e)
    css = CSS(asker, judge, a.inner, a.regen)
    cfg = dict(css.config(), providers={'generation': a.or_provider, 'judge': a.judge_provider or a.or_provider})

    done = {r['id']: r for r in json.load(open(a.out, encoding='utf-8'))} if os.path.exists(a.out) else {}
    stale = [i for i, r in done.items() if r.get('config') != cfg]
    for i in stale:
        done.pop(i)
    if stale:
        print('  %d chains were run with other prompts or settings; redoing them' % len(stale))
    todo = [r for r in data if done.get(r['id'], {}).get('status') != 'ok'
            and done.get(r['id'], {}).get('attempts', 0) < a.max_attempts]
    print('CSS (ours) | %s | %d chains, %d to run | %d regenerations per step'
          % (model, len(data), len(todo), a.regen), flush=True)

    def spent():
        tk = {'in': asker.tokens['in'] + (judge.tokens['in'] if judge is not asker else 0),
              'out': asker.tokens['out'] + (judge.tokens['out'] if judge is not asker else 0)}
        return tk, (tk['in'] / 1e6 * pin + tk['out'] / 1e6 * pout) if pin is not None else None

    t0, stopped = time.time(), None
    for st in range(0, len(todo), a.workers * 2):
        batch = todo[st:st + a.workers * 2]
        try:
            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                futs = [ex.submit(css.run, r) for r in batch]
                for f in as_completed(futs):
                    r = f.result()
                    r['config'] = cfg
                    r['attempts'] = done.get(r['id'], {}).get('attempts', 0) + 1
                    done[r['id']] = r
                    pc.save_json(a.out, list(done.values()))
                    tk, usd = spent()
                    print('  %-14s %-5s %2d steps | %.0fs | %s' % (r['id'], r['status'], r.get('n_steps', 0),
                          time.time() - t0, pc.spend(tk, pin, pout)), flush=True)
                    if a.budget is not None and usd is not None and usd >= a.budget and not stopped:
                        stopped = 'budget reached ($%.2f of $%.2f); re-run to continue' % (usd, a.budget)
                        for g in futs:
                            g.cancel()
        except pc.Fatal as e:
            stopped = str(e)
        recent = [done[r['id']] for r in batch if r['id'] in done]
        if stopped:
            break
        if len(recent) >= 4 and sum(r['status'] == 'error' for r in recent) > len(recent) / 2:
            stopped = 'more than half of the last batch failed: ' + '; '.join(
                '%s: %s' % (r['id'], r.get('error', '')[:150]) for r in recent if r['status'] == 'error')
            break
    pc.save_json(a.out, list(done.values()))
    ok = [r for r in done.values() if r.get('status') == 'ok']
    steps = [s for r in ok for s in r['steps']]
    if steps:
        print('saved %d ok, %d error | steps shortened %d/%d, with >=1 step probe %d | unread A/B %d | '
              'api errors %s | %s' % (len(ok), sum(r.get('status') == 'error' for r in done.values()),
                                      sum(s['shortened'] for s in steps), len(steps),
                                      sum(bool(s['alts']) for s in steps), css.unread, asker.errors or '{}',
                                      pc.spend(spent()[0], pin, pout)), flush=True)
    if stopped:
        print('STOPPED: %s' % stopped, flush=True)
        sys.exit(2)
    print('DONE -> %s' % a.out, flush=True)


if __name__ == '__main__':
    main()
