import os
import sys
import json
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb_common as pc
from css import Failed, letter_prob
from pb_selfprobe import parse_alternatives
from probe_filter import is_copy, is_vacuous

FACT_TYPES = ['factual', 'temporal', 'quantitative', 'logical', 'causal']


def why_dropped(claim, probe, seen):
    if not probe or probe.strip('"\' .').upper() == 'NONE':
        return 'none'
    if (probe == claim or is_copy(claim, probe) or is_vacuous(claim, probe)
            or pc.differs_only_in_punctuation(claim, probe) or pc.drops_only_scaffold(claim, probe)):
        return 'same claim'
    if pc.is_truncated(claim, probe):
        return 'truncated'
    if probe.lower() in seen:
        return 'duplicate'
    return None


def clean_claim(reply):
    for line in (reply or '').splitlines():
        line = ' '.join(line.strip().strip('`*"').split())
        if line.lower().startswith('claim:'):
            line = line[6:].strip().strip('"')
        if line:
            return line
    return None


class FactCSS:
    def __init__(self, asker, judge, inner):
        self.a, self.j, self.inner = asker, judge, inner
        self.claim_prompt, self.claim_sha = pc.load_prompt('fact_claim.md')
        self.probe_prompt, self.probe_sha = pc.load_prompt('fact_counterfactuals.md')
        self.judge_prompt, self.judge_sha = pc.load_prompt('fact_judge.md')
        self.lock = threading.Lock()
        self.unread = 0
        self.unparsed = 0

    def config(self):
        return {'version': 'fact-1', 'claim_sha': self.claim_sha, 'probe_sha': self.probe_sha,
                'judge_sha': self.judge_sha, 'probe_types': FACT_TYPES, 'n_regen': 0, 'prior_label': 'neutral'}

    @staticmethod
    def _prior(prior):
        return '\n'.join('  %d. %s' % (j, s) for j, s in enumerate(prior, 1)) if prior else '  (none)'

    def claim(self, step):
        return clean_claim(self.a.ask(None, self.claim_prompt.replace('<<STEP>>', step), max_tokens=150,
                                      temperature=0.0))

    def probes(self, problem, prior, claim):
        user = (self.probe_prompt.replace('<<QUESTION>>', problem).replace('<<PRIOR>>', self._prior(prior))
                .replace('<<STEP>>', claim))
        for temp in (0.0, 0.3, 0.3):
            o = parse_alternatives(self.a.ask(None, user, max_tokens=600, temperature=temp), FACT_TYPES)
            if o:
                return o
            with self.lock:
                self.unparsed += 1
        return {t: 'NONE' for t in FACT_TYPES}

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

    def _map(self, fn, jobs):
        def go(job):
            try:
                return job, fn(job), None
            except (Failed, pc.APIGaveUp) as e:
                return job, None, e
        with ThreadPoolExecutor(max_workers=self.inner) as ex:
            return list(ex.map(go, jobs))

    def run(self, rec):
        out = {k: rec[k] for k in ('id', 'subset', 'generator', 'label', 'problem')}
        full = [' '.join(s.split()) for s in rec['steps']]
        out.update(n_steps=len(full), model=self.a.model, config=self.config())
        errs = []

        claims = [None] * len(full)
        for k, v, e in self._map(lambda k: self.claim(full[k]), range(len(full))):
            if e is not None:
                errs.append('step %d claim: %s' % (k, str(e)[:200]))
            claims[k] = v
        if errs:
            out.update(status='error', error='; '.join(errs)[:1500])
            return out
        steps = [c or f for c, f in zip(claims, full)]

        rows = [{'i': k, 'text': steps[k], 'text_full': full[k], 'shortened': claims[k] is not None,
                 'p': None, 'p_cf': None, 'alts': [], 'dropped': {}, 'regen': []} for k in range(len(steps))]
        jobs = [(k, 'p') for k in range(len(steps))] + [(k, 'p_cf') for k in range(len(steps))] + \
               [(k, 'probes') for k in range(len(steps))]

        def first(job):
            k, slot = job
            if slot == 'p':
                return self.judge(rec['problem'], steps[:k], steps[k])
            if slot == 'p_cf':
                return self.judge_cf(rec['problem'], steps[:k])
            return self.probes(rec['problem'], steps[:k], steps[k])

        raw = {}
        for (k, slot), v, e in self._map(first, jobs):
            if e is not None:
                errs.append('step %d %s: %s' % (k, slot, str(e)[:200]))
            elif slot == 'probes':
                raw[k] = v
            else:
                rows[k][slot] = v
        if errs:
            out.update(status='error', error='; '.join(errs)[:1500])
            return out

        alt_jobs = []
        for k, row in enumerate(rows):
            seen = set()
            for t in FACT_TYPES:
                p = ' '.join((raw[k].get(t) or '').split())
                why = why_dropped(row['text'], p, seen)
                if why:
                    if why != 'none':
                        row['dropped'][why] = row['dropped'].get(why, 0) + 1
                    continue
                seen.add(p.lower())
                row['alts'].append({'text': p, 'source': 'edit', 'slot': t, 'type': t, 'p': None})
                alt_jobs.append((k, len(row['alts']) - 1))

        unscored = []
        for (k, j), v, e in self._map(lambda kj: self.judge(rec['problem'], steps[:kj[0]],
                                                           rows[kj[0]]['alts'][kj[1]]['text']), alt_jobs):
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
    ap.add_argument('--from-json', required=True, help='pb_solve.py output on yes/no questions')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--out', required=True)
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
    data = pc.load_processbench(['gsm8k'], a.generator, a.from_json, a.limit)
    if not data:
        raise SystemExit('no chains for %s in %s' % (a.generator, a.from_json))
    if any(r.get('gold') not in ('yes', 'no') for r in data):
        raise SystemExit('css_fact.py is for yes/no chains; use css.py for arithmetic chains')
    asker = pc.Asker(pc.make_client(a.api), model, pc.provider_body(a.api, a.or_provider, a.quant))
    judge = (pc.Asker(pc.make_client(a.api), model, pc.provider_body(a.api, a.judge_provider, a.quant))
             if a.judge_provider else asker)
    try:
        judge.ask_logprobs(None, 'Is Paris in France? (A) Yes (B) No. Answer with a single letter.', top=5, max_tokens=1)
    except pc.NoLogprobs as e:
        raise SystemExit('needs token logprobs: %s (check_logprobs.py lists providers that return them)' % e)
    css = FactCSS(asker, judge, a.inner)
    cfg = dict(css.config(), providers={'generation': a.or_provider, 'judge': a.judge_provider or a.or_provider})

    done = {r['id']: r for r in json.load(open(a.out, encoding='utf-8'))} if os.path.exists(a.out) else {}
    stale = [i for i, r in done.items() if r.get('config') != cfg]
    for i in stale:
        done.pop(i)
    if stale:
        print('  %d chains were run with other prompts or settings; redoing them' % len(stale))
    todo = [r for r in data if done.get(r['id'], {}).get('status') != 'ok'
            and done.get(r['id'], {}).get('attempts', 0) < a.max_attempts]
    print('R-CSA step probes, non-arithmetic | %s | %d chains, %d to run' % (model, len(data), len(todo)), flush=True)

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
                    print('  %-26s %-5s %2d steps | %.0fs | %s' % (r['id'], r['status'], r.get('n_steps', 0),
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
        print('saved %d ok, %d error | claims rewritten %d/%d, with >=1 step probe %d | unparsed probe replies %d | '
              'unread A/B %d | api errors %s | %s'
              % (len(ok), sum(r.get('status') == 'error' for r in done.values()), sum(s['shortened'] for s in steps),
                 len(steps), sum(bool(s['alts']) for s in steps), css.unparsed, css.unread, asker.errors or '{}',
                 pc.spend(spent()[0], pin, pout)), flush=True)
    if stopped:
        print('STOPPED: %s' % stopped, flush=True)
        sys.exit(2)
    print('DONE -> %s' % a.out, flush=True)


if __name__ == '__main__':
    main()
