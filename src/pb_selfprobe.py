import os
import sys
import re
import json
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb_common as pc
from probe_filter import is_copy, is_vacuous

PROBE_TYPES = ['factual', 'temporal', 'quantitative', 'logical']
ALT_TYPES = ['logical', 'factual', 'temporal', 'quantitative', 'pragmatic']
ALT2_TYPES = ['logical', 'factual', 'temporal', 'quantitative', 'causal']
ALT3_TYPES = ALT2_TYPES + ['reading']
JUDGE_T = 0.1


class StepFailed(Exception):
    pass


def ptrue_of(tops):
    import math
    t = f = 0.0
    for tok, lp in tops:
        s = tok.strip().upper()
        if s and ('TRUE'.startswith(s) or s.startswith('TRUE')):
            t += math.exp(lp)
        elif s and ('FALSE'.startswith(s) or s.startswith('FALSE')):
            f += math.exp(lp)
    return t / (t + f) if t + f > 0 else None


_ALT_KEY = re.compile(r'"(%s)"\s*:\s*"' % '|'.join(sorted(set(ALT_TYPES) | set(ALT3_TYPES))))


def parse_alternatives(raw, types=ALT_TYPES):
    o = pc.xjson(raw)
    if isinstance(o, dict) and any(isinstance(o.get(t), str) for t in types):
        return {t: (o.get(t) if isinstance(o.get(t), str) else 'NONE') for t in types}
    keys = list(_ALT_KEY.finditer(raw or ''))
    if not keys:
        return None
    out = {t: 'NONE' for t in types}
    for m, nxt in zip(keys, keys[1:] + [None]):
        seg = (raw[m.end():nxt.start()] if nxt else raw[m.end():])
        seg = re.sub(r'"\s*[,}]?\s*$', '', seg.strip())
        out[m.group(1)] = seg.replace('\\"', '"').strip()
    return out


def timed(fn, *args):
    t = time.time()
    return fn(*args), time.time() - t


PRIOR_LABELS = {
    'accepted': 'Prior accepted steps:',
    'unverified': 'Previous steps (not yet verified, may contain errors):',
}


def judge_message(problem, prior, cand, prior_label='accepted'):
    parts = ['Question: %s' % problem]
    if prior:
        parts.append(PRIOR_LABELS[prior_label])
        parts += ['  %d. %s' % (i, s) for i, s in enumerate(prior, 1)]
    parts.append('\nCandidate step: %s' % cand)
    return '\n'.join(parts)


class Runner:
    def __init__(self, asker, feng, chunk, inner, prior_label='accepted',
                 judge_prompt='evaluate_qacct_v1_backup.md', conf_mode='verbal', probe_style='corrupt',
                 shorten=False, judge=None, render_full=False):
        self.a, self.feng, self.chunk, self.inner = asker, feng, chunk, inner
        self.j = judge or asker
        self.probe_style = probe_style
        self.shorten = shorten
        self.short_prompt, self.short_sha = pc.load_prompt('step_shorten.md')
        self.short_kept = self.short_tried = 0
        self.prior_label = prior_label
        self.conf_mode = conf_mode
        self.ptrue_prompt, self.ptrue_sha = pc.load_prompt('step_ptrue.md')
        self.render_prompt, self.render_sha = pc.load_prompt('step_render.md')
        self.render_kept = self.render_tried = 0
        self.render_full = render_full
        self.probe_prompt, self.probe_sha = pc.load_prompt(
            'counterfactuals_alt3.md' if probe_style == 'alt3' else
            'counterfactuals_alt2.md' if probe_style == 'alt2' else
            ('counterfactuals_alt1.md' if shorten else 'counterfactuals_alt.md')
            if probe_style == 'alt' else 'counterfactuals_stmt.md')
        self.alt_types = {'alt2': ALT2_TYPES, 'alt3': ALT3_TYPES}.get(probe_style, ALT_TYPES)
        self.judge_prompt, self.judge_sha = pc.load_prompt(judge_prompt)
        self.judge_name = judge_prompt
        self.feng_prompt, self.feng_sha = pc.load_prompt('evaluate_feng.md')
        self.parse_fail = {'probe': 0, 'judge': 0, 'feng': 0, 'shorten': 0, 'render': 0}
        self._lock = threading.Lock()

    def _pf(self, k):
        with self._lock:
            self.parse_fail[k] += 1

    def _probe_call(self, texts):
        listing = 'Statements:\n' + ''.join('%d. %s\n' % (i + 1, t) for i, t in enumerate(texts))
        o = pc.xjson(self.a.ask(self.probe_prompt, listing, max_tokens=4096, temperature=0.0))
        probes = (o or {}).get('probes') or {}
        out = {}
        for i in range(len(texts)):
            e = probes.get(str(i + 1))
            if isinstance(e, dict) and all(isinstance(e.get(t), str) for t in PROBE_TYPES):
                out[i] = e
        return out

    def probes_for(self, steps):
        got = {}
        for st in range(0, len(steps), self.chunk):
            idx = list(range(st, min(st + self.chunk, len(steps))))
            for _ in range(2):
                missing = [i for i in idx if i not in got]
                if not missing:
                    break
                res = self._probe_call([steps[i] for i in missing])
                for j, i in enumerate(missing):
                    if j in res:
                        got[i] = res[j]
            for i in [i for i in idx if i not in got]:
                self._pf('probe')
                for _ in range(2):
                    res = self._probe_call([steps[i]])
                    if 0 in res:
                        got[i] = res[0]
                        break
        lost = [i for i in range(len(steps)) if i not in got]
        if lost:
            raise StepFailed('probe generation failed for steps %s' % lost)
        return got

    def shorten_steps(self, steps):
        def one(k):
            user = self.short_prompt.replace('<<STEP>>', steps[k])
            for temp in (0.0, 0.3):
                try:
                    txt = self.a.ask(None, user, max_tokens=120, temperature=temp)
                except pc.APIGaveUp:
                    return k, None
                cand = ' '.join((txt or '').split())
                cand = cand.split('Rewrite:')[-1].strip().strip('"')
                if cand and '=' in cand and len(cand) <= len(steps[k]) + 20:
                    a, b = pc.step_result(steps[k]), pc.step_result(cand)
                    if a is not None and b is not None and abs(a - b) < 1e-9:
                        return k, cand
                self._pf('shorten')
            return k, None
        with ThreadPoolExecutor(max_workers=self.inner) as ex:
            got = dict(ex.map(one, range(len(steps))))
        with self._lock:
            self.short_tried += len(steps)
            self.short_kept += sum(v is not None for v in got.values())
        return [got[k] for k in range(len(steps))]

    def alt_probes_for(self, problem, steps):
        def one(k):
            prior = ''.join('  %d. %s\n' % (j, s) for j, s in enumerate(steps[:k], 1)).rstrip() or '(none)'
            user = (self.probe_prompt.replace('<<QUESTION>>', problem).replace('<<PRIOR>>', prior)
                    .replace('<<STEP>>', steps[k]))
            raw = ''
            for temp in (0.0, 0.3, 0.3):
                raw = self.a.ask(None, user, max_tokens=1200, temperature=temp)
                o = parse_alternatives(raw, self.alt_types)
                if o:
                    return k, o, None
                self._pf('probe')
            return k, {t: 'NONE' for t in self.alt_types}, (raw or '')[:1500]
        with ThreadPoolExecutor(max_workers=self.inner) as ex:
            res = list(ex.map(one, range(len(steps))))
        return {k: o for k, o, _ in res}, {k: raw for k, _, raw in res if raw is not None}

    def render(self, problem, full, short, alts):
        out = {}
        for temp in (0.0, 0.3):
            todo = {t: p for t, p in alts.items() if t not in out}
            if not todo:
                break
            user = (self.render_prompt.replace('<<QUESTION>>', problem).replace('<<STEP>>', full)
                    .replace('<<SHORT>>', short)
                    .replace('<<ALTS>>', '\n'.join('  "%s": %s' % (t, p) for t, p in todo.items())))
            o = parse_alternatives(self.a.ask(None, user, max_tokens=300 * len(todo), temperature=temp),
                                   list(todo)) or {}
            for t, p in todo.items():
                txt = ' '.join((o.get(t) or '').split())
                want, got = pc.step_result(p), pc.step_result(txt)
                if (txt and txt != full and txt.upper() != 'NONE' and want is not None
                        and got is not None and abs(want - got) < 1e-6):
                    out[t] = txt
        with self._lock:
            self.render_tried += len(alts)
            self.render_kept += len(out)
            self.parse_fail['render'] += len(alts) - len(out)
        return out

    def _score(self, system, user, kind):
        for _ in range(2):
            o = pc.xjson(self.a.ask(system, user, max_tokens=1500, temperature=JUDGE_T))
            try:
                p = float(o['p_correct'])
                if 0.0 <= p <= 1.0:
                    return p, (o.get('error_found') if isinstance(o, dict) else None)
            except (KeyError, TypeError, ValueError):
                pass
            self._pf(kind)
        raise StepFailed('%s judge returned no usable p_correct' % kind)

    def grounded(self, problem, prior, cand):
        if self.conf_mode == 'ptrue':
            return self.ptrue(problem, prior, cand), None
        return self._score(self.judge_prompt, judge_message(problem, prior, cand, self.prior_label), 'judge')

    def ptrue(self, problem, prior, cand):
        head = PRIOR_LABELS[self.prior_label]
        prior_txt = (head + '\n' + ''.join('  %d. %s\n' % (j, s) for j, s in enumerate(prior, 1))
                     if prior else head + ' (none)')
        user = (self.ptrue_prompt.replace('<<QUESTION>>', problem)
                .replace('<<PRIOR>>', prior_txt.rstrip()).replace('<<STEP>>', cand))
        for _ in range(2):
            _, tops = self.j.ask_logprobs(None, user, top=20, max_tokens=1)
            p = ptrue_of(tops)
            if p is not None:
                return p
            self._pf('judge')
        raise StepFailed('judge returned no usable p_correct (no TRUE/FALSE among the top logprobs)')

    def standalone(self, cand):
        return self._score(self.feng_prompt, 'Statement: %s' % cand, 'feng')

    def run(self, rec):
        steps = [' '.join(s.split()) for s in rec['steps']]
        out = {k: rec[k] for k in ('id', 'subset', 'generator', 'label', 'problem')}
        out.update(n_steps=len(steps), model=self.a.model,
                   config={'probe_sha': self.probe_sha, 'judge_sha': self.judge_sha,
                           'feng_sha': self.feng_sha if self.feng else None,
                           'probe_style': self.probe_style, 'judge_temperature': JUDGE_T, 'prior_label': self.prior_label,
                           'judge_prompt': self.judge_name, 'conf_mode': self.conf_mode,
                           'ptrue_sha': self.ptrue_sha if self.conf_mode == 'ptrue' else None,
                           'shorten': self.shorten,
                           'providers': getattr(self, 'providers', None),
                           'shorten_sha': self.short_sha if self.shorten else None,
                           'render_sha': self.render_sha if self.render_full else None})
        alt = self.probe_style in ('alt', 'alt2', 'alt3')
        types = self.alt_types if alt else PROBE_TYPES
        time_values = sorted(pc.time_numbers(rec['problem'])) if self.probe_style in ('alt2', 'alt3') else []
        full = steps
        if self.shorten:
            try:
                short = self.shorten_steps(steps)
            except pc.APIGaveUp as e:
                out.update(status='error', error=str(e)[:500])
                return out
            steps = [sh or st for sh, st in zip(short, steps)]
            out['shortened'] = [i for i, sh in enumerate(short) if sh is not None]
        try:
            if alt:
                raw, failed = self.alt_probes_for(rec['problem'], steps)
                if failed:
                    out['no_alternatives'] = {str(k): v for k, v in failed.items()}
            else:
                raw = self.probes_for(steps)
        except (StepFailed, pc.APIGaveUp) as e:
            out.update(status='error', error=str(e)[:500])
            return out

        if self.probe_style == 'alt3':
            try:
                return self._run_alt3(rec, out, full, steps, raw, time_values)
            except pc.APIGaveUp as e:
                out.update(status='error', error=str(e)[:500])
                return out

        rows = []
        tasks = []
        for k, txt in enumerate(steps):
            real, copies, off_type = {}, [], []
            rule = pc.not_an_alternative if alt else pc.not_a_probe
            prior_values = [pc.step_result(x) for x in steps[:k]]
            for t in types:
                p = ' '.join(raw[k][t].split())
                if (not p or p == txt or is_copy(txt, p) or is_vacuous(txt, p) or rule(txt, p)
                        or p in real.values()):
                    copies.append(t)
                elif self.probe_style == 'alt2' and not pc.type_matches(t, txt, p, prior_values, time_values):
                    off_type.append(t)
                else:
                    real[t] = p
            rows.append({'i': k, 'text': txt, 'probes': real, 'copies': copies,
                         'conf': None, 'error_found': None, 'probe_conf': {},
                         'conf_feng': None, 'probe_conf_feng': {}})
            if alt:
                rows[-1]['style'] = self.probe_style
            if self.probe_style == 'alt2':
                rows[-1].update(off_type={t: ' '.join(raw[k][t].split()) for t in off_type},
                                prior_values=prior_values, time_values=time_values)
            if self.shorten and full[k] != txt:
                rows[-1]['text_full'] = full[k]
            prior = steps[:k]
            tasks.append((k, ('conf',), lambda pr=prior, c=txt: self.grounded(rec['problem'], pr, c)))
            for t, p in real.items():
                tasks.append((k, ('probe_conf', t),
                              lambda pr=prior, c=p: self.grounded(rec['problem'], pr, c)))
            if self.feng:
                tasks.append((k, ('conf_feng',), lambda c=txt: self.standalone(c)))
                for t, p in real.items():
                    tasks.append((k, ('probe_conf_feng', t), lambda c=p: self.standalone(c)))

        return self._score_tasks(out, rows, tasks)

    def _run_alt3(self, rec, out, full, short, raw, time_values):
        rows, kept = [], {}
        for k, txt in enumerate(short):
            prior_values = [pc.step_result(x) for x in short[:k]]
            real, ptype, copies, not_single = {}, {}, [], {}
            for t in self.alt_types:
                p = ' '.join(raw[k][t].split())
                if (not p or p == txt or is_copy(txt, p) or is_vacuous(txt, p) or pc.not_an_alternative(txt, p)
                        or p in real.values()):
                    copies.append(t)
                    continue
                kind = pc.alternative_type(txt, p, prior_values, time_values)
                if kind is None:
                    not_single[t] = p
                else:
                    real[t], ptype[t] = p, kind
            kept[k] = real
            rows.append({'i': k, 'text': full[k] if self.render_full else txt, 'text_full': full[k],
                         'text_short': txt, 'probes': {} if self.render_full else dict(real), 'probes_short': real,
                         'probe_type': ptype, 'copies': copies, 'not_single_edit': not_single,
                         'render_failed': {}, 'prior_values': prior_values, 'time_values': time_values,
                         'style': 'alt3', 'conf': None, 'error_found': None, 'probe_conf': {},
                         'conf_feng': None, 'probe_conf_feng': {}})

        def rend(k):
            if not kept[k]:
                return k, {}
            if full[k] == short[k]:
                return k, dict(kept[k])
            return k, self.render(rec['problem'], full[k], short[k], kept[k])
        if self.render_full:
            with ThreadPoolExecutor(max_workers=self.inner) as ex:
                for k, got in ex.map(rend, range(len(short))):
                    rows[k]['probes'] = got
                    rows[k]['render_failed'] = {t: p for t, p in kept[k].items() if t not in got}
        scored = full if self.render_full else short

        tasks = []
        for k, row in enumerate(rows):
            prior = scored[:k]
            tasks.append((k, ('conf',), lambda pr=prior, c=scored[k]: self.grounded(rec['problem'], pr, c)))
            for t, p in row['probes'].items():
                tasks.append((k, ('probe_conf', t), lambda pr=prior, c=p: self.grounded(rec['problem'], pr, c)))
        return self._score_tasks(out, rows, tasks)

    def _score_tasks(self, out, rows, tasks):
        def go(task):
            k, slot, fn = task
            try:
                return task, fn(), None
            except (StepFailed, pc.APIGaveUp) as e:
                return task, None, str(e)[:300]

        errs, unscored = [], []
        with ThreadPoolExecutor(max_workers=self.inner) as ex:
            for (k, slot, _), res, err in ex.map(go, tasks):
                if err:
                    if len(slot) == 2 and 'returned no usable' in err:
                        unscored.append('step %d %s' % (k, '/'.join(slot)))
                        continue
                    errs.append('step %d %s: %s' % (k, '/'.join(slot), err))
                    continue
                p, why = res
                if len(slot) == 1:
                    rows[k][slot[0]] = p
                    if slot[0] == 'conf':
                        rows[k]['error_found'] = why
                else:
                    rows[k][slot[0]][slot[1]] = p
        out['steps'] = rows
        out['unscored'] = unscored
        if errs:
            out.update(status='error', error='; '.join(errs)[:1500])
        else:
            out['status'] = 'ok'
        return out


def _tokens(*askers):
    seen, out = set(), {'in': 0, 'out': 0}
    for x in askers:
        if id(x) in seen:
            continue
        seen.add(id(x))
        out['in'] += x.tokens['in']
        out['out'] += x.tokens['out']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--generator', required=True, help='ProcessBench generator name, e.g. Llama-3.1-8B-Instruct')
    ap.add_argument('--api', required=True, choices=sorted(pc.APIS), help='where the model runs (see pb_common.APIS)')
    ap.add_argument('--model', default=None, help='model id at --api; defaults to pb_common.MODEL_IDS')
    ap.add_argument('--subsets', default=','.join(pc.SUBSETS))
    ap.add_argument('--limit', type=int, default=None, help='max chains per subset (smoke tests)')
    ap.add_argument('--per-subset', type=int, default=None, help='stratified sample: N chains per subset, half wrong / half correct, seed 42')
    ap.add_argument('--ids', default=None, help='comma-separated chain ids to run, e.g. gsm8k-104,omnimath-162')
    ap.add_argument('--out', default=None)
    ap.add_argument('--workers', type=int, default=4, help='chains in parallel')
    ap.add_argument('--inner', type=int, default=6, help='judge calls in parallel within a chain')
    ap.add_argument('--chunk', type=int, default=4, help='steps per probe-generation call')
    ap.add_argument('--no-feng', action='store_true', help="skip Feng's statement-alone channel")
    ap.add_argument('--or-provider', default=None, help='OpenRouter only: pin providers, comma-separated, no fallback')
    ap.add_argument('--judge-provider', default=None,
                    help='OpenRouter only: provider for the one-token P(TRUE) calls, if it differs from '
                         '--or-provider (same model; e.g. only one provider returns logprobs but it '
                         'cannot serve long generations)')
    ap.add_argument('--quant', default=None, help='OpenRouter only: allowed quantizations, e.g. bf16,fp16')
    ap.add_argument('--max-attempts', type=int, default=3)
    ap.add_argument('--price', default=None, help='USD per million tokens as IN,OUT, e.g. 0.36,0.40 -- prints spend')
    ap.add_argument('--budget', type=float, default=None, help='USD; stop once this run has spent this much (needs --price)')
    ap.add_argument('--from-json', default=None, help='local ProcessBench-shaped records (offline tests)')
    ap.add_argument('--conf', choices=['verbal', 'ptrue'], default='verbal',
                    help="step confidence: 'verbal' = the judge's stated p_correct; 'ptrue' = the "
                         "research plan's P(TRUE | question, previous steps, step) from token logprobs")
    ap.add_argument('--judge-prompt', default='evaluate_qacct_v1_backup.md',
                    help="prompts/ file for the in-context step judge; evaluate_step_short.md is the "
                         "short version for small models (the long one was tuned on Gemini)")
    ap.add_argument('--shorten', action='store_true',
                    help="the research plan's statement conversion: the model first rewrites each step as "
                         "one sentence '<quantity> = <expression> = <value>' (prompts/step_shorten.md), kept "
                         "only when it ends in the same number as the step; alternatives are then generated "
                         "in that format (counterfactuals_alt1.md)")
    ap.add_argument('--probe-style', choices=['corrupt', 'alt', 'alt2', 'alt3'], default='corrupt',
                    help="'corrupt' = minimal edits that make the step wrong (counterfactuals_stmt.md); "
                         "'alt' = plausible alternative steps, one different choice each, generated with "
                         "the question and previous steps in context (counterfactuals_alt.md)")
    ap.add_argument('--render', action='store_true',
                    help="alt3 only: write alternatives back into the full step and read P(TRUE) on full "
                         "text (rejected on synthetic v2; kept for reproducibility)")
    ap.add_argument('--prior-label', choices=sorted(PRIOR_LABELS), default='accepted',
                    help="how earlier steps are introduced to the judge; 'unverified' for the combined model")
    a = ap.parse_args()

    if a.probe_style in ('alt2', 'alt3') and not a.shorten:
        raise SystemExit('--probe-style %s is defined on shortened steps: add --shorten' % a.probe_style)
    if a.render and a.probe_style != 'alt3':
        raise SystemExit('--render is defined for --probe-style alt3 only')
    if a.probe_style == 'alt3' and a.conf != 'ptrue':
        raise SystemExit('--probe-style alt3 is defined with --conf ptrue')
    model = pc.model_for(a.api, a.generator, a.model)
    pin, pout = ([float(x) for x in a.price.split(',')] if a.price else (None, None))
    if a.budget is not None and pin is None:
        raise SystemExit('--budget needs --price IN,OUT to know what a token costs')
    out_path = a.out or 'pb_%s_selfprobe.json' % a.generator.lower().replace('.', '')
    subsets = [s.strip() for s in a.subsets.split(',') if s.strip()]

    data = pc.load_processbench(subsets, a.generator, a.from_json, a.limit, a.per_subset,
                                ids=a.ids.split(',') if a.ids else None)
    if not data:
        raise SystemExit('no ProcessBench chains for generator %s in %s' % (a.generator, subsets))

    asker = pc.Asker(pc.make_client(a.api), model, pc.provider_body(a.api, a.or_provider, a.quant))
    try:
        asker.ask(None, 'Reply with the single word OK.', max_tokens=5, temperature=0.0)
    except pc.Fatal as e:
        raise SystemExit('preflight failed for %s: %s' % (model, e))
    judge = (pc.Asker(pc.make_client(a.api), model, pc.provider_body(a.api, a.judge_provider, a.quant))
             if a.judge_provider else asker)
    runner = Runner(asker, feng=not a.no_feng, chunk=a.chunk, inner=a.inner, prior_label=a.prior_label,
                    judge_prompt=a.judge_prompt, conf_mode=a.conf, probe_style=a.probe_style,
                    shorten=a.shorten, judge=judge, render_full=a.render)
    runner.providers = {'generation': a.or_provider, 'ptrue': a.judge_provider or a.or_provider}
    if a.conf == 'ptrue':
        try:
            judge.ask_logprobs(None, 'Is 2 + 2 = 4? Answer TRUE or FALSE.', top=5, max_tokens=1)
        except pc.NoLogprobs as e:
            raise SystemExit('--conf ptrue needs token logprobs: %s. Pin a provider that returns '
                             'them with --or-provider (check_logprobs.py lists them).' % e)

    done = {r['id']: r for r in json.load(open(out_path, encoding='utf-8'))} \
        if os.path.exists(out_path) else {}
    stale = [i for i, r in done.items()
             if r.get('config', {}).get('prior_label', 'accepted') != a.prior_label
             or r.get('config', {}).get('judge_sha') != runner.judge_sha
             or r.get('config', {}).get('conf_mode', 'verbal') != a.conf
             or r.get('config', {}).get('probe_style', 'corrupt') != a.probe_style
             or bool(r.get('config', {}).get('shorten')) != a.shorten
             or r.get('config', {}).get('probe_sha') != runner.probe_sha
             or r.get('config', {}).get('render_sha') != (runner.render_sha if a.render else None)]
    for i in stale:
        done.pop(i)
    if stale:
        print('  %d chains were run with another judge prompt, prior-steps label, confidence or probe '
              'prompt; redoing them'
              % len(stale))
    todo = [r for r in data if done.get(r['id'], {}).get('status') != 'ok'
            and done.get(r['id'], {}).get('attempts', 0) < a.max_attempts]
    by_sub = {}
    for r in data:
        by_sub.setdefault(r['subset'], [0, 0])
        by_sub[r['subset']][0] += 1
        by_sub[r['subset']][1] += r['label'] != -1
    print('ProcessBench self-probing | generator=%s | judge=%s | feng=%s'
          % (a.generator, model, not a.no_feng), flush=True)
    for s, (n, w) in by_sub.items():
        print('  %-14s %4d chains, %4d with an error' % (s, n, w), flush=True)
    print('  %d to run (%d already ok)' % (len(todo), sum(1 for r in done.values()
                                                          if r.get('status') == 'ok')), flush=True)

    stopped = None
    t0, n_done = time.time(), 0
    for st in range(0, len(todo), a.workers * 2):
        batch = todo[st:st + a.workers * 2]
        try:
            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                futs = {ex.submit(timed, runner.run, rec): rec for rec in batch}
                for f in as_completed(futs):
                    r, secs = f.result()
                    r['attempts'] = done.get(r['id'], {}).get('attempts', 0) + 1
                    r['api'] = a.api
                    r['seconds'] = round(secs, 1)
                    done[r['id']] = r
                    n_done += 1
                    pc.save_json(out_path, list(done.values()))
                    rate = (time.time() - t0) / n_done
                    print('  %-18s %-6s %2d steps %6.0fs | %d/%d this run, ~%.0f min left | %s'
                          % (r['id'], r['status'], r.get('n_steps', 0), secs, n_done, len(todo),
                             rate * (len(todo) - n_done) / 60, pc.spend(_tokens(asker, judge), pin, pout)),
                          flush=True)
                    if a.budget is not None and not stopped:
                        tk = _tokens(asker, judge)
                        spent = tk['in'] / 1e6 * pin + tk['out'] / 1e6 * pout
                        if spent >= a.budget:
                            stopped = ('budget reached: $%.2f spent of $%.2f (chains already in '
                                       'flight finish first). Raise --budget and re-run the same '
                                       'command to continue.' % (spent, a.budget))
                            for other in futs:
                                other.cancel()
        except pc.Fatal as e:
            stopped = str(e)
        pc.save_json(out_path, list(done.values()))
        ok = sum(1 for r in done.values() if r.get('status') == 'ok')
        bad = sum(1 for r in done.values() if r.get('status') == 'error')
        print('  saved %d ok, %d error | api errors %s | parse failures %s | providers %s%s'
              % (ok, bad, asker.errors or '{}', runner.parse_fail, asker.providers,
                 ' | shortened %d/%d steps' % (runner.short_kept, runner.short_tried)
                 if a.shorten else '')
              + (' | rendered %d/%d alternatives' % (runner.render_kept, runner.render_tried)
                 if a.render else ''), flush=True)
        if stopped:
            print('STOPPED: %s\nFix it and re-run; finished records are kept.' % stopped, flush=True)
            sys.exit(2)
        recent = [done[r['id']] for r in batch]
        if len(recent) >= 4 and sum(r['status'] == 'error' for r in recent) > len(recent) / 2:
            print('STOPPED: more than half of the last batch failed. Errors:', flush=True)
            for r in recent:
                if r['status'] == 'error':
                    print('   %s: %s' % (r['id'], r['error'][:200]), flush=True)
            sys.exit(3)

    left = [r['id'] for r in done.values() if r.get('status') == 'error']
    print('DONE -> %s | this run: %s' % (out_path, pc.spend(_tokens(asker, judge), pin, pout)), flush=True)
    if left:
        print('WARNING: %d records in error: %s\nRe-run the same command to retry them (up to '
              '--max-attempts each); score_pb.py excludes and lists whatever is left.'
              % (len(left), left[:20]), flush=True)


if __name__ == '__main__':
    main()
