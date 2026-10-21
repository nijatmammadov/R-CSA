import os
import re
import sys
import json
import argparse
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb_common as pc
from probe_filter import is_copy

CONF_T = 0.1
PROBE_T = 0.0
EXTRACT_T = 0.0

TYPES = {
    'factual': ('Factual Probes alter key entities, relationships, or attributes while '
                'maintaining semantic plausibility.',
                'Einstein developed the theory of relativity.',
                'Newton developed the theory of relativity.'),
    'temporal': ('Temporal Probes modify time-related information, such as dates, durations, '
                 'or temporal sequences.',
                 'World War II ended in 1945.',
                 'World War II ended in 1944.'),
    'quantitative': ('Quantitative Probes perturb numerical values, statistics, or measurements.',
                     'The human heart has four chambers.',
                     'The human heart has three chambers.'),
    'logical': ('Logical Probes introduce logical inconsistencies or invalid causal '
                'relationships.',
                'Rain causes wet streets.',
                'Wet streets cause rain.'),
}
_NUM = re.compile(r'-?\d+(?:\.\d+)?')


class Failed(Exception):
    pass


def fill(template, **kw):
    for k, v in kw.items():
        template = template.replace('<<%s>>' % k.upper(), v)
    return template


def parse_conf(reply):
    m = _NUM.search((reply or '').replace(',', '.'))
    if not m:
        return None
    v = float(m.group())
    if '%' in reply or 1.0 < v <= 100.0:
        v /= 100.0
    return v if 0.0 <= v <= 1.0 else None


class Feng:
    def __init__(self, asker, extract=True, inner=6):
        self.a, self.extract, self.inner = asker, extract, inner
        self.p_probe, self.sha_probe = pc.load_prompt('feng_probe.md')
        self.p_conf, self.sha_conf = pc.load_prompt('feng_confidence.md')
        self.p_extract, self.sha_extract = pc.load_prompt('feng_extract.md')
        self.fails = {'extract': 0, 'probe': 0, 'conf': 0}
        self._lock = threading.Lock()

    def config(self):
        return {'probe_sha': self.sha_probe, 'conf_sha': self.sha_conf,
                'extract_sha': self.sha_extract if self.extract else None,
                'conf_temperature': CONF_T, 'probe_temperature': PROBE_T}

    def _fail(self, k):
        with self._lock:
            self.fails[k] += 1

    def statements(self, text):
        if not self.extract:
            return [text]
        for _ in range(2):
            o = pc.xjson(self.a.ask(None, fill(self.p_extract, text=text), max_tokens=800,
                                    temperature=EXTRACT_T))
            st = (o or {}).get('statements')
            if isinstance(st, list):
                st = [s.strip() for s in st if isinstance(s, str) and s.strip()]
                if st:
                    return st
            self._fail('extract')
        raise Failed('statement extraction failed')

    def probe(self, statement, kind):
        definition, ex_in, ex_out = TYPES[kind]
        user = fill(self.p_probe, type=kind, definition=definition, example_in=ex_in,
                    example_out=ex_out, statement=statement)
        for _ in range(2):
            t = self.a.ask(None, user, max_tokens=300, temperature=PROBE_T).strip()
            t = re.sub(r'^\s*counterfactual\s*:\s*', '', t, flags=re.I).strip().strip('"')
            if t:
                return t.splitlines()[0].strip()
            self._fail('probe')
        raise Failed('empty %s probe' % kind)

    def conf(self, statement):
        for _ in range(3):
            v = parse_conf(self.a.ask(None, fill(self.p_conf, statement=statement),
                                      max_tokens=10, temperature=CONF_T))
            if v is not None:
                return v
            self._fail('conf')
        raise Failed('no confidence number')

    def score(self, statement):
        with ThreadPoolExecutor(max_workers=4) as ex:
            probes = dict(zip(TYPES, ex.map(lambda k: self.probe(statement, k), TYPES)))
        texts = [statement] + [probes[k] for k in TYPES]
        with ThreadPoolExecutor(max_workers=5) as ex:
            confs = list(ex.map(self.conf, texts))
        c0, pcs = confs[0], dict(zip(TYPES, confs[1:]))
        return {'text': statement, 'probes': probes, 'conf': c0, 'probe_conf': pcs,
                'sens': sum(abs(c0 - v) for v in pcs.values()) / len(pcs),
                'var': statistics.pvariance(confs),
                'n_copies': sum(is_copy(statement, p) for p in probes.values())}


def run_chain(fg, rec):
    out = {k: rec.get(k) for k in ('id', 'subset', 'generator', 'label', 'problem')}
    out['steps'] = []
    try:
        steps = [' '.join(s.split()) for s in rec['steps']]

        def extract(text):
            try:
                return fg.statements(text), False
            except Failed:
                return [text], True

        def score(st):
            try:
                return fg.score(st)
            except Failed as e:
                return {'text': st, 'failed': str(e)}

        with ThreadPoolExecutor(max_workers=fg.inner) as ex:
            sts = list(ex.map(extract, steps))
        out['extract_fallback'] = [i for i, (_, fb) in enumerate(sts) if fb]
        out['dropped_statements'] = 0
        for i, (text, (ss, _)) in enumerate(zip(steps, sts)):
            with ThreadPoolExecutor(max_workers=fg.inner) as ex:
                scored = list(ex.map(score, ss))
            out['dropped_statements'] += sum('failed' in x for x in scored)
            out['steps'].append({'i': i, 'text': text,
                                 'statements': [x for x in scored if 'failed' not in x],
                                 'dropped': [x['text'] for x in scored if 'failed' in x]})
        out['status'] = 'ok'
    except (Failed, pc.APIGaveUp) as e:
        out.update(status='error', error=str(e)[:500])
    return out


def run_statement(fg, rec):
    out = {k: rec.get(k) for k in ('id', 'question', 'statement', 'hallucinated')}
    try:
        s = fg.score(rec['statement'])
        out.update(probes=s['probes'], conf=s['conf'], probe_conf=s['probe_conf'],
                   sens=s['sens'], var=s['var'], n_probes=len(s['probes']),
                   n_copies=s['n_copies'], status='ok')
    except (Failed, pc.APIGaveUp) as e:
        out.update(status='error', error=str(e)[:500], sens=None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--api', required=True, choices=sorted(pc.APIS))
    ap.add_argument('--generator', default=None, help='chains: the model that wrote them')
    ap.add_argument('--model', default=None)
    ap.add_argument('--from-json', default=None, help='chains in ProcessBench shape')
    ap.add_argument('--subsets', default=','.join(pc.SUBSETS))
    ap.add_argument('--ids', default=None)
    ap.add_argument('--ids-from', default=None, help='take the ids of the completed records of this run file')
    ap.add_argument('--statements', default=None, help='records with id, statement, hallucinated')
    ap.add_argument('--no-extract', action='store_true', help='treat each step as one statement')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--inner', type=int, default=6)
    ap.add_argument('--or-provider', default=None)
    ap.add_argument('--quant', default=None)
    ap.add_argument('--price', default=None)
    ap.add_argument('--budget', type=float, default=None)
    a = ap.parse_args()
    if a.from_json and a.statements:
        raise SystemExit('give --statements, or chains (--from-json, or ProcessBench from HF), not both')
    if a.ids_from:
        a.ids = ','.join(r['id'] for r in json.load(open(a.ids_from, encoding='utf-8'))
                         if r.get('status') == 'ok')

    if a.statements:
        items = [r for r in json.load(open(a.statements, encoding='utf-8'))
                 if r.get('statement') is not None]
        model = a.model
        if not model:
            raise SystemExit('--statements needs --model')
        work = run_statement
    else:
        if not a.generator:
            raise SystemExit('chains need --generator')
        items = pc.load_processbench([s for s in a.subsets.split(',') if s], a.generator,
                                     a.from_json, ids=a.ids.split(',') if a.ids else None)
        model = pc.model_for(a.api, a.generator, a.model)
        work = run_chain
    if a.ids and a.statements:
        want = set(a.ids.split(','))
        items = [r for r in items if r['id'] in want]

    pin, pout = ([float(x) for x in a.price.split(',')] if a.price else (None, None))
    if a.budget is not None and pin is None:
        raise SystemExit('--budget needs --price')
    asker = pc.Asker(pc.make_client(a.api), model, pc.provider_body(a.api, a.or_provider, a.quant))
    fg = Feng(asker, extract=not (a.no_extract or a.statements), inner=a.inner)

    done = {r['id']: r for r in json.load(open(a.out, encoding='utf-8'))} \
        if os.path.exists(a.out) else {}
    cfg = fg.config()
    done = {i: r for i, r in done.items() if r.get('config') == cfg and r.get('status') == 'ok'}
    todo = [r for r in items if r['id'] not in done]
    print('Feng reconstruction | %s | %d items, %d to run' % (model, len(items), len(todo)), flush=True)

    stopped = None
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, fg, r) for r in todo]
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result()
            r.update(model=model, config=cfg, api=a.api)
            done[r['id']] = r
            pc.save_json(a.out, list(done.values()))
            print('  %-18s %-6s | %d/%d | %s' % (r['id'], r['status'], n, len(todo),
                                                pc.spend(asker.tokens, pin, pout)), flush=True)
            if a.budget is not None and (asker.tokens['in'] / 1e6 * pin
                                         + asker.tokens['out'] / 1e6 * pout) >= a.budget:
                stopped = 'budget reached'
                for g in futs:
                    g.cancel()
                break
    res = list(done.values())
    pc.save_json(a.out, res)
    print('saved %d ok, %d error | parse failures %s | api errors %s | %s'
          % (sum(r['status'] == 'ok' for r in res), sum(r['status'] != 'ok' for r in res),
             fg.fails, asker.errors or '{}', pc.spend(asker.tokens, pin, pout)), flush=True)
    if stopped:
        print('STOPPED: %s. Re-run to continue.' % stopped)
        sys.exit(2)
    print('DONE -> %s' % a.out)


if __name__ == '__main__':
    main()
