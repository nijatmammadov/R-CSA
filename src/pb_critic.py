import os
import re
import sys
import json
import argparse
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb_common as pc


def tagged(steps):
    return '\n\n'.join('<paragraph_%d>\n%s\n</paragraph_%d>' % (i, s, i)
                       for i, s in enumerate(steps))


def parse_index(text):
    m = re.findall(r'\\boxed\{([^}]*)\}', text or '')
    if not m:
        return None
    try:
        return int(m[-1].strip())
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--generator', required=True)
    ap.add_argument('--api', required=True, choices=sorted(pc.APIS), help='where the model runs (see pb_common.APIS)')
    ap.add_argument('--model', default=None, help='model id at --api; defaults to pb_common.MODEL_IDS')
    ap.add_argument('--subsets', default=','.join(pc.SUBSETS))
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--per-subset', type=int, default=None, help='stratified sample: N chains per subset, half wrong / half correct, seed 42')
    ap.add_argument('--ids', default=None, help='comma-separated chain ids to run, e.g. gsm8k-104,omnimath-162')
    ap.add_argument('--out', default=None)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--max-tokens', type=int, default=8192, help='run_eval.py uses 8192 for non-QwQ models')
    ap.add_argument('--or-provider', default=None)
    ap.add_argument('--quant', default=None)
    ap.add_argument('--max-attempts', type=int, default=3)
    ap.add_argument('--price', default=None, help='USD per million tokens as IN,OUT, e.g. 0.36,0.40 -- prints spend')
    ap.add_argument('--from-json', default=None)
    a = ap.parse_args()

    model = pc.model_for(a.api, a.generator, a.model)
    out_path = a.out or 'pb_%s_critic.json' % a.generator.lower().replace('.', '')
    subsets = [s.strip() for s in a.subsets.split(',') if s.strip()]
    data = pc.load_processbench(subsets, a.generator, a.from_json, a.limit, a.per_subset,
                                ids=a.ids.split(',') if a.ids else None)
    if not data:
        raise SystemExit('no ProcessBench chains for %s' % a.generator)
    template, sha = pc.load_prompt('pb_critique_template.txt')
    asker = pc.Asker(pc.make_client(a.api), model, pc.provider_body(a.api, a.or_provider, a.quant))

    def run(rec):
        out = {k: rec[k] for k in ('id', 'subset', 'generator', 'label')}
        out.update(model=model, config={'template_sha': sha, 'temperature': 0.0})
        prompt = template.format(problem=rec['problem'], tagged_response=tagged(rec['steps']))
        try:
            reply = asker.ask(None, prompt, max_tokens=a.max_tokens, temperature=0.0)
        except pc.APIGaveUp as e:
            out.update(status='error', error=str(e)[:500])
            return out
        out.update(status='ok', prediction=parse_index(reply), critique=reply)
        return out

    done = {r['id']: r for r in json.load(open(out_path, encoding='utf-8'))} \
        if os.path.exists(out_path) else {}
    todo = [r for r in data if done.get(r['id'], {}).get('status') != 'ok'
            and done.get(r['id'], {}).get('attempts', 0) < a.max_attempts]
    print('ProcessBench critic | generator=%s | model=%s | %d chains, %d to run'
          % (a.generator, model, len(data), len(todo)), flush=True)

    for st in range(0, len(todo), a.workers * 4):
        try:
            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                for r in ex.map(run, todo[st:st + a.workers * 4]):
                    r['attempts'] = done.get(r['id'], {}).get('attempts', 0) + 1
                    r['api'] = a.api
                    done[r['id']] = r
        except pc.Fatal as e:
            pc.save_json(out_path, list(done.values()))
            raise SystemExit('STOPPED: %s' % e)
        pc.save_json(out_path, list(done.values()))
        ok = [r for r in done.values() if r['status'] == 'ok']
        print('  saved %d ok (%d unparsed index), %d error | api errors %s'
              % (len(ok), sum(r['prediction'] is None for r in ok),
                 sum(r['status'] == 'error' for r in done.values()), asker.errors or '{}'),
              flush=True)
    pin, pout = ([float(x) for x in a.price.split(',')] if a.price else (None, None))
    print('DONE -> %s | this run: %s' % (out_path, pc.spend(asker.tokens, pin, pout)), flush=True)


if __name__ == '__main__':
    main()
