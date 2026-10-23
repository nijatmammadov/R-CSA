import os
import re
import sys
import json
import random
import argparse
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb_common as pc

QUESTIONS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'data', 'platinum', 'plat_questions.json')

SOLVE = """Solve the math word problem step by step.

Write each step on its own line, starting with "Step 1:", "Step 2:", and so on. One step = one
thing established or computed. Show the arithmetic in each step.

After the last step, write a final line of the form "Answer: <number>" with the number only."""

SOLVE_YESNO = """Answer the yes/no question step by step.

Write each step on its own line, starting with "Step 1:", "Step 2:", and so on. One step = one
fact or one inference. Write only the step itself, without labels or sources.

After the last step, write a final line of the form "Answer: Yes" or "Answer: No". Choose one of
the two even if you are unsure."""

_STEP = re.compile(r'^\s*\**\s*Step\s*(\d+)\s*\**\s*[:.)-]\s*\**\s*', re.I)
_ANSWER = re.compile(r'^\s*\**\s*(?:final\s+)?answer\s*\**\s*[:=]\s*(.+)$', re.I)
FOLLOWUP = """Question: {question}

Reasoning:
{reply}

Based on this reasoning, is the answer to the question Yes or No? Reply with one word: Yes or No."""
_ENDED = re.compile(r'[.!?)"\'\u201d]\s*$')
_YESNO_ANSWER = re.compile(r'^\W*(yes|no)\b', re.I)
_YESNO_IN_STEP = re.compile(r'answer\s*(?:is|:)\s*\W*(yes|no)\b', re.I)
_NUM = re.compile(r'-?\d{1,3}(?:[ ,]\d{3})+(?:\.\d+)?|-?\d[\d,]*(?:\.\d+)?')


def gold_of(gsm8k_answer):
    m = re.findall(r'####\s*(.+)', gsm8k_answer)
    if not m:
        raise ValueError('no #### in gold: %r' % gsm8k_answer[-80:])
    return number(m[-1])


def number(text):
    nums = _NUM.findall((text or '').replace('$', ''))
    if not nums:
        return None
    return float(nums[-1].replace(',', '').replace(' ', ''))


def parse(reply):
    steps, answer = [], None
    for line in (reply or '').splitlines():
        if not line.strip():
            continue
        a = _ANSWER.match(line)
        if a:
            answer = a.group(1).strip()
            if steps:
                steps[-1] += ' ' + line.strip()
            continue
        m = _STEP.match(line)
        if m:
            steps.append(line[m.end():].strip())
        elif steps:
            steps[-1] += ' ' + line.strip()
    return steps, answer


def yesno(text):
    m = _YESNO_ANSWER.match(text or '')
    return m.group(1).lower() if m else None


def needs_followup(rec):
    return (rec.get('gold') in ('yes', 'no') and rec.get('status') == 'ok' and rec.get('answer') is None
            and 'answer_followup' not in rec and bool(_ENDED.search(rec.get('raw') or '')))


def finish(rec):
    steps, ans = parse(rec['raw'])
    rec.pop('note', None)
    if rec['gold'] in ('yes', 'no'):
        pred = yesno(ans)
        if pred is None and steps:
            m = _YESNO_IN_STEP.findall(steps[-1])
            if m:
                pred = m[-1].lower()
                rec['note'] = 'no Answer line: answer taken from "the answer is ..." in the last step'
        if pred is None and rec.get('answer_followup') in ('yes', 'no'):
            pred = rec['answer_followup']
            rec['note'] = 'no Answer line: Yes/No from a follow-up call to the same model'
        correct = pred == rec['gold']
    else:
        pred = number(ans)
        if pred is None and steps:
            pred = number(steps[-1])
            if pred is not None:
                rec['note'] = 'no Answer line: answer taken from the last number of the last step'
        correct = pred is not None and abs(pred - rec['gold']) < 1e-6
    rec.update(steps=steps, answer=pred, answer_correct=correct)
    rec.pop('error', None)
    if not steps:
        rec.update(status='error', error='no "Step k:" lines in the reply')
        return rec
    rec['status'] = 'ok'
    rec['label'] = -1 if rec['answer_correct'] else None
    if pred is None:
        rec['note'] = 'no final answer: the reply ended without one'
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--api', required=True, choices=sorted(pc.APIS))
    ap.add_argument('--generator', required=True, help='name stored in the records, e.g. Qwen2.5-72B-Instruct')
    ap.add_argument('--model', default=None, help='model id at --api; defaults to pb_common.MODEL_IDS')
    ap.add_argument('--n', type=int, default=40, help='questions, a seeded sample of GSM8K-Platinum')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--questions', default=QUESTIONS,
                    help='question file (id, question, gsm8k_answer with "#### N"); default GSM8K-Platinum. '
                         'synth_questions.json from make_synthetic.py is the development set')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--or-provider', default=None)
    ap.add_argument('--quant', default=None)
    ap.add_argument('--price', default=None, help='USD per million tokens as IN,OUT')
    ap.add_argument('--max-tokens', type=int, default=1024,
                    help='reply length cap; 1024 for GSM8K, raise it for MATH (a cut-off reply counts as wrong)')
    a = ap.parse_args()

    qs = json.load(open(a.questions, encoding='utf-8'))
    qs = sorted(qs, key=lambda q: q['id'])
    random.Random(a.seed).shuffle(qs)
    qs = qs[:a.n]

    model = pc.model_for(a.api, a.generator, a.model)
    asker = pc.Asker(pc.make_client(a.api), model, pc.provider_body(a.api, a.or_provider, a.quant))
    pin, pout = ([float(x) for x in a.price.split(',')] if a.price else (None, None))

    done = {r['id']: r for r in json.load(open(a.out, encoding='utf-8'))} \
        if os.path.exists(a.out) else {}
    for r in done.values():
        if r.get('raw') and r.get('label') in (None, -1):
            finish(r)
    todo = [q for q in qs if q['id'] not in done or not done[q['id']].get('raw')]
    print('Solving | %s | %d questions, %d to do' % (model, len(qs), len(todo)), flush=True)

    def solve(q):
        yn = q.get('gold') in ('yes', 'no')
        rec = {'id': q['id'], 'subset': 'gsm8k', 'generator': a.generator, 'model': model,
               'problem': q['question'], 'gold': q['gold'] if yn else gold_of(q['gsm8k_answer'])}
        try:
            reply = asker.ask(SOLVE_YESNO if yn else SOLVE, q['question'], max_tokens=a.max_tokens,
                              temperature=0.0)
        except pc.APIGaveUp as e:
            rec.update(status='error', error=str(e)[:300])
            return rec
        rec['raw'] = reply
        return finish(rec)

    import time
    from concurrent.futures import as_completed
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(solve, q) for q in todo]
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result()
            done[r['id']] = r
            pc.save_json(a.out, [done[q['id']] for q in qs if q['id'] in done])
            mark = r['status'] if r['status'] != 'ok' else ('right' if r.get('answer_correct') else 'WRONG')
            print('  %-10s %-6s | %d/%d | %.0fs elapsed | api errors %s | %s'
                  % (r['id'], mark, n, len(todo), time.time() - t0, asker.errors or '{}',
                     pc.spend(asker.tokens, pin, pout)), flush=True)
    follow = [r for r in done.values() if needs_followup(r)]
    if follow:
        print('Follow-up Yes/No call for %d replies that ended without an answer' % len(follow), flush=True)

        def ask_followup(r):
            try:
                reply = asker.ask(None, FOLLOWUP.format(question=r['problem'], reply=r['raw'].strip()),
                                  max_tokens=5, temperature=0.0)
            except pc.APIGaveUp:
                return r
            r['answer_followup'] = yesno(reply) or 'unparsed: ' + reply.strip()[:40]
            return finish(r)

        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for r in ex.map(ask_followup, follow):
                done[r['id']] = r
                print('  %-10s follow-up %s -> %s' % (r['id'], r.get('answer_followup'),
                                                      'right' if r['answer_correct'] else 'WRONG'), flush=True)
    out = [done[q['id']] for q in qs if q['id'] in done]
    pc.save_json(a.out, out)

    ok = [r for r in out if r['status'] == 'ok']
    wrong = [r['id'] for r in ok if not r['answer_correct']]
    print('saved %d | %d ok, %d errors | %d right answers, %d wrong: %s | %s'
          % (len(out), len(ok), len(out) - len(ok), len(ok) - len(wrong), len(wrong),
             ', '.join(wrong) or '-', pc.spend(asker.tokens, pin, pout)), flush=True)
    print('Next: write the first wrong step (0-indexed) into "label" for the wrong ones.')


if __name__ == '__main__':
    main()
