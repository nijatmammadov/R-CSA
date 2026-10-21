import os
import sys
import json
import math
import argparse
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb_common as pc

PROMPT = ('Given the question and previous accepted steps, is the following step valid?\n\n'
          'Question: Tom has 3 apples and buys 2 more. How many apples does he have?\n'
          'Step: "3 + 2 = 5 apples"\n\nAnswer should be either TRUE or FALSE.')


def _get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'thesis-check'})
    with urllib.request.urlopen(req, timeout=30) as f:
        return json.load(f)


def endpoints(model):
    try:
        data = _get('https://openrouter.ai/api/v1/models/%s/endpoints' % model)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise SystemExit('OpenRouter has no model "%s". Find the exact id with:\n'
                             '    python src/check_logprobs.py --search <part of the name>' % model)
        raise
    return (data.get('data') or {}).get('endpoints') or []


def search(text):
    rows = []
    for m in _get('https://openrouter.ai/api/v1/models').get('data') or []:
        if text.lower() not in m.get('id', '').lower():
            continue
        pr = m.get('pricing') or {}
        sp = set(m.get('supported_parameters') or [])
        rows.append((m['id'], _m(pr.get('prompt')), _m(pr.get('completion')),
                     'yes' if ('logprobs' in sp or 'top_logprobs' in sp) else 'no'))
    if not rows:
        raise SystemExit('no OpenRouter model id contains "%s"' % text)
    print('%-55s %10s %10s  logprobs listed' % ('id', 'in $/M', 'out $/M'))
    for r in sorted(rows):
        print('%-55s %10s %10s  %s' % r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=None)
    ap.add_argument('--search', default=None, help='list OpenRouter model ids containing this text')
    ap.add_argument('--all', action='store_true', help='also try providers that do not list logprobs')
    ap.add_argument('--raw', default=None, metavar='PROVIDER',
                    help='print the raw replies of that provider to the R-CSA prompts that read one token')
    a = ap.parse_args()
    if a.search:
        return search(a.search)
    if not a.model:
        raise SystemExit('give --model <id> or --search <text>')
    if a.raw:
        return raw(a.model, a.raw)

    eps = endpoints(a.model)
    if not eps:
        raise SystemExit('no endpoints listed for %s' % a.model)
    print('%d providers for %s' % (len(eps), a.model))
    cands = []
    for e in eps:
        sp = set(e.get('supported_parameters') or [])
        lp = 'logprobs' in sp or 'top_logprobs' in sp
        name = e.get('provider_name') or e.get('name')
        pr = e.get('pricing') or {}
        print('  %-22s logprobs listed: %-5s  price in/out per M: %s / %s  quant: %s'
              % (name, lp, _m(pr.get('prompt')), _m(pr.get('completion')), e.get('quantization')))
        if lp or a.all:
            cands.append(name)

    client = pc.make_client('openrouter')
    print('\nTesting %d provider(s) with one 1-token call each:' % len(cands))
    for name in cands:
        asker = pc.Asker(client, a.model, pc.provider_body('openrouter', order=name), retries=2)
        try:
            text, tops = asker.ask_logprobs(None, PROMPT, top=20, max_tokens=1)
            p = ptrue(tops)
            print('  %-22s OK  first token %r  P(TRUE)=%s  top: %s'
                  % (name, text, '%.3f' % p if p is not None else 'n/a',
                     ', '.join('%r %.2f' % (t, math.exp(l)) for t, l in tops[:5])))
        except pc.NoLogprobs:
            print('  %-22s NO logprobs returned' % name)
        except Exception as e:
            print('  %-22s failed: %s' % (name, str(e)[:120]))


def raw(model, provider):
    judge, _ = pc.load_prompt('css_judge.md')
    q = 'Tom has 3 apples and buys 2 more. How many apples does he have?'
    prompts = {
        'ptrue_full startup': 'Is 2 + 2 = 4? Answer TRUE or FALSE.',
        'css startup': 'Is 2 + 2 = 4? (A) Yes (B) No. Answer with a single letter.',
        'css judge, real step': judge.replace('<<QUESTION>>', q).replace('<<PRIOR>>', '  (none)')
                                     .replace('<<STEP>>', 'apples = 3 + 2 = 5'),
        'css judge, N/A step': judge.replace('<<QUESTION>>', q).replace('<<PRIOR>>', '  (none)')
                                    .replace('<<STEP>>', 'N/A'),
    }
    client = pc.make_client('openrouter')
    body = pc.provider_body('openrouter', order=provider)
    for name, prompt in prompts.items():
        for mt in (1, 8):
            r = client.chat.completions.create(model=model, messages=[{'role': 'user', 'content': prompt}],
                                               max_tokens=mt, temperature=0.0, logprobs=True, top_logprobs=5,
                                               extra_body=body, timeout=120)
            ch = r.choices[0] if getattr(r, 'choices', None) else None
            msg = ch.message if ch else None
            lp = getattr(ch, 'logprobs', None) if ch else None
            content = getattr(lp, 'content', None) if lp else None
            first = content[0] if content else None
            print('%-22s max_tokens=%d | finish %s | text %r | reasoning %r | %d logprob tokens | first: %s'
                  % (name, mt, getattr(ch, 'finish_reason', None), getattr(msg, 'content', None),
                     (getattr(msg, 'reasoning', None) or '')[:60], len(content or []),
                     ', '.join('%r %.2f' % (t.token, math.exp(t.logprob)) for t in (first.top_logprobs or []))
                     if first else '-'))


def _m(x):
    try:
        return '%.3f' % (float(x) * 1e6)
    except (TypeError, ValueError):
        return '?'


def ptrue(tops):
    t = f = 0.0
    for tok, lp in tops:
        s = tok.strip().upper()
        if s and 'TRUE'.startswith(s) or s.startswith('TRUE'):
            t += math.exp(lp)
        elif s and 'FALSE'.startswith(s) or s.startswith('FALSE'):
            f += math.exp(lp)
    return t / (t + f) if t + f > 0 else None


if __name__ == '__main__':
    main()
