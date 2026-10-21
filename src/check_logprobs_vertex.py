import os
import sys
import math
import argparse

from check_logprobs import PROMPT


def client():
    from dotenv import load_dotenv
    from google import genai
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))
    if os.getenv('VERTEX_API_KEY'):
        print('credentials: VERTEX_API_KEY (Vertex AI express mode)')
        return genai.Client(vertexai=True, api_key=os.getenv('VERTEX_API_KEY'))
    key = os.getenv('GOOGLE_API_KEY')
    if key and not os.getenv('GOOGLE_CLOUD_PROJECT'):
        print('credentials: GOOGLE_API_KEY (Vertex AI express mode)')
        return genai.Client(vertexai=True, api_key=key)
    print('credentials: GOOGLE_CLOUD_PROJECT + service account')
    project, location = os.getenv('GOOGLE_CLOUD_PROJECT'), os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
    if not project:
        raise SystemExit('set VERTEX_API_KEY, GOOGLE_CLOUD_PROJECT (+ GOOGLE_APPLICATION_CREDENTIALS) '
                         'or GOOGLE_API_KEY in .env')
    return genai.Client(vertexai=True, project=project, location=location)


def ptrue(top):
    t = f = 0.0
    for c in top:
        tok = (c.token or '').strip().upper()
        p = math.exp(c.log_probability)
        if tok and 'TRUE'.startswith(tok):
            t += p
        elif tok and 'FALSE'.startswith(tok):
            f += p
    return None if t + f == 0 else t / (t + f)


def check(cl, model):
    from google.genai import types
    tries = [('thinking off', types.GenerateContentConfig(
                 temperature=0, max_output_tokens=1, response_logprobs=True, logprobs=20,
                 thinking_config=types.ThinkingConfig(thinking_budget=0))),
             ('thinking on', types.GenerateContentConfig(
                 temperature=0, max_output_tokens=2048, response_logprobs=True, logprobs=20))]
    for label, cfg in tries:
        try:
            r = cl.models.generate_content(model=model, contents=PROMPT, config=cfg)
        except Exception as e:
            print('  %-12s request refused: %s' % (label, str(e)[:400]))
            continue
        cand = (r.candidates or [None])[0]
        lr = getattr(cand, 'logprobs_result', None) if cand else None
        tops = (lr.top_candidates or []) if lr else []
        if not tops or not tops[0].candidates:
            print('  %-12s answered %r but returned NO logprobs' % (label, (r.text or '')[:40]))
            continue
        first = tops[0].candidates
        print('  %-12s logprobs OK: %d top candidates for the first token; text %r; P(TRUE) = %s'
              % (label, len(first), (r.text or '')[:40], ptrue(first)))
        print('               first token alternatives: %s'
              % ', '.join('%r %.3f' % (c.token, math.exp(c.log_probability)) for c in first[:6]))
        return True
    try:
        r = cl.models.generate_content(model=model, contents=PROMPT,
                                       config=types.GenerateContentConfig(temperature=0, max_output_tokens=2048))
        print('  %-12s model answers %r, so only the logprobs are refused' % ('no logprobs', (r.text or '')[:40]))
    except Exception as e:
        print('  %-12s request refused too: %s' % ('no logprobs', str(e)[:400]))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='gemini-2.5-flash')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    cl = client()
    if a.list:
        try:
            for m in cl.models.list():
                if 'gemini' in (m.name or ''):
                    print(m.name)
        except Exception as e:
            print('cannot list models with these credentials: %s' % str(e)[:160])
            print('pass names to --models instead, e.g. --models gemini-2.5-flash,gemini-2.5-pro,'
                  'gemini-2.5-flash-lite,gemini-2.0-flash')
        return
    ok = []
    for m in [x.strip() for x in a.models.split(',') if x.strip()]:
        print(m)
        if check(cl, m):
            ok.append(m)
    print('\nmodels with logprobs: %s' % (', '.join(ok) or 'none'))


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
