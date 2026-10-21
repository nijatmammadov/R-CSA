import os
import re
import json
import time
import hashlib
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))

SUBSETS = ['gsm8k', 'math', 'olympiadbench', 'omnimath']

APIS = {
    'vertex': (None, None),
    'nim': ('https://integrate.api.nvidia.com/v1', 'NVIDIA_API_KEY'),
    'ollama': (None, None),
    'openrouter': ('https://openrouter.ai/api/v1', 'OPENROUTER_API_KEY'),
    'groq': ('https://api.groq.com/openai/v1', 'GROQ_API_KEY'),
    'vllm': (None, None),
    'hf': ('https://router.huggingface.co/v1', 'HF_TOKEN'),
}
_HF_IDS = {
    'Llama-3.1-8B-Instruct': 'meta-llama/Llama-3.1-8B-Instruct',
    'Llama-3.1-70B-Instruct': 'meta-llama/Llama-3.1-70B-Instruct',
    'Qwen2.5-7B-Instruct': 'Qwen/Qwen2.5-7B-Instruct',
    'Qwen2.5-72B-Instruct': 'Qwen/Qwen2.5-72B-Instruct',
    'Mistral-7B-Instruct-v0.3': 'mistralai/Mistral-7B-Instruct-v0.3',
}
MODEL_IDS = {
    'nim': {
        'Llama-3.1-8B-Instruct': 'meta/llama-3.1-8b-instruct',
        'Llama-3.1-70B-Instruct': 'meta/llama-3.1-70b-instruct',
    },
    'ollama': {
        'Llama-3.1-8B-Instruct': 'llama3.1:8b-instruct-fp16',
        'Qwen2.5-7B-Instruct': 'qwen2.5:7b-instruct-fp16',
    },
    'openrouter': {
        'Llama-3.1-8B-Instruct': 'meta-llama/llama-3.1-8b-instruct',
        'Llama-3.1-70B-Instruct': 'meta-llama/llama-3.1-70b-instruct',
        'Qwen2.5-7B-Instruct': 'qwen/qwen-2.5-7b-instruct',
        'Qwen2.5-72B-Instruct': 'qwen/qwen-2.5-72b-instruct',
        'gpt-oss-120b': 'openai/gpt-oss-120b',
        'Llama-3.3-70B-Instruct': 'meta-llama/llama-3.3-70b-instruct',
        'GPT-4o-mini': 'openai/gpt-4o-mini',
        'DeepSeek-V3-0324': 'deepseek/deepseek-chat-v3-0324',
        'Gemma-3-27B-it': 'google/gemma-3-27b-it',
        'Qwen3-Next-80B-A3B-Instruct': 'qwen/qwen3-next-80b-a3b-instruct',
        'Qwen3-30B-A3B-Instruct-2507': 'qwen/qwen3-30b-a3b-instruct-2507',
        'Qwen3-14B': 'qwen/qwen3-14b',
        'Gemma-3-12B-it': 'google/gemma-3-12b-it',
        'Mistral-Small-3.1-24B': 'mistralai/mistral-small-3.1-24b-instruct',
        'Gemma-4-31B-it': 'google/gemma-4-31b-it',
    },
    'groq': {
        'Llama-3.1-8B-Instruct': 'llama-3.1-8b-instant',
    },
    'vllm': _HF_IDS,
    'hf': _HF_IDS,
    'vertex': {
        'Gemini-2.5-Flash': 'gemini-2.5-flash',
    },
}


def model_for(api, generator, override=None):
    m = override or MODEL_IDS.get(api, {}).get(generator)
    if not m:
        raise SystemExit('no %s model id known for %s; pass --model' % (api, generator))
    return m


MIN_PROBE_LEN_RATIO = 0.7


def is_truncated(original, probe):
    return len(probe.strip()) < MIN_PROBE_LEN_RATIO * len(original.strip())


_LEAD_MARK = re.compile(r'^\s*(?:[-*\u2022]|\d{1,2}[.)])\s+')
_LEAD_HEAD = re.compile(r"^[A-Z][A-Za-z0-9 ,'()/-]{0,60}:\s+(?=\S)")


def _flat(t):
    return ' '.join(re.sub(r'\$+|\\[()\[\]]', '', t or '').split()).lower()


def drops_only_scaffold(original, probe):
    o = original.strip()
    variants = {o}
    m = _LEAD_MARK.match(o)
    if m:
        variants.add(o[m.end():])
    for v in list(variants):
        h = _LEAD_HEAD.match(v)
        if h:
            variants.add(v[h.end():])
    p = _flat(probe)
    return any(_flat(v) == p for v in variants if v != o)


_PUNCT = re.compile(r'(?<!\d)[.,;:]|[.,;:](?!\d)')


def differs_only_in_punctuation(original, probe):
    def key(t):
        return re.sub(r'\s+', '', _PUNCT.sub('', _flat(t)))
    return key(original) == key(probe)


_ANSWER_TAG = re.compile(r'\s*\**\s*(?:final\s+)?answer\s*\**\s*[:=].*$', re.I | re.S)


def drops_only_answer_tag(original, probe):
    o = _ANSWER_TAG.sub('', original or '')
    if o == (original or '') or _ANSWER_TAG.search(probe or ''):
        return False
    return differs_only_in_punctuation(o, probe)


_WORD = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?|\S")


def renames_only_a_variable(original, probe):
    o, p = _WORD.findall(_flat(original)), _WORD.findall(_flat(probe))
    if len(o) != len(p) or o == p:
        return False
    swap = {}
    for a, b in zip(o, p):
        if a == b:
            continue
        if not (len(a) == len(b) == 1 and a.islower() and b.islower() and a.isalpha()
                and b.isalpha() and 'a' not in (a, b)):
            return False
        if swap.setdefault(a, b) != b:
            return False
    return True


def not_a_probe(original, probe):
    return (is_truncated(original, probe) or drops_only_scaffold(original, probe)
            or differs_only_in_punctuation(original, probe)
            or drops_only_answer_tag(original, probe)
            or renames_only_a_variable(original, probe))


_RESULT_NUM = re.compile(r'-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?')


def step_result(text):
    ns = _RESULT_NUM.findall(_ANSWER_TAG.sub('', text or ''))
    return float(ns[-1].replace(',', '')) if ns else None


def not_an_alternative(original, probe):
    p = (probe or '').strip()
    if not p or p.strip('"\' .').upper() == 'NONE':
        return True
    r0 = step_result(original)
    if r0 is not None:
        r1 = step_result(p)
        if r1 is None or abs(r1 - r0) < 1e-9:
            return True
    return (drops_only_scaffold(original, p) or differs_only_in_punctuation(original, p)
            or drops_only_answer_tag(original, p)
            or renames_only_a_variable(original, p))


class APIGaveUp(Exception):
    pass


class NoLogprobs(APIGaveUp):
    pass


class Fatal(Exception):
    pass


def load_prompt(name):
    path = os.path.join(_HERE, 'prompts', name)
    text = open(path, encoding='utf-8').read()
    return text, hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def load_processbench(subsets, generator, from_json=None, limit=None, per_subset=None, seed=42, ids=None):
    if from_json:
        rows = [r for r in json.load(open(from_json, encoding='utf-8'))
                if r.get('status', 'ok') == 'ok']
    else:
        from datasets import load_dataset
        rows = []
        for sp in subsets:
            for r in load_dataset('Qwen/ProcessBench', split=sp):
                rows.append({'id': r['id'], 'subset': sp, 'generator': r['generator'],
                             'problem': r['problem'], 'steps': list(r['steps']),
                             'label': int(r['label'])})
    rows = [r for r in rows if r['subset'] in subsets
            and (generator is None or r['generator'] == generator)]
    if per_subset:
        import random
        kept = []
        for sp in subsets:
            rng = random.Random('%d|%s' % (seed, sp))
            pool = sorted((r for r in rows if r['subset'] == sp), key=lambda r: r['id'])
            err = [r for r in pool if r['label'] != -1]
            cor = [r for r in pool if r['label'] == -1]
            rng.shuffle(err)
            rng.shuffle(cor)
            half = per_subset // 2
            take_e = min(len(err), max(half, per_subset - len(cor)))
            take_c = min(len(cor), per_subset - take_e)
            kept += err[:take_e] + cor[:take_c]
        rows = kept
    if ids:
        want = set(ids)
        rows = [r for r in rows if r['id'] in want]
        missing = want - {r['id'] for r in rows}
        if missing:
            raise SystemExit('chain ids not found for this generator/subsets: %s' % sorted(missing))
    if limit:
        per = {}
        kept = []
        for r in rows:
            per[r['subset']] = per.get(r['subset'], 0) + 1
            if per[r['subset']] <= limit:
                kept.append(r)
        rows = kept
    return rows


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class VertexError(Exception):
    def __init__(self, msg, status_code=None):
        super().__init__(msg)
        self.status_code = status_code


class VertexChat:
    def __init__(self, gclient, thinking_budget, rpm=60):
        self._g, self._budget = gclient, thinking_budget
        self.chat = _NS(completions=_NS(create=self._create))
        self.models = gclient.models
        self._gap, self._next, self._lock = 60.0 / max(rpm, 1), 0.0, threading.Lock()

    def _wait_turn(self):
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self._gap
        if start > now:
            time.sleep(start - now)

    def _create(self, model, messages, max_tokens, temperature, timeout=None, logprobs=False,
                top_logprobs=None, extra_body=None):
        from google.genai import types, errors
        system = '\n\n'.join(m['content'] for m in messages if m['role'] == 'system') or None
        user = '\n\n'.join(m['content'] for m in messages if m['role'] != 'system')
        cfg = dict(temperature=temperature, max_output_tokens=max_tokens, system_instruction=system,
                   thinking_config=types.ThinkingConfig(thinking_budget=self._budget))
        if logprobs:
            cfg.update(response_logprobs=True, logprobs=top_logprobs or 1)
        self._wait_turn()
        try:
            r = self._g.models.generate_content(model=model, contents=user,
                                                config=types.GenerateContentConfig(**cfg))
        except errors.APIError as e:
            raise VertexError(str(e)[:500], getattr(e, 'code', None))
        cand = (r.candidates or [None])[0]
        text = ''
        if cand is not None and cand.content is not None:
            text = ''.join(p.text or '' for p in (cand.content.parts or []) if not getattr(p, 'thought', False))
        lp = None
        lr = getattr(cand, 'logprobs_result', None) if cand is not None else None
        if lr and lr.chosen_candidates:
            content = []
            for i, ch in enumerate(lr.chosen_candidates):
                tops = lr.top_candidates[i].candidates if lr.top_candidates and i < len(lr.top_candidates) else []
                content.append(_NS(token=ch.token, logprob=ch.log_probability,
                                   top_logprobs=[_NS(token=c.token, logprob=c.log_probability) for c in tops]))
            lp = _NS(content=content)
        um = r.usage_metadata
        usage = _NS(prompt_tokens=getattr(um, 'prompt_token_count', 0) or 0,
                    completion_tokens=(getattr(um, 'candidates_token_count', 0) or 0)
                    + (getattr(um, 'thoughts_token_count', 0) or 0)) if um else None
        return _NS(choices=[_NS(message=_NS(content=text), logprobs=lp)], usage=usage, provider='vertex')


def make_client(api):
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(_HERE), '.env'))
    load_dotenv()
    if api not in APIS:
        raise SystemExit('unknown --api %s; choose from %s' % (api, ', '.join(APIS)))
    if api == 'vertex':
        from google import genai
        budget = int(os.getenv('VERTEX_THINKING_BUDGET', '0'))
        rpm = int(os.getenv('VERTEX_RPM', '60'))
        if os.getenv('VERTEX_API_KEY'):
            return VertexChat(genai.Client(vertexai=True, api_key=os.getenv('VERTEX_API_KEY')), budget, rpm)
        if os.getenv('GOOGLE_API_KEY') and not os.getenv('GOOGLE_CLOUD_PROJECT'):
            return VertexChat(genai.Client(vertexai=True, api_key=os.getenv('GOOGLE_API_KEY')), budget, rpm)
        if not os.getenv('GOOGLE_CLOUD_PROJECT'):
            raise SystemExit('set VERTEX_API_KEY, GOOGLE_CLOUD_PROJECT (+ GOOGLE_APPLICATION_CREDENTIALS) '
                             'or GOOGLE_API_KEY in .env')
        return VertexChat(genai.Client(vertexai=True, project=os.getenv('GOOGLE_CLOUD_PROJECT'),
                                       location=os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')), budget, rpm)
    from openai import OpenAI
    base, env = APIS[api]
    if api == 'ollama':
        return OpenAI(base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1'),
                      api_key='ollama')
    if api == 'vllm':
        base = os.getenv('VLLM_BASE_URL')
        if not base:
            raise SystemExit('VLLM_BASE_URL is not set, e.g. http://localhost:8000/v1')
        return OpenAI(base_url=base, api_key=os.getenv('VLLM_API_KEY', 'EMPTY'))
    key = os.getenv(env)
    if not key:
        raise SystemExit('%s is not set (set it in the repo-root .env)' % env)
    return OpenAI(base_url=base, api_key=key)


def provider_body(api, order=None, quant=None):
    if (order or quant) and api != 'openrouter':
        raise SystemExit('--or-provider/--quant only apply to --api openrouter')
    p = {}
    if order:
        p['order'] = [x.strip() for x in order.split(',') if x.strip()]
        p['allow_fallbacks'] = False
    if quant:
        p['quantizations'] = [x.strip() for x in quant.split(',') if x.strip()]
    body = {'provider': p} if p else {}
    effort = os.getenv('LLM_REASONING_EFFORT')
    if effort:
        if api != 'openrouter':
            raise SystemExit('LLM_REASONING_EFFORT is only wired for --api openrouter')
        if effort == 'off':
            body['reasoning'] = {'enabled': False}
        else:
            body['reasoning'] = {'effort': effort, 'exclude': True}
    return body or None


REASONING_MIN_TOKENS = 4096

LOGPROB_TRIES = 10


def spend(tokens, price_in=None, price_out=None):
    t = '%.2fM in / %.2fM out' % (tokens['in'] / 1e6, tokens['out'] / 1e6)
    if price_in is not None and price_out is not None:
        t += ' = $%.2f' % (tokens['in'] / 1e6 * price_in + tokens['out'] / 1e6 * price_out)
    return t


class Asker:
    def __init__(self, client, model, extra_body=None, retries=5):
        self.client, self.model, self.extra_body, self.retries = client, model, extra_body, retries
        self.errors = {}
        self.providers = {}
        self.tokens = {'in': 0, 'out': 0}
        self._lock = threading.Lock()

    def _count(self, d, k):
        with self._lock:
            d[k] = d.get(k, 0) + 1

    def _create(self, system, user, max_tokens, temperature, extra=None):
        msgs = ([{'role': 'system', 'content': system}] if system else []) + \
               [{'role': 'user', 'content': user}]
        delay = 3.0
        last = None
        for _ in range(self.retries):
            try:
                mt = max_tokens
                rsn = (self.extra_body or {}).get('reasoning')
                if rsn and rsn.get('enabled', True):
                    mt = max(mt, REASONING_MIN_TOKENS)
                kw = dict(model=self.model, messages=msgs, max_tokens=mt,
                          temperature=temperature, timeout=180)
                if extra:
                    kw.update(extra)
                if self.extra_body:
                    kw['extra_body'] = self.extra_body
                r = self.client.chat.completions.create(**kw)
                if not getattr(r, 'choices', None):
                    raise RuntimeError('empty choices: %s' % getattr(r, 'error', r))
                self._count(self.providers, str(getattr(r, 'provider', None)))
                u = getattr(r, 'usage', None)
                if u is not None:
                    with self._lock:
                        self.tokens['in'] += getattr(u, 'prompt_tokens', 0) or 0
                        self.tokens['out'] += getattr(u, 'completion_tokens', 0) or 0
                return r
            except Exception as e:
                s = str(e)
                code = getattr(e, 'status_code', None)
                if code in (401, 402, 403, 404, 410):
                    raise Fatal('HTTP %s: %s' % (code, s))
                if code == 400:
                    self._count(self.errors, 'HTTP400')
                    raise APIGaveUp('HTTP 400: %s' % s)
                last = s
                kind = 'RATE' if (code == 429 or 'rate limit' in s.lower()) else type(e).__name__
                self._count(self.errors, kind)
                time.sleep(delay if kind == 'RATE' else 2)
                if kind == 'RATE':
                    delay = min(delay * 2, 60)
        self._count(self.errors, 'GAVE_UP')
        raise APIGaveUp(last)

    def ask(self, system, user, max_tokens, temperature):
        r = self._create(system, user, max_tokens, temperature)
        return r.choices[0].message.content or ''

    def ask_logprobs(self, system, user, top=20, max_tokens=1):
        for attempt in range(LOGPROB_TRIES):
            r = self._create(system, user, max_tokens, 0.0,
                             extra={'logprobs': True, 'top_logprobs': top})
            lp = getattr(r.choices[0], 'logprobs', None)
            content = getattr(lp, 'content', None) if lp is not None else None
            if content:
                break
            self._count(self.errors, 'NO_LOGPROBS')
            time.sleep(1)
        else:
            raise NoLogprobs('%s returned no logprobs %d times (provider %s)'
                             % (self.model, LOGPROB_TRIES, getattr(r, 'provider', None)))
        first = content[0]
        tops = [(clean_token(t.token), t.logprob) for t in (getattr(first, 'top_logprobs', None) or [])]
        if not tops:
            tops = [(clean_token(first.token), first.logprob)]
        return r.choices[0].message.content or '', tops


_TOKEN_MARKS = str.maketrans({'▁': ' ', 'Ġ': ' ', 'Ċ': '\n'})


def clean_token(tok):
    return (tok or '').translate(_TOKEN_MARKS)


_LONE_LATEX = re.compile(r'(?<!\\)((?:\\\\)*)\\(?=[A-Za-z]{2,})')
_BAD_ESCAPE = re.compile(r'(?<!\\)((?:\\\\)*)\\(?![\\"/bfnrtu])')


def xjson(text):
    t = re.sub(r'<think>.*?</think>', '', text or '', flags=re.DOTALL)
    t = re.sub(r'```(?:json)?', '', t).strip()
    i, j = t.find('{'), t.rfind('}')
    if i < 0 or j <= i:
        return None
    t = t[i:j + 1]
    for cand in (_BAD_ESCAPE.sub(r'\1\\\\', _LONE_LATEX.sub(r'\1\\\\', t)), t):
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None


def save_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.25 + 0.25 * attempt)


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == '--list':
        ids = sorted(m.id for m in make_client(sys.argv[2]).models.list())
        flt = sys.argv[3].lower() if len(sys.argv) > 3 else ''
        print('\n'.join(i for i in ids if flt in i.lower()))


_P_OPS = set('+-*/x×')
_P_NUM = re.compile(r'-?\d[\d,]*(?:\.\d+)?')
_TIME = re.compile(r'\b(second|minute|hour|day|week|weekend|month|quarter|year|decade|monday|tuesday|'
                   r'wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|'
                   r'july|august|september|october|november|december|am|pm|before|after|first|second|'
                   r'third|last|start|end|begin|remaining|left|per\s+\w+)s?\b', re.I)


def probe_parts(text):
    if '=' not in (text or ''):
        return None
    seg = [p.strip() for p in text.split('=')]
    if len(seg) < 2:
        return None
    return seg[0].lower(), (seg[-2] if len(seg) >= 3 else ''), seg[-1]


def _nums(s):
    return [x.replace(',', '') for x in _P_NUM.findall(s or '')]


def _ops(s):
    return [c for c in (s or '') if c in _P_OPS]


def probe_edit(original, probe):
    a, b = probe_parts(original), probe_parts(probe)
    if not a or not b:
        return 'unparsed'
    if a[0] != b[0]:
        return 'quantity'
    oa, ob, na, nb = _ops(a[1]), _ops(b[1]), _nums(a[1]), _nums(b[1])
    if oa != ob and na == nb:
        return 'operator'
    if na != nb and oa == ob:
        return 'operand'
    if oa == ob and na == nb and a[2] != b[2]:
        return 'value'
    if oa != ob and na != nb:
        return 'mixed'
    if _TIME.search(original or '') and _TIME.sub('', original or '') != _TIME.sub('', probe or ''):
        return 'time'
    return 'mixed'


_TIME_UNIT = (r'(?:seconds?|minutes?|mins?|hours?|hrs?|days?|nights?|weeks?|weekends?|months?|'
              r'quarters?|years?|decades?)')
_TIME_NUM = re.compile(r'(?<![$\d.,])(\d[\d,]*(?:\.\d+)?)\s*-?\s*' + _TIME_UNIT + r'\b', re.I)
_CLOCK = re.compile(r'\b(\d{1,2}):(\d{2})\b')


def time_numbers(question):
    out = {m.group(1).replace(',', '') for m in _TIME_NUM.finditer(question or '')}
    for h, mi in _CLOCK.findall(question or ''):
        out |= {str(int(h)), str(int(mi)), str(int(h) * 60 + int(mi))}
    return out


def type_matches(kind, original, probe, prior_values=(), time_values=()):
    from collections import Counter
    e = probe_edit(original, probe)
    if e in ('unparsed', 'quantity', 'mixed', 'time'):
        return False
    if kind == 'logical':
        return e == 'operator'
    if kind == 'quantitative':
        return e == 'value'
    if e != 'operand':
        return False
    a, b = probe_parts(original), probe_parts(probe)
    ca, cb = Counter(_nums(a[1])), Counter(_nums(b[1]))
    added, removed = set(cb - ca), set(ca - cb)
    prior = {str(v).replace(',', '') for v in prior_values if v is not None}
    prior |= {p[:-2] for p in prior if p.endswith('.0')}
    timev = {str(v) for v in time_values}
    is_causal = bool(added & prior)
    is_temporal = not is_causal and bool((added | removed) & timev)
    if kind == 'causal':
        return is_causal
    if kind == 'temporal':
        return is_temporal
    if kind == 'factual':
        return not is_causal and not is_temporal
    return False


def alternative_type(original, probe, prior_values=(), time_values=()):
    from collections import Counter
    a, b = probe_parts(original), probe_parts(probe)
    if not a or not b or ' '.join(a[0].split()) != ' '.join(b[0].split()):
        return None
    ea, eb = a[1] or a[2], b[1] or b[2]
    na, nb, oa, ob = _nums(ea), _nums(eb), _ops(ea), _ops(eb)
    ca, cb = Counter(na), Counter(nb)
    if ca == cb:
        if oa != ob or na != nb:
            return 'logical'
        return 'quantitative' if a[2] != b[2] else None
    added, removed = cb - ca, ca - cb
    n_add, n_rem = sum(added.values()), sum(removed.values())
    extra_op, lost_op = Counter(ob) - Counter(oa), Counter(oa) - Counter(ob)
    swap = (n_add, n_rem) == (1, 1) and oa == ob
    add = (n_add, n_rem) == (1, 0) and sum(extra_op.values()) == 1 and not lost_op
    drop = (n_add, n_rem) == (0, 1) and sum(lost_op.values()) == 1 and not extra_op
    if not (swap or add or drop):
        return None
    prior = {str(v).replace(',', '') for v in prior_values if v is not None}
    prior |= {p[:-2] for p in prior if p.endswith('.0')}
    if set(added) & prior:
        return 'causal'
    if (set(added) | set(removed)) & {str(v) for v in time_values}:
        return 'temporal'
    return 'factual'


def _arith(node):
    import ast
    if isinstance(node, ast.Expression):
        return _arith(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _arith(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
        a, b = _arith(node.left), _arith(node.right)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Div):
            return a / b
        if abs(b) > 10:
            raise ValueError('exponent too large')
        return a ** b
    raise ValueError('not plain arithmetic')


def arithmetic_ok(text):
    import ast
    p = probe_parts(text)
    if not p or not p[1]:
        return None
    expr, val = p[1], p[2]
    if ':' in expr or '%' in expr or '%' in val:
        return None
    e = expr.replace('$', '').replace('×', '*').replace('÷', '/').replace('^', '**')
    e = re.sub(r'(?<=\d),(?=\d{3}\b)', '', e)
    e = re.sub(r'(?<=[\d)])\s*[xX]\s*(?=[\d(])', '*', e)
    if re.search(r'[A-Za-z_]', e):
        return None
    vs = _RESULT_NUM.findall(val.replace('$', ''))
    if not vs:
        return None
    vtxt = vs[0].replace(',', '')
    try:
        exact = _arith(ast.parse(e, mode='eval'))
        v = float(vtxt)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, RecursionError):
        return None
    dec = len(vtxt.split('.')[1]) if '.' in vtxt else 0
    tol = max(1e-6 * max(1.0, abs(exact)), 0.5 * 10 ** (-dec) + 1e-9)
    return abs(exact - v) <= tol
