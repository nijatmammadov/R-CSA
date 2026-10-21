import re
import json
import random
import argparse

SUBJECTS = ['prealgebra', 'algebra', 'number_theory', 'counting_and_probability']
ALL_SUBJECTS = SUBJECTS + ['geometry', 'intermediate_algebra', 'precalculus']


def boxed(solution):
    i = solution.rfind('\\boxed')
    if i < 0:
        return None
    j = solution.find('{', i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(solution)):
        if solution[k] == '{':
            depth += 1
        elif solution[k] == '}':
            depth -= 1
            if depth == 0:
                return solution[j + 1:k]
    return None


def integer_answer(ans):
    if ans is None:
        return None
    s = ans.strip()
    for junk in ('\\$', '$', '\\!', '\\,', ' ', '^\\circ', '^{\\circ}', '\\%', '%', '\\text{}'):
        s = s.replace(junk, '')
    s = re.sub(r'\\text\{\s*\}', '', s).replace('{,}', ',')
    if ',' in s and not re.fullmatch(r'-?\d{1,3}(,\d{3})+', s):
        return None
    s = s.replace(',', '')
    return int(s) if re.fullmatch(r'-?\d+', s) else None


def level_of(x):
    m = re.search(r'\d', str(x))
    return int(m.group()) if m else None


def load(split, subjects):
    from datasets import load_dataset
    rows = []
    if split == 'dev':
        for sub in subjects:
            for r in load_dataset('EleutherAI/hendrycks_math', sub, split='train'):
                rows.append({'problem': r['problem'], 'solution': r['solution'], 'level': level_of(r['level']),
                             'subject': sub, 'answer': boxed(r['solution'])})
        for i, r in enumerate(rows):
            r['id'] = 'mtr_%s_%04d' % (r['subject'][:4], i)
    else:
        seen = {' '.join(r['problem'].split()) for r in load_dataset('HuggingFaceH4/MATH-500', split='test')}
        for sub in subjects:
            for i, r in enumerate(load_dataset('EleutherAI/hendrycks_math', sub, split='test')):
                if ' '.join(r['problem'].split()) in seen:
                    continue
                rows.append({'problem': r['problem'], 'solution': r['solution'], 'level': level_of(r['level']),
                             'subject': sub, 'answer': boxed(r['solution']), 'id': 'mte_%s_%04d' % (sub[:4], i)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['dev', 'test'], required=True)
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--levels', default='3,4,5')
    ap.add_argument('--subjects', default='word',
                    help="'word' = %s; 'all' = all seven MATH subjects; or a comma list" % ','.join(SUBJECTS))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    levels = {int(x) for x in a.levels.split(',')}
    subjects = SUBJECTS if a.subjects == 'word' else ALL_SUBJECTS if a.subjects == 'all' else a.subjects.split(',')
    bad = set(subjects) - set(ALL_SUBJECTS)
    if bad:
        raise SystemExit('unknown subjects %s; choose from %s' % (sorted(bad), ALL_SUBJECTS))
    rows = load(a.split, subjects)
    keep = []
    for r in rows:
        v = integer_answer(r['answer'])
        if v is None or r['level'] not in levels:
            continue
        keep.append({'id': r['id'], 'question': r['problem'], 'gsm8k_answer': r['solution'] + '\n#### %d' % v,
                     'reference': [], 'level': r['level'], 'subject': r['subject'],
                     'source': 'MATH train' if a.split == 'dev' else 'MATH test (not MATH-500)'})
    if len(keep) < a.n:
        raise SystemExit('only %d integer-answer problems at levels %s, asked for %d' % (len(keep), a.levels, a.n))
    keep.sort(key=lambda q: q['id'])
    random.Random(a.seed).shuffle(keep)
    out = sorted(keep[:a.n], key=lambda q: q['id'])
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    by = {}
    for q in out:
        by[(q['subject'], q['level'])] = by.get((q['subject'], q['level']), 0) + 1
    print('wrote %d of %d eligible problems -> %s' % (len(out), len(keep), a.out))
    for k in sorted(by):
        print('  %-26s level %d: %d' % (k[0], k[1], by[k]))


if __name__ == '__main__':
    main()
