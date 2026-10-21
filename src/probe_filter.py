import os
import re
import json
import argparse

PROBE_TYPES = ["factual", "temporal", "quantitative", "logical"]

SYNONYM_GROUPS = [
    {"flavor", "flavour", "type", "kind", "variety", "sort", "category"},
    {"buys", "purchases", "acquires"},
    {"buy", "purchase", "acquire"},
    {"gives", "hands", "donates"},
    {"total", "sum", "aggregate"},
    {"each", "every", "per"},
    {"remaining", "left", "leftover"},
    {"cost", "price", "charge"},
    {"begins", "starts", "commences"},
    {"begin", "start", "commence"},
    {"receives", "gets", "obtains"},
    {"makes", "produces", "creates"},
    {"finishes", "completes", "ends"},
    {"amount", "quantity", "number"},
    {"reduces", "decreases", "lowers"},
    {"increases", "raises", "grows"},
]

_DETERMINERS = {"a", "an", "the"}
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def _tokens(s):
    return _TOKEN.findall((s or "").lower())


def _same_group(a, b):
    for g in SYNONYM_GROUPS:
        if a in g and b in g:
            return True
    return False


_MATH_DELIM = re.compile(r"\$+|\\[()\[\]]")


def normalise(s):
    return " ".join(_MATH_DELIM.sub("", s or "").split())


def is_copy(original, probe):
    if probe is None:
        return False
    return normalise(original).lower() == normalise(probe).lower()


def is_vacuous(original, probe):
    if probe is None:
        return False
    o, p = _tokens(original), _tokens(probe)
    if o == p:
        return False
    o = [t for t in o if t not in _DETERMINERS]
    p = [t for t in p if t not in _DETERMINERS]
    if o == p:
        return True
    if len(o) != len(p):
        return False
    diffs = [(x, y) for x, y in zip(o, p) if x != y]
    if not diffs:
        return True
    return all(_same_group(x, y) for x, y in diffs)


def filter_record(rec):
    n = 0
    for s in rec.get("steps_with_probes", []):
        orig = s.get("original") or ""
        for pt in PROBE_TYPES:
            pv = s.get(pt)
            if pv is None or pv.strip() == orig.strip():
                continue
            if is_copy(orig, pv):
                s[pt] = orig
                s["%s_check" % pt] = ("FILTERED - identical to the original once math "
                                      "delimiters and whitespace are normalised")
                n += 1
                continue
            if is_vacuous(orig, pv):
                s[pt] = orig
                s["%s_check" % pt] = ("FILTERED - synonym/determiner-only edit, not a "
                                      "counterfactual; converted to a copy")
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    total = sum(1 for r in data for s in r.get("steps_with_probes", [])
                for pt in PROBE_TYPES
                if s.get(pt) and s[pt].strip() != (s.get("original") or "").strip())
    removed = sum(filter_record(r) for r in data)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("probe_filter: %d records" % len(data))
    print("  non-copy probes in     : %d" % total)
    print("  vacuous, converted     : %d (%.1f%%)"
          % (removed, 100.0 * removed / max(total, 1)))
    print("  real probes remaining  : %d" % (total - removed))
    print("wrote %s" % args.output)


if __name__ == "__main__":
    main()
