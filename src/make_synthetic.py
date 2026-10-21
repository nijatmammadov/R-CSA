import json
import math
import random
import argparse

NAMES = ['Ava', 'Ben', 'Carla', 'Dmitri', 'Elif', 'Farah', 'Goran', 'Hana', 'Ivo', 'Jonas',
         'Kemal', 'Lena', 'Mateo', 'Nora', 'Omar', 'Petra', 'Quinn', 'Rosa', 'Sami', 'Tara']


def t_times_as_many(r):
    a, b, c = r.sample(NAMES, 3)
    item = r.choice(['stamps', 'marbles', 'stickers', 'shells', 'cards'])
    n, k = r.randint(6, 25), r.randint(2, 5)
    m = r.randint(2, n * k - 2)
    q = ('%s has %d times as many %s as %s. %s has %d %s. %s has %d fewer %s than %s. '
         'How many %s do the three of them have together?' % (a, k, item, b, b, n, item, c, m, item, a, item))
    ref = [('%s: %d x %d' % (a, k, n), n * k), ('%s: %d - %d' % (c, n * k, m), n * k - m),
           ('total: %d + %d + %d' % (n * k, n, n * k - m), n * k + n + n * k - m)]
    return q, ref


def t_fraction_of_remaining(r):
    a = r.choice([3, 4, 5])
    b = r.choice([2, 3, 4])
    base = r.randint(3, 12)
    total = a * b * base
    new = r.randint(5, 40)
    shop = r.choice(['bakery', 'bookstore', 'garden shop', 'toy store'])
    item = {'bakery': 'rolls', 'bookstore': 'notebooks', 'garden shop': 'pots', 'toy store': 'yo-yos'}[shop]
    s1 = total // a
    rem1 = total - s1
    s2 = rem1 // b
    rem2 = rem1 - s2
    q = ('A %s had %d %s. On Monday it sold 1/%d of them. On Tuesday it sold 1/%d of the remaining %s. '
         'On Wednesday it received %d new %s. How many %s does it have now?'
         % (shop, total, item, a, b, item, new, item, item))
    ref = [('Monday sold %d / %d' % (total, a), s1), ('left %d - %d' % (total, s1), rem1),
           ('Tuesday sold %d / %d' % (rem1, b), s2), ('left %d - %d' % (rem1, s2), rem2),
           ('now %d + %d' % (rem2, new), rem2 + new)]
    return q, ref


def t_paid_from_change(r):
    a = r.choice(NAMES)
    n1, p1 = r.randint(2, 6), r.randint(3, 15)
    n2, p2 = r.randint(1, 4), r.randint(5, 25)
    ch = r.randint(1, 19)
    i1, i2 = r.sample(['notebooks', 'pens', 'mugs', 'candles', 'scarves', 'plants'], 2)
    q = ('%s bought %d %s at $%d each and %d %s at $%d each. The cashier gave %s $%d in change. '
         'How much money did %s give the cashier?' % (a, n1, i1, p1, n2, i2, p2, a, ch, a))
    c1, c2 = n1 * p1, n2 * p2
    ref = [('%s: %d x %d' % (i1, n1, p1), c1), ('%s: %d x %d' % (i2, n2, p2), c2),
           ('cost %d + %d' % (c1, c2), c1 + c2), ('given = cost + change %d + %d' % (c1 + c2, ch), c1 + c2 + ch)]
    return q, ref


def t_rate_units(r):
    m = r.choice([2, 3, 4, 5, 6])
    s = m * r.randint(1, 3)
    while (60 - s) % m:
        s += 1
    b = r.randint(2, 9)
    h = r.randint(2, 8)
    per_h = (60 - s) // m * b
    q = ('A machine fills %d bottles every %d minutes. Each hour it stops for %d minutes of cleaning and fills '
         'nothing during that time. How many bottles does it fill in %d hours?' % (b, m, s, h))
    ref = [('working minutes per hour 60 - %d' % s, 60 - s), ('batches per hour %d / %d' % (60 - s, m), (60 - s) // m),
           ('bottles per hour %d x %d' % ((60 - s) // m, b), per_h), ('total %d x %d' % (per_h, h), per_h * h)]
    return q, ref


def t_percent_new_price(r):
    while True:
        p = r.choice(range(40, 401, 20))
        a = r.choice([10, 20, 25, 50])
        b = r.choice([10, 20, 25, 50])
        up = p * (100 + a) / 100
        fin = up * (100 - b) / 100
        if up == int(up) and fin == int(fin):
            break
    item = r.choice(['jacket', 'lamp', 'bicycle helmet', 'backpack'])
    q = ('A %s cost $%d. The store raised the price by %d%%, and a month later put it on sale at %d%% off '
         'the new price. What is the sale price?' % (item, p, a, b))
    ref = [('raise %d x %d%%' % (p, a), p * a // 100), ('new price', int(up)),
           ('discount %d%% of %d' % (b, int(up)), int(up) - int(fin)), ('sale price', int(fin))]
    return q, ref


def t_round_up_packs(r):
    a = r.choice(NAMES)
    g, k = r.randint(8, 30), r.randint(2, 4)
    s = r.choice([6, 8, 10, 12])
    have = r.randint(1, s - 1)
    c = r.randint(2, 6)
    need = g * k - have
    packs = math.ceil(need / s)
    if need % s == 0:
        have += 1
        need = g * k - have
        packs = math.ceil(need / s)
    item = r.choice(['juice boxes', 'cupcakes', 'party hats', 'balloons'])
    q = ('%s is having %d guests, and each guest will get %d %s. %s already has %d %s left over. '
         '%s come in packs of %d that cost $%d per pack. How much will %s spend?'
         % (a, g, k, item, a, have, item, item.capitalize(), s, c, a))
    ref = [('needed %d x %d' % (g, k), g * k), ('to buy %d - %d' % (g * k, have), need),
           ('packs ceil(%d / %d)' % (need, s), packs), ('cost %d x %d' % (packs, c), packs * c)]
    return q, ref


def t_grouped_days(r):
    a = r.choice(NAMES)
    while True:
        x, y, z = r.randint(20, 90), r.randint(20, 120), r.randint(5, 60)
        if (x + y + z) % 6 == 0:
            break
    q = ('%s is reading a long book. On days one and two %s read a total of %d pages. On day three %s read '
         'nothing. On days four and five %s read %d pages in total, and on day six %s read %d pages. On '
         'average, how many pages did %s read per day over the six days?' % (a, a, x, a, a, y, a, z, a))
    ref = [('total %d + 0 + %d + %d' % (x, y, z), x + y + z), ('average %d / 6' % (x + y + z), (x + y + z) // 6)]
    return q, ref


def t_backwards(r):
    a = r.choice(NAMES)
    c = r.randint(5, 30)
    b = r.randint(3, 20)
    g = r.randint(3, 15)
    item = r.choice(['stickers', 'coins', 'cookies', 'beads'])
    before_half = 2 * c
    before_recv = before_half - b
    start = before_recv + g
    if before_recv <= 0:
        b = 1
        before_recv = before_half - b
        start = before_recv + g
    q = ('%s gave away %d %s, then received %d more, then used half of the %s %s had. %s now has %d %s. '
         'How many %s did %s start with?' % (a, g, item, b, item, a, a, c, item, item, a))
    ref = [('before spending half 2 x %d' % c, before_half), ('before receiving %d - %d' % (before_half, b), before_recv),
           ('start %d + %d' % (before_recv, g), start)]
    return q, ref


def t_future_ages(r):
    a, b = r.sample(NAMES, 2)
    n, k, y = r.randint(4, 14), r.randint(2, 4), r.randint(3, 12)
    q = ('%s is %d times as old as %s. %s is %d years old now. What will be the sum of their ages in %d years?'
         % (a, k, b, b, n, y))
    ref = [('%s now %d x %d' % (a, k, n), k * n), ('%s then %d + %d' % (a, k * n, y), k * n + y),
           ('%s then %d + %d' % (b, n, y), n + y), ('sum', k * n + n + 2 * y)]
    return q, ref


def t_schedule_exceptions(r):
    a = r.choice(NAMES)
    w = r.randint(6, 12)
    h = r.randint(3, 8)
    h2 = h + r.randint(1, 3)
    h3 = h + r.randint(2, 5)
    act = r.choice(['practised piano', 'volunteered', 'trained', 'studied French'])
    q = ('Over %d weeks, %s %s %d hours every week, with two exceptions: for two weeks in a row %s %s %d hours '
         'each week, and in one other week %d hours. How many hours did %s spend in total over the %d weeks?'
         % (w, a, act, h, a, act.split()[0], h2, h3, a, w))
    ref = [('normal weeks %d - 3' % w, w - 3), ('normal hours %d x %d' % (w - 3, h), (w - 3) * h),
           ('two weeks 2 x %d' % h2, 2 * h2), ('total %d + %d + %d' % ((w - 3) * h, 2 * h2, h3), (w - 3) * h + 2 * h2 + h3)]
    return q, ref


TEMPLATES = [t_times_as_many, t_fraction_of_remaining, t_paid_from_change, t_rate_units, t_percent_new_price,
             t_round_up_packs, t_grouped_days, t_backwards, t_future_ages, t_schedule_exceptions]


def h_two_rate_journey(r):
    pairs = [(r1, r2) for r1 in (40, 45, 50, 60, 75, 90) for r2 in (30, 36, 40, 45, 50, 60) if r2 < r1]
    while True:
        r1, r2 = r.choice(pairs)
        t1 = r.randint(2, 5)
        d = r1 * t1
        if d % r2 == 0 and d // r2 != t1:
            break
    t2 = d // r2
    job = r.choice(['courier', 'delivery van', 'cyclist'])
    q = ('A %s travels %d km at %d km/h, then returns along the same road at %d km/h. '
         'How many MINUTES longer does the return take than the outward trip?' % (job, d, r1, r2))
    ref = [('time out = %d / %d' % (d, r1), t1), ('time back = %d / %d' % (d, r2), t2),
           ('hours longer = %d - %d' % (t2, t1), t2 - t1), ('minutes longer = %d x 60' % (t2 - t1), (t2 - t1) * 60)]
    return q, ref


def h_up_then_down(r):
    while True:
        p = r.choice(range(120, 1001, 40))
        a = r.choice([25, 50, 75])
        b = r.choice([10, 20, 25])
        up = p + p * a // 100
        if p * a % 100 or up * b % 100:
            continue
        cut = up * b // 100
        fin = up - cut
        if fin > p:
            break
    item = r.choice(['bicycle', 'sofa', 'camera', 'guitar'])
    q = ('A %s cost $%d. The shop raised the price by %d%%, then later took %d%% off the raised price. '
         'How many dollars MORE than the original $%d is the final price?' % (item, p, a, b, p))
    ref = [('rise = %d x %d%%' % (p, a), p * a // 100), ('raised price = %d + %d' % (p, p * a // 100), up),
           ('discount = %d x %d%%' % (up, b), cut), ('final price = %d - %d' % (up, cut), fin),
           ('more than original = %d - %d' % (fin, p), fin - p)]
    return q, ref


def h_tiles_two_fractions(r):
    a, b = r.choice([(4, 3), (3, 2), (4, 2), (6, 3), (5, 2)])
    c = r.choice([8, 10, 12])
    while True:
        T = a * b * r.randint(3, 20)
        u1 = T // a
        l1 = T - u1
        if T % a or l1 % b:
            continue
        u2 = l1 // b
        l2 = l1 - u2
        if l2 > c and l2 % c:
            break
    crates = -(-l2 // c)
    q = ('A warehouse has %d tiles. One job used 1/%d of them. A second job used 1/%d of the tiles that were '
         'LEFT. The rest are packed into crates of %d, and the last crate may be part full. How many crates '
         'are needed?' % (T, a, b, c))
    ref = [('first job used = %d / %d' % (T, a), u1), ('left after first = %d - %d' % (T, u1), l1),
           ('second job used = %d / %d' % (l1, b), u2), ('left after second = %d - %d' % (l1, u2), l2),
           ('crates = ceil(%d / %d)' % (l2, c), crates)]
    return q, ref


def h_backwards_three(r):
    a = r.choice(NAMES)
    f = r.choice([2, 3, 4])
    m = r.randint(2, 15)
    k = r.randint(2, 12)
    start = f * k
    after_give = start - k
    end = after_give - m
    while end < 3:
        k += 1
        start, after_give, end = f * k, f * k - k, f * k - k - m
    item = r.choice(['tokens', 'seedlings', 'badges', 'tickets'])
    q = ('%s gave away 1/%d of the %s, then lost %d more, and now has %d %s. How many %s did %s have at '
         'the start?' % (a, f, item, m, end, item, item, a))
    ref = [('before losing = %d + %d' % (end, m), after_give),
           ('that is %d of every %d at the start' % (f - 1, f), after_give),
           ('start = %d x %d / %d' % (after_give, f, f - 1), start)]
    return q, ref


def h_week_closed_day(r):
    s = r.choice([12, 16, 20, 24, 30, 40])
    w = r.randint(2, 5)
    sat = s // 2
    week = 5 * s + sat
    shop = r.choice(['bakery', 'florist', 'deli'])
    item = {'bakery': 'loaves', 'florist': 'bouquets', 'deli': 'pies'}[shop]
    q = ('A %s sells %d %s on each day from Monday to Friday, half that many on Saturday, and is closed '
         'on Sunday. How many %s does it sell in %d weeks?' % (shop, s, item, item, w))
    ref = [('Monday to Friday = 5 x %d' % s, 5 * s), ('Saturday = %d / 2' % s, sat),
           ('one week = %d + %d' % (5 * s, sat), week), ('%d weeks = %d x %d' % (w, week, w), week * w)]
    return q, ref


def h_ratio_difference(r):
    r1, r2 = r.choice([(5, 2), (7, 3), (4, 1), (5, 3), (7, 2)])
    per = r.randint(4, 20)
    N = (r1 + r2) * per
    a, b = r.sample(NAMES, 2)
    item = r.choice(['sweets', 'walnuts', 'chips', 'tokens'])
    q = ('%s and %s share %d %s in the ratio %d:%d. How many MORE %s does %s get than %s?'
         % (a, b, N, item, r1, r2, item, a, b))
    ref = [('parts = %d + %d' % (r1, r2), r1 + r2), ('one part = %d / %d' % (N, r1 + r2), per),
           ('%s = %d x %d' % (a, r1, per), r1 * per), ('%s = %d x %d' % (b, r2, per), r2 * per),
           ('more = %d - %d' % (r1 * per, r2 * per), (r1 - r2) * per)]
    return q, ref


def h_unit_price_cents(r):
    while True:
        n1, p1 = r.choice([(4, 6), (5, 8), (6, 9), (8, 12), (10, 15)])
        n2, p2 = r.choice([(6, 8), (10, 14), (12, 18), (5, 7), (15, 21)])
        c1, c2 = p1 * 100 // n1, p2 * 100 // n2
        if p1 * 100 % n1 == 0 and p2 * 100 % n2 == 0 and c1 != c2:
            break
    item, one = r.choice([('batteries', 'battery'), ('sponges', 'sponge'),
                          ('light bulbs', 'light bulb'), ('pencils', 'pencil')])
    q = ('Pack A has %d %s for $%d. Pack B has %d %s for $%d. For the better value pack, how many CENTS '
         'cheaper is one %s than in the other pack?' % (n1, item, p1, n2, item, p2, one))
    lo, hi = min(c1, c2), max(c1, c2)
    ref = [('pack A per item = %d x 100 / %d' % (p1, n1), c1), ('pack B per item = %d x 100 / %d' % (p2, n2), c2),
           ('cheaper per item = %d - %d' % (hi, lo), hi - lo)]
    return q, ref


def h_midnight_shift(r):
    sh = r.choice([20, 21, 22, 23])
    sm = r.choice([0, 15, 30, 45])
    dh = r.randint(5, 9)
    dm = r.choice([0, 15, 30, 45])
    k = r.choice([20, 30, 40, 45])
    total = sh * 60 + sm + dh * 60 + dm + k
    if total < 24 * 60 + 30:
        dh += 3
        total = sh * 60 + sm + dh * 60 + dm + k
    end = total - 24 * 60
    q = ('A night shift starts at %02d:%02d and lasts %d hours and %d minutes. Straight after it there is a '
         'handover of %d minutes. How many minutes after midnight does the handover end?'
         % (sh, sm, dh, dm, k))
    ref = [('start in minutes after midnight = %d x 60 + %d' % (sh, sm), sh * 60 + sm),
           ('shift length in minutes = %d x 60 + %d' % (dh, dm), dh * 60 + dm),
           ('shift ends = %d + %d' % (sh * 60 + sm, dh * 60 + dm), sh * 60 + sm + dh * 60 + dm),
           ('handover ends = %d + %d' % (sh * 60 + sm + dh * 60 + dm, k), total),
           ('after midnight = %d - 1440' % total, end)]
    return q, ref


def h_border_area(r):
    a, b = r.randint(5, 14), r.randint(3, 12)
    w = r.choice([1, 2])
    item = r.choice(['rug', 'pond', 'flower bed'])
    oa, ob = a + 2 * w, b + 2 * w
    q = ('A rectangular %s measures %d feet by %d feet. A path %d %s wide is laid all the way around it. '
         'How many square feet does the PATH alone cover?' % (item, a, b, w, 'foot' if w == 1 else 'feet'))
    ref = [('outer length = %d + 2 x %d' % (a, w), oa), ('outer width = %d + 2 x %d' % (b, w), ob),
           ('outer area = %d x %d' % (oa, ob), oa * ob), ('%s area = %d x %d' % (item, a, b), a * b),
           ('path area = %d - %d' % (oa * ob, a * b), oa * ob - a * b)]
    return q, ref


def h_schedule_exceptions(r):
    a = r.choice(NAMES)
    w = r.randint(7, 12)
    h = r.randint(3, 6)
    h2 = h + r.randint(1, 3)
    h3 = h2 + r.randint(1, 3)
    act = r.choice(['practised piano', 'volunteered', 'trained', 'studied French'])
    q = ('Over %d weeks, %s %s %d hours in a normal week. Two of the weeks were not normal: in those two '
         '%s %s %d hours each week. One further week was also not normal: in that week %s %s %d hours. '
         'How many hours in total over the %d weeks?'
         % (w, a, act, h, a, act.split()[0], h2, a, act.split()[0], h3, w))
    ref = [('normal weeks = %d - 3' % w, w - 3), ('normal hours = %d x %d' % (w - 3, h), (w - 3) * h),
           ('two weeks = 2 x %d' % h2, 2 * h2),
           ('total = %d + %d + %d' % ((w - 3) * h, 2 * h2, h3), (w - 3) * h + 2 * h2 + h3)]
    return q, ref


HARD = [h_two_rate_journey, h_up_then_down, h_tiles_two_fractions, h_backwards_three, h_week_closed_day,
        h_ratio_difference, h_unit_price_cents, h_midnight_shift, h_border_area, h_schedule_exceptions]
SETS = {'v1': TEMPLATES, 'v2': HARD}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='synth_questions.json')
    ap.add_argument('--set', choices=sorted(SETS), default='v1',
                    help="v1 = the first set; v2 = the harder set (traps in the middle or at the end)")
    ap.add_argument('--per-template', type=int, default=10)
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()
    out = []
    seen = set()
    for ti, tmpl in enumerate(SETS[a.set]):
        for j in range(a.per_template):
            for salt in range(50):
                r = random.Random('%d|%s|%d|%d' % (a.seed, tmpl.__name__, j, salt) if salt
                                  else '%d|%s|%d' % (a.seed, tmpl.__name__, j))
                q, ref = tmpl(r)
                if q not in seen:
                    break
            seen.add(q)
            ans = ref[-1][1]
            assert isinstance(ans, int) and ans > 0, (tmpl.__name__, ans)
            out.append({'id': ('syn_%02d%02d' if a.set == 'v1' else 'sy2_%02d%02d') % (ti, j), 'template': tmpl.__name__[2:], 'question': q,
                        'reference': [{'step': s, 'value': v} for s, v in ref],
                        'gsm8k_answer': '\n'.join('%s = %s' % (s, v) for s, v in ref) + '\n#### %d' % ans,
                        'source': 'synthetic'})
    json.dump(out, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('%d questions -> %s' % (len(out), a.out))


if __name__ == '__main__':
    main()
