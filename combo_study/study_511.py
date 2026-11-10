"""Back-test all 511 combinations of the nine debaters again, on results worked out from D:/data.

    python -X utf8 combo_study/extract_bars.py      # D:/data -> combo_study/cache (D:/data is only read)
    python -X utf8 combo_study/outcomes.py          # what every named stock did, from the minute bars
    python -X utf8 combo_study/study_511.py [eod|intraday]

Nothing here starts from the rule in force (optimal_combo.py) or from the earlier study's answer. A rule is a set of
debaters, whether only their first choices count or both choices, and how many members must name the same stock:
511 sets x 2 x (1..size) = 4608 rules. Each review has one unit of capital, split equally over the stocks the rule
buys; a share that could not be bought stays cash, a review without a stock earns 0, and a rule's total is the sum of
its reviews' returns (the project's convention, as is leaving out trading costs; net_mid deducts them).

The answer is fixed before looking: the highest total at the average-case sale price among rules with at least
MIN_TRADES trades. The best of 4608 rules always looks good, so three checks measure the luck in it: the same choice
made on one half of the reviews and judged on the other, made again before every review from the reviews sold by then
(walk-forward), and repeated on shuffled results (each review's results dealt at random to the stocks named in it).

Writes combo_study/output/: rules-<mode>.csv (4608 rules), trades-best-<mode>.csv, summary-<mode>.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = HERE / 'cache'
OUT = HERE / 'output'
ENGINE_OUT = ROOT / 'skills' / 'stock-roundtable' / 'output'
NINE = ['炒股养家', '92科比', '陈小群', '赵老哥', '章盟主', '方新侠', '作手新一', '乔帮主', '瑞鹤仙']
PRICES = ('mid', 'best', 'worst', 'open', 'close', '1450', 'limit7')
MIN_TRADES = 30    # completed trades a rule needs to be ranked
COST = 0.1         # percent per round trip, for net_mid
BURN_IN = 40       # reviews before the walk-forward starts choosing
SHUFFLES = 2000
SEED = 20261110
IN_FORCE = (143, 2, True)   # the rule optimal_combo.py runs, looked up after the ranking, for comparison only
POP = np.array([bin(i).count('1') for i in range(512)])


def names(s):
    return '、'.join(n for i, n in enumerate(NINE) if s >> i & 1)


def load(mode, cal):
    """Settled reviews and, per review, every named stock with its result and who named it (bit masks)."""
    pos = {d: i for i, d in enumerate(cal)}
    lag = 2 if mode == 'eod' else 1   # trading days from the review to the sale day
    held = pd.read_csv(ENGINE_OUT / mode / 'days.csv')['date'].astype(str).tolist()
    dates = [d for d in held if pos[d] + lag < len(cal)]
    noms = pd.read_csv(ENGINE_OUT / mode / 'nominations.csv', dtype={'code': str},
                       usecols=['date', 'debater', 'order', 'code'])
    bit = noms['debater'].map({n: 1 << i for i, n in enumerate(NINE)})
    if bit.isna().any() or noms.duplicated(['date', 'debater', 'code']).any():
        raise SystemExit(f'{mode}: nominations.csv is not what this study expects')
    bit = bit.astype(int)
    noms = noms.assign(any=bit, first=bit.where(noms['order'] == 1, 0))
    masks = noms.groupby(['date', 'code'], as_index=False)[['first', 'any']].sum()
    f = pd.read_csv(OUT / f'outcomes-{mode}.csv', dtype={'code': str}).merge(masks, on=['date', 'code'],
                                                                            validate='1:1')
    return dates, f[f['date'].isin(dates)].sort_values(['date', 'code']).reset_index(drop=True)


def line(r):
    return (f'{r["members"]} | {r["choices"]}≥{r["votes"]}票 | 笔数 {r["trades"]} 胜率 {r["win"]:.0%} | 每笔 最好{r["per_trade_best"]:+.2f} '
            f'平均{r["per_trade_mid"]:+.2f} 最差{r["per_trade_worst"]:+.2f} | 合计 最好{r["total_best"]:+.1f} 平均{r["total_mid"]:+.1f} '
            f'最差{r["total_worst"]:+.1f} 开盘卖{r["total_open"]:+.1f} 收盘卖{r["total_close"]:+.1f} 挂7%{r["total_limit7"]:+.1f} '
            f'扣费{r["net_mid"]:+.1f} | 前半{r["first_half"]:+.1f} 后半{r["second_half"]:+.1f} | 回撤{r["max_drawdown"]:+.1f} '
            f't={r["t"]:.2f} 去掉最好3场{r["without_best_3"]:+.1f}')


def run(mode, cal):
    rng = np.random.default_rng(SEED)
    dates, f = load(mode, cal)
    n, m = len(dates), len(f)
    review = f['date'].map({d: i for i, d in enumerate(dates)}).to_numpy()
    g = np.zeros((m, n))
    g[np.arange(m), review] = 1.0
    sold = f['filled'].to_numpy(bool) & f['t1_mid'].notna().to_numpy()
    ret = {x: np.where(sold, f[f't1_{x}'].to_numpy(float), 0.0) for x in PRICES}

    rows = [(s, k, first) for first in (True, False) for s in range(1, 512) for k in range(1, POP[s] + 1)]
    sets, need, first = (np.array(c) for c in zip(*rows))
    m1, m2 = f['first'].to_numpy(), f['any'].to_numpy()
    votes = np.where(first[:, None], POP[m1[None, :] & sets[:, None]], POP[m2[None, :] & sets[:, None]])
    chosen = votes >= need[:, None]                                # [rule, named stock]
    weight = chosen / np.maximum(chosen @ g, 1.0)[:, review]       # equal split of the review's unit
    by = {x: (weight * ret[x]) @ g for x in PRICES}                # [rule, review]
    mid = by['mid']
    done = chosen & sold
    trades = done.sum(1)
    traded = done @ g
    curve = np.concatenate([np.zeros((len(sets), 1)), np.cumsum(mid, 1)], 1)
    h = n // 2
    with np.errstate(invalid='ignore', divide='ignore'):
        t = pd.DataFrame({
            'members': [names(s) for s in sets], 'size': POP[sets], 'choices': np.where(first, '第一选择', '两个选择'),
            'votes': need, 'reviews_bought': (traded > 0).sum(1), 'named': chosen.sum(1), 'trades': trades,
            'stuck': (done & f['t1_stuck'].to_numpy(bool)).sum(1), 'win': (done & (ret['mid'] > 0)).sum(1) / trades,
            **{f'per_trade_{x}': (done * ret[x]).sum(1) / trades for x in ('best', 'mid', 'worst')},
            **{f'total_{x}': by[x].sum(1) for x in PRICES},
            'net_mid': mid.sum(1) - COST * (weight * sold).sum(1),
            'first_half': mid[:, :h].sum(1), 'second_half': mid[:, h:].sum(1),
            't': mid.mean(1) / (mid.std(1, ddof=1) / np.sqrt(n)),
            'max_drawdown': (curve - np.maximum.accumulate(curve, 1)).min(1),
            'without_best_3': mid.sum(1) - np.sort(mid, 1)[:, -3:].sum(1), 'set': sets, 'first_only': first})
    ranked = trades >= MIN_TRADES
    t['ranked'] = ranked
    t['rank'] = t['total_mid'].where(ranked).rank(ascending=False, method='min')
    order = t[ranked].sort_values(['total_mid', 'size', 'votes'], ascending=[False, True, True])
    win = int(order.index[0])
    t.sort_values(['ranked', 'total_mid'], ascending=[False, False]).to_csv(OUT / f'rules-{mode}.csv', index=False,
                                                                           encoding='utf-8-sig')
    where = {(int(s), int(k), bool(a)): i for i, (s, k, a) in enumerate(zip(sets, need, first))}

    print(f'== {mode}: {n} settled reviews {dates[0]} .. {dates[-1]}, {m} named stocks, {int(sold.sum())} sold; '
          f'{len(sets)} rules, {int(ranked.sum())} with >= {MIN_TRADES} trades')
    print('-- top 15 by total at the average price')
    for i, (_, r) in enumerate(order.head(15).iterrows(), 1):
        print(f'{i:2d}. {line(r)}')
    print('-- best set under each vote rule')
    for a in (True, False):
        for k in (1, 2, 3, 4):
            x = order[(order['first_only'] == a) & (order['votes'] == k)]
            if len(x):
                print(f'   {line(x.iloc[0])}')
    print('-- best rule of each size')
    for size in range(1, 10):
        x = order[order['size'] == size]
        if len(x):
            print(f'   {size}: {line(x.iloc[0])}')
    print('-- each debater alone, first choice / both choices')
    for i in range(9):
        for a in (True, False):
            print(f'   {line(t.loc[where[(1 << i, 1, a)]])}')
    print('-- all nine, every named stock:      ' + line(t.loc[where[(511, 1, False)]]))
    print('-- all nine, every first choice:     ' + line(t.loc[where[(511, 1, True)]]))
    force = where[IN_FORCE]
    print(f'-- rule in force (rank {t.loc[force, "rank"]:.0f}): ' + line(t.loc[force]))
    print('-- the winner with one member added or removed')
    s0, k0, a0 = int(sets[win]), int(need[win]), bool(first[win])
    for i in range(9):
        j = where.get((s0 ^ (1 << i), k0, a0))
        if j is not None:
            r = t.loc[j]
            print(f'   {"去掉" if s0 >> i & 1 else "加上"}{NINE[i]}: 笔数 {r["trades"]} 合计平均 {r["total_mid"]:+.1f} '
                  f'前半{r["first_half"]:+.1f} 后半{r["second_half"]:+.1f}')
    top = order.head(50)
    print('-- members of the top 50 rules: ' + ', '.join(
        f'{NINE[i]} {int((top["set"].to_numpy() >> i & 1).sum())}' for i in range(9)))

    # One half chooses, the other half judges.
    a, b = mid[:, :h].sum(1), mid[:, h:].sum(1)
    ta, tb = traded[:, :h].sum(1), traded[:, h:].sum(1)
    oka, okb = ta >= MIN_TRADES * h / n, tb >= MIN_TRADES * (n - h) / n
    ja, jb = int(np.where(oka, a, -np.inf).argmax()), int(np.where(okb, b, -np.inf).argmax())
    halves = {
        'split_after': dates[h - 1],
        'first_half_winner': f'{names(sets[ja])} {t.loc[ja, "choices"]}≥{need[ja]}', 'its_first_half': float(a[ja]),
        'its_second_half': float(b[ja]), 'place_in_second_half': int((b[oka] > b[ja]).sum()) + 1,
        'of': int(oka.sum()), 'median_second_half': float(np.median(b[oka])),
        'second_half_winner': f'{names(sets[jb])} {t.loc[jb, "choices"]}≥{need[jb]}', 'its_second': float(b[jb]),
        'its_first': float(a[jb]), 'place_in_first_half': int((a[okb] > a[jb]).sum()) + 1, 'of_': int(okb.sum()),
        'median_first_half': float(np.median(a[okb])),
        'rank_correlation': float(pd.Series(a[oka & okb]).corr(pd.Series(b[oka & okb]), method='spearman'))}

    # Before every review, follow the rule that leads on the reviews already sold.
    cum, cumt = np.cumsum(mid, 1), np.cumsum(traded, 1)
    follow = []
    for i in range(BURN_IN, n):
        k = i - 2   # the last review sold by the time review i is held
        ok = cumt[:, k] >= MIN_TRADES * (k + 1) / n
        follow.append(int(np.where(ok, cum[:, k], -np.inf).argmax()))
    followed = np.array([mid[j, i] for j, i in zip(follow, range(BURN_IN, n))])
    window = mid[:, BURN_IN:].sum(1)
    forward = {
        'from': dates[BURN_IN], 'reviews': n - BURN_IN, 'followed_total': float(followed.sum()),
        'rules_followed': len(set(follow)), 'last_followed': f'{names(sets[follow[-1]])} {t.loc[follow[-1], "choices"]}≥{need[follow[-1]]}',
        'winner_same_window': float(window[win]), 'in_force_same_window': float(window[force]),
        'all_nine_same_window': float(window[where[(511, 1, False)]]),
        'median_ranked_rule': float(np.median(window[ranked])),
        'place_among_ranked': int((window[ranked] > followed.sum()).sum()) + 1, 'of': int(ranked.sum())}

    # Shuffled results: what the best of 4608 rules scores when no rule knows anything.
    best, own = [], []
    cf, sf = chosen.astype(float), sold.astype(float)
    for _ in range(SHUFFLES // 250):
        o = np.lexsort((rng.random((250, m)), np.broadcast_to(review, (250, m))))
        totals = weight @ ret['mid'][o].T
        best.append(np.where(cf @ sf[o].T >= MIN_TRADES, totals, -np.inf).max(0))
        own.append(totals[win])
    best, own = np.concatenate(best), np.concatenate(own)
    actual = float(t.loc[win, 'total_mid'])
    luck = {'shuffles': len(best), 'winner_total': actual, 'best_of_all_median': float(np.median(best)),
            'best_of_all_p95': float(np.percentile(best, 95)), 'share_best_at_least_winner': float((best >= actual).mean()),
            'winner_rule_shuffled_mean': float(own.mean()), 'share_winner_rule_at_least': float((own >= actual).mean())}
    draws = mid[win][rng.integers(0, n, (10000, n))].sum(1)
    gap = mid[win] - mid[force]
    extra = {'winner_total_90pct_interval': [float(np.percentile(draws, 5)), float(np.percentile(draws, 95))],
             'winner_minus_in_force': float(gap.sum()),
             'winner_minus_in_force_t': float(gap.mean() / (gap.std(ddof=1) / np.sqrt(n))) if gap.std() > 0 else 0.0}
    base = pd.read_csv(ENGINE_OUT / mode / 'days.csv')
    base = base[base['date'].astype(str).isin(dates)]
    extra['market_total_engine'] = float(base['base_all_t1'].sum())

    who = f[['date', 'code', 'name', 'buy_date', 'sell_date', 'filled', 'queued', 'entry', 't1_best', 't1_mid',
             't1_worst', 't1_open', 't1_stuck']].assign(
        voters=[names(int(x) & s0) for x in (m1 if a0 else m2)], share=weight[win])[chosen[win]]
    who.to_csv(OUT / f'trades-best-{mode}.csv', index=False, encoding='utf-8-sig')
    summary = {'mode': mode, 'reviews': n, 'first': dates[0], 'last': dates[-1], 'rules': len(sets),
               'ranked': int(ranked.sum()), 'winner': json.loads(t.loc[win].to_json(force_ascii=False)),
               'in_force': json.loads(t.loc[force].to_json(force_ascii=False)), 'halves': halves,
               'walk_forward': forward, 'luck': luck, 'extra': extra}
    (OUT / f'summary-{mode}.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    for k in ('halves', 'walk_forward', 'luck', 'extra'):
        print(f'-- {k}: ' + json.dumps(summary[k], ensure_ascii=False))


def main():
    cal = json.loads((CACHE / 'extract.json').read_text(encoding='utf-8'))['calendar']
    for mode in (sys.argv[1:] or ['eod']):
        run(mode, cal)


if __name__ == '__main__':
    main()
