"""What every stock a debater named did afterwards, worked out again from the 1-minute bars of D:/data.

    python -X utf8 combo_study/extract_bars.py      # first: fills combo_study/cache from D:/data
    python -X utf8 combo_study/outcomes.py

Uses the bars, the price-limit rules written out below and two tables the engine caches (read only): the ex-rights
events (cache/xdxr.parquet) and the official daily closes (cache/daily_raw.parquet) for the days before the minute
files cover the whole market. The engine's own outcome columns in nominations.csv are read at the end, for the
comparison alone. Writes combo_study/output/:
  outcomes-eod.csv        15:10 review: buy at the next trading day's opening trade, sell on the trading day after it
  outcomes-intraday.csv   10:00 review: buy at the 10:00 bar's close, sell on the next trading day
  check-vs-engine.csv     every named stock whose result here differs from the engine's backtest

Buying, as the project states it: when the buying price is the limit-up price the order queues at the limit and
fills there only if a later bar trades below it; a day without a trade fills nothing. Selling on the next trading day
at its high (best case), its low (worst case) or their midpoint (average case), and at four prices one could really
ask for: the opening trade, the close, the 14:50 bar, and a limit order at the reference price +7% (capped at the
limit-up price) that falls back to 14:50. Returns are on the entry price carried to the sale day's price basis
(ex-rights), in percent. A sale day that traded only at the limit-down price is marked stuck; a sale day without a
trade gives no return.
"""
import bisect
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = HERE / 'cache'
OUT = HERE / 'output'
ENGINE = ROOT / 'skills' / 'stock-roundtable'
N_BARS = 240
BAR_1000 = 29    # the bar stamped 10:00
BAR_1450 = 229   # the bar stamped 14:50
ST_TEN_PERCENT_FROM = '2026-07-06'   # main-board ST band 5% -> 10%, the date the project uses
HALF_TICK = 0.0051
SELL_LIMIT = 7.0   # percent above the sale day's reference price
PRICES = ('best', 'mid', 'worst', 'open', 'close', '1450', 'limit7')


def tick(x):
    """Exchange rounding to 0.01, half up."""
    return float(np.floor(x * 100.0 + 0.5 + 1e-9) / 100.0)


def band(code, name, date):
    """Daily price band in percent, None when the stock has no limit (N/C names: the first days of a listing)."""
    name = re.sub(r'\s', '', name or '')
    if re.match(r'[NC](?!ST)', name, re.I):
        return None
    if code.startswith(('92', '4', '8')):
        return 30.0
    if code.startswith(('30', '688', '689')):
        return 20.0
    if re.search(r'(^|\*)ST', name, re.I) and date < ST_TEN_PERCENT_FROM:
        return 5.0
    return 10.0


class Market:
    """Previous closes, ex-rights reference prices and price limits.

    The first files in D:/data hold only part of the market. From the first complete file on, closes come from the
    minute bars; before it, from the official daily history, which is needed only for the day before the first
    reviews and to tell a new listing from a stock that has traded for years."""

    def __init__(self, cal):
        self.cal = cal
        daily = pd.read_parquet(CACHE / 'daily.parquet', columns=['date', 'code', 'traded_bars', 'close'])
        stocks = daily.groupby('date')['code'].size()
        self.complete_from = str(stocks.index[(stocks >= 0.97 * stocks.median()).to_numpy().argmax()])
        daily = daily[(daily['traded_bars'] > 0) & (daily['date'] >= self.complete_from)]
        raw = pd.read_parquet(ENGINE / 'cache' / 'daily_raw.parquet', columns=['date', 'code', 'close', 'volume'])
        raw = raw[(raw['volume'] > 0) & (raw['date'] < self.complete_from) & raw['code'].str.match(r'^\d{6}$')]
        self.history_from = str(raw['date'].min())
        closes = pd.concat([raw[['date', 'code', 'close']], daily[['date', 'code', 'close']]])
        closes = closes.sort_values(['code', 'date'])
        self.days = {code: (g['date'].tolist(), g['close'].to_numpy(float)) for code, g in closes.groupby('code')}
        x = pd.read_parquet(ENGINE / 'cache' / 'xdxr.parquet')
        x = x[x['category'] == 1].sort_values(['code', 'date'])
        cols = ['fenhong', 'songzhuangu', 'peigu', 'peigujia']
        x[cols] = x[cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        self.events = {code: list(g[['date'] + cols].itertuples(index=False, name=None))
                       for code, g in x.groupby('code')}
        self.new_listing_days, self.without_reference = [], []

    def reference(self, code, date):
        """(reference price, previous close) of a trading day: the last traded close before it, moved by any
        ex-rights event since then. NaN when the stock has no earlier trade in the data."""
        dates, closes = self.days.get(code, ((), ()))
        k = bisect.bisect_left(dates, date)
        if k == 0:
            return np.nan, np.nan
        prev_date, prev = dates[k - 1], float(closes[k - 1])
        ref = prev
        for ex, cash, bonus, rights, rights_price in self.events.get(code, ()):
            if prev_date < ex <= date:
                ref = (ref - cash / 10.0 + rights / 10.0 * rights_price) / (1.0 + bonus / 10.0 + rights / 10.0)
        return ref, prev

    def limits(self, code, name, date):
        """(reference, limit-up, limit-down); the limits are inf/-inf when the stock has none that day."""
        ref, _ = self.reference(code, date)
        pct = band(code, name, date)
        dates = self.days.get(code, ((), ()))[0]
        if dates and dates[0] > self.history_from and bisect.bisect_left(dates, date) < 5:
            pct = None   # listed inside the history: the first five trading days have no limit
            self.new_listing_days.append((date, code))
        if not np.isfinite(ref):
            self.without_reference.append((date, code))
        if pct is None or not np.isfinite(ref):
            return ref, np.inf, -np.inf
        return ref, tick(ref * (1 + pct / 100.0)), tick(ref * (1 - pct / 100.0))


def load_bars():
    """{(date, code): [open, high, low, close, volume] x 240}, NaN where the file has no bar."""
    bars = pd.read_parquet(CACHE / 'bars.parquet')
    grid = {}
    for key, g in bars.groupby(['date', 'code'], sort=False):
        a = np.full((5, N_BARS), np.nan)
        a[:, g['m'].to_numpy()] = g[['open', 'high', 'low', 'close', 'volume']].to_numpy(float).T
        grid[key] = a
    return grid


class Day:
    """One stock's trading day from its first trade on; .traded is False when nothing traded."""

    def __init__(self, a):
        self.traded = False
        if a is None:
            return
        self.open_, self.high_, self.low_, self.close_, vol = a
        self.real = np.isfinite(self.close_)
        trades = self.real & (vol > 0)
        if not trades.any():
            return
        self.traded = True
        self.first = int(trades.argmax())
        self.live = self.real & (np.arange(N_BARS) >= self.first)   # bars from the first trade on
        self.open = float(self.open_[self.first])
        self.high = float(self.high_[self.live].max())
        self.low = float(self.low_[self.live].min())
        self.close = float(self.close_[self.real][-1])

    def last_close(self, upto):
        """Close of the last bar at or before position `upto`, counted from the first trade (else the opening trade)."""
        seen = np.nonzero(self.live[:upto + 1])[0]
        return float(self.close_[seen[-1]]) if len(seen) else self.open

    def trades_below(self, price, start):
        """Does any bar from position `start` on trade below `price`?"""
        live = self.live & (np.arange(N_BARS) >= start)
        return bool((self.low_[live] < price - HALF_TICK).any())


def sale(mk, grid, i_sell, code, name, entry, close_buy):
    """Selling on trading day number i_sell what was bought at `entry` on a day that closed at close_buy."""
    out = {'sell_date': '', 'sale_traded': False}
    if i_sell >= len(mk.cal):
        return out
    s = mk.cal[i_sell]
    out['sell_date'] = s
    day = Day(grid.get((s, code)))
    if not day.traded:
        return out
    ref, limit_up, limit_down = mk.limits(code, name, s)
    base = entry * ref / close_buy   # the entry on the sale day's price basis
    p1450 = day.last_close(BAR_1450)
    target = min(tick(ref * (1 + SELL_LIMIT / 100.0)), limit_up)
    before_1450 = day.high_[day.live & (np.arange(N_BARS) <= BAR_1450)]
    if day.open >= target - 1e-9:
        limit7 = day.open
    elif len(before_1450) and before_1450.max() >= target - 1e-9:
        limit7 = target
    else:
        limit7 = p1450
    price = {'best': day.high, 'mid': (day.high + day.low) / 2.0, 'worst': day.low, 'open': day.open,
             'close': day.close, '1450': p1450, 'limit7': limit7}
    out.update(sale_traded=True, ref_sell=ref, s_open=day.open, s_high=day.high, s_low=day.low, s_close=day.close,
               t1_stuck=bool(day.high <= day.low and day.low <= limit_down + HALF_TICK),
               **{f't1_{k}': (price[k] / base - 1.0) * 100.0 for k in PRICES})
    return out


def buy(mk, grid, mode, date, code, name):
    """One named stock: the buy (or why not) and the sale after it."""
    i = mk.cal.index(date)
    out = {'date': date, 'code': code, 'name': name, 'buy_date': '', 'traded_buy_day': False, 'filled': False,
           'queued': False, 'entry': np.nan, 'ref_buy': np.nan, 'limit_up': np.nan, 'buy_close': np.nan,
           'sell_date': '', 'sale_traded': False, 't1_stuck': False}
    i_buy = i + 1 if mode == 'eod' else i
    if i_buy >= len(mk.cal):
        return out
    b = mk.cal[i_buy]
    out['buy_date'] = b
    day = Day(grid.get((b, code)))
    if not day.traded or (mode == 'intraday' and day.first > BAR_1000):
        return out
    ref, limit_up, _ = mk.limits(code, name, b)
    if mode == 'eod':
        price, start = day.open, day.first        # the opening trade
    else:
        price, start = day.last_close(BAR_1000), BAR_1000 + 1   # the 10:00 close
    queued = bool(price >= limit_up - HALF_TICK)
    filled = (not queued) or day.trades_below(limit_up, start)
    entry = limit_up if queued else price
    out.update(traded_buy_day=True, queued=queued, filled=filled, ref_buy=ref, buy_close=day.close,
               limit_up=limit_up if np.isfinite(limit_up) else np.nan)
    if not filled:
        out['sell_date'] = mk.cal[i_buy + 1] if i_buy + 1 < len(mk.cal) else ''
        return out
    out['entry'] = entry
    out['buy_chg'] = (entry / ref - 1.0) * 100.0
    out.update(sale(mk, grid, i_buy + 1, code, name, entry, day.close))
    return out


def compare(mine, noms, mode):
    """Rows where the engine's backtest says something else about the same named stock."""
    theirs = noms.drop_duplicates(['date', 'code']).set_index(['date', 'code'])
    rows = []
    for r in mine.to_dict('records'):
        e = theirs.loc[(r['date'], r['code'])]
        diff = []
        e_filled, e_queued = bool(e['filled']), bool(e['queued'])
        if e_filled != r['filled']:
            diff.append('filled')
        if e_queued != r['queued']:
            diff.append('queued')
        if r['filled'] and e_filled and abs(float(e['entry']) - r['entry']) > HALF_TICK:
            diff.append('entry')
        for k in ('best', 'mid', 'worst'):
            a, b = r.get(f't1_{k}', np.nan), float(e[f't1_{k}'])
            a = np.nan if a is None else a
            if np.isfinite(a) != np.isfinite(b) or (np.isfinite(a) and abs(a - b) > 0.01):
                diff.append(f't1_{k}')
        if bool(e['t1_stuck']) != bool(r['t1_stuck']):
            diff.append('stuck')
        if diff:
            rows.append({'mode': mode, 'date': r['date'], 'code': r['code'], 'name': r['name'],
                         'differs': ' '.join(diff), 'buy_date': r['buy_date'], 'sell_date': r['sell_date'],
                         'filled': r['filled'], 'engine_filled': e_filled, 'queued': r['queued'],
                         'engine_queued': e_queued, 'entry': r['entry'], 'engine_entry': e['entry'],
                         **{f'{side}t1_{k}': (r.get(f't1_{k}', np.nan) if side == '' else e[f't1_{k}'])
                            for k in ('best', 'mid', 'worst') for side in ('', 'engine_')}})
    return rows


def main():
    cal = json.loads((CACHE / 'extract.json').read_text(encoding='utf-8'))['calendar']
    mk = Market(cal)
    grid = load_bars()
    OUT.mkdir(parents=True, exist_ok=True)
    differences, summary = [], {'complete_from': mk.complete_from, 'official_history_from': mk.history_from}
    for mode in ('eod', 'intraday'):
        noms = pd.read_csv(ENGINE / 'output' / mode / 'nominations.csv', dtype={'code': str})
        named = noms.drop_duplicates(['date', 'code'])[['date', 'code', 'name']]
        mine = pd.DataFrame([buy(mk, grid, mode, d, c, n) for d, c, n in named.itertuples(index=False)])
        mine.to_csv(OUT / f'outcomes-{mode}.csv', index=False, encoding='utf-8-sig')
        rows = compare(mine, noms, mode)
        differences += rows
        kinds = pd.Series([k for r in rows for k in r['differs'].split()], dtype=object).value_counts()
        summary[mode] = {
            'named': int(len(mine)), 'first': mine['date'].min(), 'last': mine['date'].max(),
            'filled': int(mine['filled'].sum()), 'queued': int(mine['queued'].sum()),
            'queued_filled': int((mine['queued'] & mine['filled']).sum()),
            'no_trade_on_buy_day': int(((mine['buy_date'] != '') & ~mine['traded_buy_day']).sum()),
            'sold': int(mine['t1_mid'].notna().sum()), 'stuck': int(mine['t1_stuck'].sum()),
            'no_trade_on_sale_day': int((mine['filled'] & (mine['sell_date'] != '') & ~mine['sale_traded']).sum()),
            'differs_from_engine': len(rows), 'differences': {k: int(n) for k, n in kinds.items()}}
        print(f'{mode}: ' + ', '.join(f'{k} {v}' for k, v in summary[mode].items()))
    pd.DataFrame(differences).to_csv(OUT / 'check-vs-engine.csv', index=False, encoding='utf-8-sig')
    summary['new_listing_days'] = sorted(set(mk.new_listing_days))
    summary['without_reference'] = sorted(set(mk.without_reference))
    (OUT / 'outcomes.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'minute files complete from {mk.complete_from}; no limit as a new listing: {summary["new_listing_days"]}; '
          f'no reference price: {summary["without_reference"]}; written {OUT}')


if __name__ == '__main__':
    main()
