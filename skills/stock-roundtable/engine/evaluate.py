"""Outcomes, computed only after every decision of the day has been recorded.

The buying day (entry to that day's close and the path in between) and, in next_day, the A-share T+1 sale on the
trading day after the buy at that day's high (best case), low (worst case) and midpoint (average case).
  intraday: buy at the close of the cutoff bar (e.g. the 10:00 bar). If the stock sits at the limit-up price
            then, the order queues at the limit and fills only if a later bar trades below it (the board opens).
  eod:      buy at the next trading day's first-bar open. An open at the limit-up price queues the same way.
            A stock with no bars on the next day is not filled.
Metrics (filled orders only): close return, highest and lowest point after entry, touched the limit,
sealed at the close (y3 in the project's terms), rose at least 5% board-scaled after entry (y2), close > entry.
"""
import numpy as np
import pandas as pd

from . import minutes, profile

RATE_COLUMNS = ('hit_lu', 'sealed', 'y2', 'win')
VALUE_COLUMNS = ('entry', 'ret_close', 'ret_close_s', 'max_up', 'max_up_s', 'max_dn')


def _metrics(entry, hi, lo, close, lu, pct, filled, queued):
    with np.errstate(invalid='ignore', divide='ignore'):
        scale = 10.0 / pct
        ret_close = (close / entry - 1.0) * 100.0
        max_up = np.where(queued, 0.0, (np.maximum(hi, entry) / entry - 1.0) * 100.0)
        max_dn = (np.minimum(lo, entry) / entry - 1.0) * 100.0
        out = pd.DataFrame({
            'filled': filled, 'queued': queued, 'entry': entry,
            'ret_close': ret_close, 'ret_close_s': ret_close * scale,
            'max_up': max_up, 'max_up_s': max_up * scale, 'max_dn': max_dn,
            'hit_lu': (queued | profile.at_or_above(hi, lu)).astype(float),
            'sealed': profile.at_or_above(close, lu).astype(float),
        })
    out['y2'] = (out['max_up_s'] >= 5.0).astype(float)
    out['win'] = (out['ret_close'] > 0).astype(float)
    for col in VALUE_COLUMNS + RATE_COLUMNS:
        out.loc[~out['filled'], col] = np.nan
    return out


def same_day(day, rows, k, limit_up, pct):
    """Buy at the cutoff bar's close; rows index the day's arrays."""
    rows = np.asarray(rows, int)
    px = day.close[rows, k - 1]
    queued = profile.at_or_above(px, limit_up)
    opened = (~profile.at_or_above(day.low[rows, k:], limit_up[:, None])).any(axis=1)
    filled = ~queued | opened
    entry = np.where(queued, limit_up, px)
    return _metrics(entry, day.high[rows, k:].max(axis=1), day.low[rows, k:].min(axis=1), day.close[rows, -1],
                    limit_up, pct, filled, queued)


def next_open(next_day, rows_next, limit_up, pct):
    """Buy at the next day's opening trade; rows_next is -1 where the stock has no bars that day. A stock without a
    trade that day (TDX writes flat zero-volume bars for a suspended stock) is not filled; bars before the first trade
    carry no price, so the entry is the first trade's bar and the day's range starts there."""
    rows_next = np.asarray(rows_next, int)
    r = np.where(rows_next >= 0, rows_next, 0)
    traded = minutes.traded_bars(next_day)[r]
    trading = (rows_next >= 0) & traded.any(axis=1)
    first = np.where(trading, traded.argmax(axis=1), 0)
    started = np.arange(traded.shape[1])[None, :] >= first[:, None]
    px = np.where(trading, next_day.open[r, first], np.nan)
    low = np.where(started, next_day.low[r, :], np.inf)
    high = np.where(started, next_day.high[r, :], -np.inf)
    queued = trading & profile.at_or_above(px, limit_up)
    opened = (started & ~profile.at_or_above(next_day.low[r, :], limit_up[:, None])).any(axis=1)
    filled = trading & (~queued | opened)
    entry = np.where(queued, limit_up, px)
    return _metrics(entry, high.max(axis=1), low.min(axis=1), next_day.close[r, -1], limit_up, pct, filled, queued)


def outcomes_for(ctx, panel, day_or_next, mode):
    """Outcomes for every eligible stock of the context, indexed like ctx.stocks."""
    s = ctx.stocks[ctx.stocks['eligible']]
    if mode == 'intraday':
        out = same_day(day_or_next, s['row'].to_numpy(), ctx.k, s['limit_up'].to_numpy(float), s['pct'].to_numpy(float))
    else:
        t1 = ctx.t + 1
        j = s['j'].to_numpy()
        pos = np.searchsorted(day_or_next.panel_j, j)
        pos = np.clip(pos, 0, len(day_or_next.panel_j) - 1)
        found = day_or_next.panel_j[pos] == j
        if day_or_next.real is not None:
            found &= day_or_next.real[pos].any(axis=1)
        rows_next = np.where(found, pos, -1)
        out = next_open(day_or_next, rows_next, panel.limit_up[t1, j], panel.pct[t1, j])
    out.index = s.index
    return out


def next_day(panel, t_buy, j, entry):
    """The buy day's change at the entry and at its close, and the trading day after the buy (A-share T+1): that
    day's high, low and their midpoint as changes from its reference price (涨幅) and as returns on the entry (收益,
    adjusted for ex-rights events between the two days). NaN where the sale day is not in the data yet or the stock
    does not trade that day. t_buy is one panel row for all stocks or one per stock.
    t1_stuck: the sale day traded at one price, the limit-down (一字跌停), so the stock could not really be sold; the
    returns still use that price, as study_t1.py does, and reports mark it."""
    j = np.asarray(j, int)
    t_buy = np.broadcast_to(np.asarray(t_buy, int), j.shape)
    entry = np.asarray(entry, float)
    s = t_buy + 1
    ok = s < len(panel.dates)
    s_ = np.where(ok, s, 0)
    ok = ok & panel.has[s_, j]
    with np.errstate(invalid='ignore', divide='ignore'):
        ref_b, close_b = panel.ref[t_buy, j], panel.close[t_buy, j]
        ref, hi, lo = panel.ref[s_, j], panel.high[s_, j], panel.low[s_, j]
        mid = (hi + lo) / 2
        base = entry * ref / close_b
        return pd.DataFrame({
            'buy_chg': (entry / ref_b - 1) * 100, 'close_chg': (close_b / ref_b - 1) * 100,
            'sell_date': [panel.dates[x] if k else None for x, k in zip(s_, ok)],
            'next_high_chg': np.where(ok, (hi / ref - 1) * 100, np.nan),
            'next_low_chg': np.where(ok, (lo / ref - 1) * 100, np.nan),
            'next_mid_chg': np.where(ok, (mid / ref - 1) * 100, np.nan),
            't1_best': np.where(ok, (hi / base - 1) * 100, np.nan),
            't1_mid': np.where(ok, (mid / base - 1) * 100, np.nan),
            't1_worst': np.where(ok, (lo / base - 1) * 100, np.nan),
            't1_stuck': ok & (hi <= lo) & profile.at_or_below(lo, panel.limit_down[s_, j])})


def baseline(ctx, db, out):
    """Comparison for a record: every eligible stock and every chosen stock (buy-day close; T+1 midpoint if known)."""
    def mean(frame, col):
        f = frame[frame['filled'] == True]  # noqa: E712
        return float(f[col].mean()) if col in f and f[col].notna().any() else np.nan

    pool = out.loc[[i for i in db.choices if i in out.index]]
    return {'all_n': int((out['filled'] == True).sum()), 'all_ret': mean(out, 'ret_close'),  # noqa: E712
            'all_t1': mean(out, 't1_mid'), 'pool_n': int((pool['filled'] == True).sum()),  # noqa: E712
            'pool_ret': mean(pool, 'ret_close'), 'pool_t1': mean(pool, 't1_mid')}


def summarize(frame):
    """Per-trade statistics over filled orders."""
    f = frame[frame['filled'] == True]  # noqa: E712
    n_all = len(frame)
    if not len(f):
        return {'n': n_all, 'filled': 0}
    return {
        'n': n_all, 'filled': int(len(f)), 'fill_rate': len(f) / n_all if n_all else np.nan,
        'ret_close': f['ret_close'].mean(), 'ret_close_med': f['ret_close'].median(),
        'ret_close_s': f['ret_close_s'].mean(), 'max_up': f['max_up'].mean(), 'max_dn': f['max_dn'].mean(),
        'win': f['win'].mean(), 'y2': f['y2'].mean(), 'hit_lu': f['hit_lu'].mean(), 'sealed': f['sealed'].mean(),
    }
