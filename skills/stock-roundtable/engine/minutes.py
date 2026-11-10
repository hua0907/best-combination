"""One trading day of 1-minute bars from D:/data/<date>.csv as [stock, minute] arrays.

Bar m covers the minute ending at TIMES[m] (09:31 ... 15:00, 240 bars). A cutoff of k bars
means only bars [0, k) are visible: k=30 is 10:00, k=240 is the close.
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from . import config, profile


def _times():
    out = []
    for hour, start, end in ((9, 31, 59), (10, 0, 59), (11, 0, 30), (13, 1, 59), (14, 0, 59), (15, 0, 0)):
        for minute in range(start, end + 1):
            out.append(f'{hour:02d}:{minute:02d}')
    return out


TIMES = _times()
TIME_INDEX = {t: i for i, t in enumerate(TIMES)}


class Day:
    def __init__(self, date, panel_j, o, h, l, c, v, a, real):
        self.date = date
        self.panel_j = panel_j      # panel code index of each row
        self.open, self.high, self.low, self.close = o, h, l, c
        self.vol, self.amt = v, a   # vol in lots (100 shares), amt in yuan
        self.real = real            # [row, minute] True where the CSV has a bar (False = filled gap)

    @property
    def size(self):
        return len(self.panel_j)


def load_day(date, panel):
    path = config.DATA_DIR / f'{date}.csv'
    table = pacsv.read_csv(path, convert_options=pacsv.ConvertOptions(
        include_columns=['code', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'amount'],
        column_types={'code': pa.string(), 'datetime': pa.string()}))
    df = table.to_pandas()
    df = df[df['datetime'].str.slice(0, 10) == date]
    df['m'] = df['datetime'].str.slice(11, 16).map(TIME_INDEX)
    df['j'] = df['code'].map(panel.code_index)
    df = df[df['m'].notna() & df['j'].notna() & (df['close'] > 0)]
    df['m'] = df['m'].astype(int)
    df['j'] = df['j'].astype(int)
    panel_j = np.sort(df['j'].unique())
    row = np.searchsorted(panel_j, df['j'].to_numpy())
    m = df['m'].to_numpy()
    n = len(panel_j)
    shape = (n, len(TIMES))
    c = np.full(shape, np.nan)
    o, h, l = c.copy(), c.copy(), c.copy()
    v = np.zeros(shape)
    a = np.zeros(shape)
    c[row, m] = df['close'].to_numpy(float)
    o[row, m] = df['open'].to_numpy(float)
    h[row, m] = df['high'].to_numpy(float)
    l[row, m] = df['low'].to_numpy(float)
    v[row, m] = df['volume'].to_numpy(float)
    a[row, m] = df['amount'].to_numpy(float)
    # Missing minutes: carry the last close; before the first bar use the first bar's open.
    filled = pd.DataFrame(c).ffill(axis=1)
    first_open = pd.DataFrame(np.where(np.isfinite(o), o, np.nan)).bfill(axis=1).iloc[:, 0].to_numpy()
    filled = filled.to_numpy()
    lead = np.isnan(filled)
    filled[lead] = np.broadcast_to(first_open[:, None], shape)[lead]
    gap = np.isnan(c)
    o[gap] = filled[gap]
    h[gap] = filled[gap]
    l[gap] = filled[gap]
    o = np.where(o > 0, o, filled)
    h = np.where(h > 0, h, filled)
    l = np.where(l > 0, l, filled)
    return Day(date, panel_j, o, h, l, filled, v, a, ~gap)


def unfinished(day, now=None):
    """Why `day` is not a finished trading day yet ('' when it is). Finished: the closing prices are final, i.e. the
    clock has passed config.CLOSE_REVIEW on the day itself and nearly every stock in the file has its 15:00 bar (the
    bar that holds the closing call auction)."""
    early = config.too_early(day.date, now)
    if early:
        return early
    share = float(day.real[:, -1].mean()) if day.size else 0.0
    if share < config.CLOSE_BAR_SHARE:
        seen = day.real.any(axis=0).nonzero()[0]
        upto = f'，分钟线只到 {TIMES[seen.max()]}' if len(seen) else ''
        return f'{day.date} 只有 {share:.1%} 的股票有 15:00 收盘那根分钟线{upto}'
    return ''


def traded_bars(day, k=None):
    """[row, minute] True where the minute had a trade: the CSV has the bar and its volume is positive. TDX also writes
    bars without a trade (240 flat zero-volume bars for a suspended stock, the minutes before an illiquid stock's first
    trade); they carry no tradable price."""
    k = day.vol.shape[1] if k is None else k
    real = day.real[:, :k] if day.real is not None else np.ones(day.vol[:, :k].shape, bool)
    return real & (day.vol[:, :k] > 0)


def cut_features(day, k, ref, pct, limit_up, limit_down):
    """Features visible at cutoff k (bars [0, k)). ref/pct/limits are pre-open facts for each row.

    Bars before a stock's first trade carry no traded price: the open is the first trade's bar and the high/low start
    there. trades_seen counts the minutes with a trade (0 = nothing traded yet: the stock is not visible)."""
    c = day.close[:, :k]
    h = day.high[:, :k]
    lo = day.low[:, :k]
    vol = day.vol[:, :k] * 100.0
    amt = day.amt[:, :k]
    scale = 10.0 / pct
    price = c[:, -1]
    traded = traded_bars(day, k)
    first_trade = np.where(traded.any(axis=1), traded.argmax(axis=1), 0)
    started = np.arange(k)[None, :] >= first_trade[:, None]   # from the first trade on (all bars when none)
    open_px = day.open[np.arange(len(price)), first_trade]
    cum_amt = amt.sum(axis=1)
    cum_vol = vol.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        vwap = np.where(cum_vol > 0, cum_amt / cum_vol, price)
        running_vwap = np.cumsum(amt, axis=1) / np.cumsum(vol, axis=1)
    running_vwap = np.where(np.isfinite(running_vwap), running_vwap, c)
    above = (c >= running_vwap * (1 - 1e-9)).mean(axis=1)
    hi = np.where(started, h, -np.inf).max(axis=1)
    low = np.where(started, lo, np.inf).min(axis=1)
    at_lu = profile.at_or_above(c, limit_up[:, None])
    touch_lu = profile.at_or_above(h, limit_up[:, None])
    touched = touch_lu.any(axis=1)
    first_touch = np.where(touched, touch_lu.argmax(axis=1), -1)
    sealed_now = at_lu[:, -1]
    opens = (at_lu[:, :-1] & ~at_lu[:, 1:]).sum(axis=1) if k > 1 else np.zeros(len(price), int)
    at_ld = profile.at_or_below(c, limit_down[:, None])
    touch_ld = profile.at_or_below(lo, limit_down[:, None])
    if k > 5:
        speed = np.nanmax(c[:, 5:] / c[:, :-5] - 1.0, axis=1) * 100.0 * scale
    else:
        speed = (price / open_px - 1.0) * 100.0 * scale
    low_at = np.where(started, lo, np.inf).argmin(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        ret = (price / ref - 1.0) * 100.0
        out = pd.DataFrame({
            'row': np.arange(len(price)),
            'j': day.panel_j,
            'bars_seen': day.real[:, :k].sum(axis=1),
            'trades_seen': traded.sum(axis=1),
            'price': price,
            'open_px': open_px,
            'ref': ref,
            'pct': pct,
            'limit_up': limit_up,
            'limit_down': limit_down,
            'ret': ret,
            'ret_s': ret * scale,
            'open_s': (open_px / ref - 1.0) * 100.0 * scale,
            'high_s': (hi / ref - 1.0) * 100.0 * scale,
            'low_s': (low / ref - 1.0) * 100.0 * scale,
            'rebound_s': (price / low - 1.0) * 100.0 * scale,
            'drop_from_open_s': (low / open_px - 1.0) * 100.0 * scale,
            'high_px': hi,
            'low_px': low,
            'one_word': profile.at_or_above(low, limit_up),
            'vwap': vwap,
            'vs_vwap': (price / vwap - 1.0) * 100.0,
            'above_vwap': above,
            'amount': cum_amt,
            'volume': cum_vol,
            'first_bar_amt': day.amt[:, 0],
            'lu_touched': touched,
            'lu_first': first_touch,
            'lu_now': sealed_now,
            'lu_opens': opens,
            'lu_reseal': touched & (opens >= 1) & sealed_now,
            'ld_now': at_ld[:, -1],
            'ld_touched': touch_ld.any(axis=1),
            'speed5_s': speed,
            'low_minute': low_at,
        })
    return out


def day_summary(day, ref, pct, limit_up, limit_down, cuts=(30,)):
    """Whole-day facts of a finished day, stored for LATER days only.

    amt_k<k> is the amount traded in the first k bars, the baseline for a later day's same-time amount ratio.
    """
    full = cut_features(day, 240, ref, pct, limit_up, limit_down)
    for k in cuts:
        full[f'amt_k{k}'] = day.amt[:, :k].sum(axis=1)
    full['close'] = day.close[:, -1]
    full['high'] = full['high_px']   # from the first trade on (cut_features)
    full['low'] = full['low_px']
    return full[full['trades_seen'] > 0].set_index('j')   # a stock without a trade all day (suspended) has no day
