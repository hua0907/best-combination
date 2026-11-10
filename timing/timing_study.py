"""买卖点研究（只读，不改选股、不改冻结引擎）。

对最优组合已结算的每一笔信号股，用 D:/data 的 1 分钟线比较：
  买入日（T+1）各种买法 —— 开盘买（现行）、定时买、开盘价下方挂单、昨收下方挂单、跌停价挂单、回踩均价线、
                            先跌后拉确认、突破开盘高点、半仓开盘 + 半仓低吸……
  卖出日（T+2）各种卖法 —— 开盘卖（现行）、定时卖、挂高卖、高开直接卖、移动止盈、跌破均价线卖、止损、
                            “冲到一定涨幅后跟踪高点回落就卖”（追最高点）、半仓挂高 + 半仓跟踪……
买法用同一卖法（均价/开盘/收盘三种基准）比，卖法用同一买法（现行开盘价买）比，最后把最好的买法和卖法合起来看。
每条卖法还算它离当天最高价有多远（距最高%、区间捕获率）。
每条规则都给前半段 / 后半段分别的结果，差得远的不可信。

    python -X utf8 timing\\timing_study.py            # 首次约 4 分钟（分钟线按日缓存到 timing\\output\\cache，之后约 1 分钟）

输出 timing\\output\\：buy_rules.csv、sell_rules.csv、combo_rules.csv、per_trade.csv、structure.csv、report.md
"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

os.environ.pop('STOCK_ROUNDTABLE_ROOT', None)
ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / 'skills' / 'stock-roundtable'
sys.path.insert(0, str(ENGINE_DIR))
from engine import config as ecfg, profile, store  # noqa: E402
from engine.minutes import TIME_INDEX, TIMES  # noqa: E402

OUT = ROOT / 'timing' / 'output'
CACHE = OUT / 'cache'
TRADES = ENGINE_DIR / 'output' / 'optimal-combo' / 'trades.csv'
LAST = 239


def bar(t):
    """'HH:MM' -> index of the 1-minute bar ending at t (09:31 -> 0, 10:00 -> 29, 15:00 -> 239)."""
    return ecfg.bars_until(t) - 1


def tick(x):
    return float(profile.round_tick(x))


def above(px, limit):
    return bool(profile.at_or_above(px, limit))


def below(px, limit):
    return bool(profile.at_or_below(px, limit))


class Bars:
    """One stock's 240 bars of one day, gaps filled the way the engine fills them."""

    def __init__(self, code, o, h, l, c, v, a, real):
        self.code = code
        self.o, self.h, self.l, self.c, self.v, self.a, self.real = o, h, l, c, v, a, real
        self.traded = real & (v > 0)
        self.f = int(np.argmax(self.traded))          # first bar with a trade
        vol = v * 100.0
        with np.errstate(invalid='ignore', divide='ignore'):
            vw = np.cumsum(a) / np.cumsum(vol)
        self.vwap = np.where(np.isfinite(vw), vw, c)

    @property
    def open(self):
        return float(self.o[self.f])

    def px(self, k):
        """Price of a market order at bar k: the day's open at the first bar, else that bar's close."""
        k = max(k, self.f)
        return self.open if k == self.f else float(self.c[k])

    def first(self, m):
        """The first price seen in bar m (the day's open in the first bar)."""
        return self.open if m == self.f else float(self.o[m])


def load_minutes(date, codes):
    """{code: Bars} for the codes that traded on `date` (read from D:/data, cached per date)."""
    key = CACHE / f'{date}-{abs(hash(tuple(sorted(codes)))) % 10**8}.pkl'
    if key.exists():
        return pickle.loads(key.read_bytes())
    path = ecfg.DATA_DIR / f'{date}.csv'
    table = pacsv.read_csv(path, convert_options=pacsv.ConvertOptions(
        include_columns=['code', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'amount'],
        column_types={'code': pa.string(), 'datetime': pa.string()}))
    table = table.filter(pc.is_in(table['code'], value_set=pa.array(sorted(codes), pa.string())))
    df = table.to_pandas()
    df = df[df['datetime'].str.slice(0, 10) == date]
    df['m'] = df['datetime'].str.slice(11, 16).map(TIME_INDEX)
    df = df[df['m'].notna() & (df['close'] > 0)]
    out = {}
    for code, g in df.groupby('code'):
        m = g['m'].astype(int).to_numpy()
        c = np.full(240, np.nan)
        o, h, l = c.copy(), c.copy(), c.copy()
        v, a = np.zeros(240), np.zeros(240)
        c[m], o[m], h[m], l[m] = g['close'].to_numpy(float), g['open'].to_numpy(float), g['high'].to_numpy(float), g['low'].to_numpy(float)
        v[m], a[m] = g['volume'].to_numpy(float), g['amount'].to_numpy(float)
        real = np.isfinite(c)
        filled = pd.Series(c).ffill().to_numpy()
        first_open = o[np.isfinite(o)][0]
        filled = np.where(np.isnan(filled), first_open, filled)
        gap = ~real
        for arr in (o, h, l):
            arr[gap] = filled[gap]
        o, h, l = (np.where(x > 0, x, filled) for x in (o, h, l))
        b = Bars(code, o, h, l, filled, v, a, real)
        if b.traded.any():
            out[code] = b
    CACHE.mkdir(parents=True, exist_ok=True)
    key.write_bytes(pickle.dumps(out))
    return out


# ----------------------------------------------------------------------------------------------- order simulation
def market_buy(b, k, lu):
    """Market buy at bar k. At the limit-up price the order queues and fills (at the limit) in the first later bar
    that trades below it; never opening = not filled. -> (price, bar) or (nan, -1)."""
    k = max(k, b.f)
    px = b.px(k)
    if above(px, lu):
        for m in range(k, 240):
            if b.traded[m] and not above(b.l[m], lu):
                return float(lu), m
        return np.nan, -1
    return px, k


def limit_buy(b, price, k0, deadline, lu, ld, fallback):
    """Resting buy order at `price`, placed at bar k0 (after seeing that bar's first price). Fills in the first bar
    whose low reaches it (at the bar's open if that is already lower). Not filled by `deadline`: market buy at the
    deadline bar when fallback, else nothing."""
    price = tick(max(price, ld) if np.isfinite(ld) else price)
    start = max(k0, b.f)
    for m in range(start, min(deadline, LAST) + 1):
        if not b.traded[m]:
            continue
        first = b.open if m == b.f else (b.c[m - 1] if m == start else b.o[m])
        if first <= price + 1e-9:
            return float(first), m
        if b.l[m] <= price + 1e-9:
            return price, m
    return market_buy(b, deadline, lu) if fallback else (np.nan, -1)


def vwap_pullback(b, deadline, lu, fallback):
    """Buy the first time the price comes back down to the running VWAP."""
    for m in range(b.f + 1, min(deadline, LAST) + 1):
        if not b.traded[m]:
            continue
        level = b.vwap[m - 1]
        if b.o[m] <= level + 1e-9:
            return float(b.o[m]), m
        if b.l[m] <= level + 1e-9:
            return float(level), m
    return market_buy(b, deadline, lu) if fallback else (np.nan, -1)


def dip_rebound(b, dip, bounce, deadline, lu, fallback):
    """After a dip of at least `dip`% below the open, buy when the price has bounced `bounce`% off the running low."""
    o, low = b.open, b.open
    for m in range(b.f, min(deadline, LAST) + 1):
        low = min(low, float(b.l[m]))
        if low <= o * (1 - dip / 100) and b.c[m] >= low * (1 + bounce / 100):
            return market_buy(b, m, lu)
    return market_buy(b, deadline, lu) if fallback else (np.nan, -1)


def breakout(b, n, deadline, lu, fallback):
    """Buy when the price exceeds the high of the first n minutes."""
    level = float(np.max(b.h[b.f:b.f + n]))
    for m in range(b.f + n, min(deadline, LAST) + 1):
        if b.traded[m] and b.h[m] > level + 1e-9:
            px = max(level, float(b.o[m]))
            if above(px, lu):
                return market_buy(b, m, lu)
            return px, m
    return market_buy(b, deadline, lu) if fallback else (np.nan, -1)


def reclaim(b, level, deadline, lu, fallback):
    """Opened below `level` (e.g. yesterday's close): buy when the price gets back up to it; opened above: buy the open."""
    if b.open >= level:
        return market_buy(b, b.f, lu)
    for m in range(b.f + 1, min(deadline, LAST) + 1):
        if b.traded[m] and b.h[m] >= level - 1e-9:
            return max(level, float(b.o[m])), m
    return market_buy(b, deadline, lu) if fallback else (np.nan, -1)


def market_sell(b, k):
    k = max(k, b.f)
    return b.px(k), k


def limit_sell(b, price, k0, deadline, lu):
    """Resting sell order at `price` from bar k0; -> (price, bar) or None when not filled by `deadline`."""
    price = tick(min(price, lu) if np.isfinite(lu) else price)
    start = max(k0, b.f)
    for m in range(start, min(deadline, LAST) + 1):
        if not b.traded[m]:
            continue
        first = b.open if m == b.f else (b.c[m - 1] if m == start else b.o[m])
        if first >= price - 1e-9:
            return float(first), m
        if b.h[m] >= price - 1e-9:
            return price, m
    return None


def trailing_sell(b, z, k0, floor=None):
    """From bar k0 track the running high; sell at the first close at least z% below it (or below `floor`)."""
    start = max(k0, b.f)
    hi = b.px(start)
    for m in range(start, 240):
        hi = max(hi, float(b.h[m]))
        if b.c[m] <= hi * (1 - z / 100) + 1e-9 or (floor is not None and b.c[m] <= floor + 1e-9):
            return float(b.c[m]), m
    return float(b.c[LAST]), LAST


def vwap_sell(b, skip, k0=None):
    """Sell at the first close below the running VWAP after the first `skip` minutes; else at the close."""
    start = max(b.f + skip, k0 or 0)
    for m in range(start, 240):
        if b.traded[m] and b.c[m] < b.vwap[m] * (1 - 1e-9):
            return float(b.c[m]), m
    return float(b.c[LAST]), LAST


def stop_sell(b, level, k0=0):
    for m in range(max(k0, b.f), 240):
        if b.c[m] <= level + 1e-9:
            return float(b.c[m]), m
    return float(b.c[LAST]), LAST


def first_of(*cands):
    """Of several (price, bar) outcomes (None = never), the one that happens first."""
    c = [x for x in cands if x]
    return min(c, key=lambda x: x[1])


def _arm_bar(b, arm):
    """First bar in which the price reaches `arm` (-1 = never)."""
    for m in range(b.f, 240):
        if b.first(m) >= arm - 1e-9 or b.h[m] >= arm - 1e-9:
            return m
    return -1


def armed_trailing(b, arm, z, fallback):
    """追最高点：价格第一次到 `arm` 后开始跟踪当天最高价，收于最高价 z% 以下就卖；到收盘没触发按收盘价；没到过 arm 用 fallback。
    到 arm 的那一分钟里如果已经回落 z%（冲高回落），就按那一分钟的收盘卖。"""
    k = _arm_bar(b, arm)
    if k < 0:
        return fallback
    hi = max(float(b.h[k]), arm)
    for m in range(k, 240):
        hi = max(hi, float(b.h[m]))
        if b.c[m] <= hi * (1 - z / 100) + 1e-9:
            return float(b.c[m]), m
    return float(b.c[LAST]), LAST


def armed_swing(b, arm, n, fallback):
    """到 `arm` 后，收盘价跌破之前 n 分钟的最低价（分时拐点）就卖。"""
    k = _arm_bar(b, arm)
    if k < 0:
        return fallback
    for m in range(k + 1, 240):
        if b.c[m] < float(np.min(b.l[max(k, m - n):m])) - 1e-9:
            return float(b.c[m]), m
    return float(b.c[LAST]), LAST


def armed_vwap(b, arm, fallback):
    """到 `arm` 后，收盘价跌破均价线就卖。"""
    k = _arm_bar(b, arm)
    if k < 0:
        return fallback
    for m in range(k, 240):
        if b.c[m] < b.vwap[m] * (1 - 1e-9):
            return float(b.c[m]), m
    return float(b.c[LAST]), LAST


# ----------------------------------------------------------------------------------------------- rule catalogues
TIMES_BUY = ['09:35', '09:40', '09:45', '10:00', '10:30', '11:00', '11:30', '13:30', '14:00', '14:30', '14:50', '15:00']
TIMES_SELL = ['09:35', '09:40', '09:45', '10:00', '10:30', '11:00', '11:30', '13:30', '14:00', '14:30', '14:50', '15:00']
DEADLINES = {'10:00': bar('10:00'), '11:30': bar('11:30'), '14:50': bar('14:50')}
# 半仓开盘买 + 半仓低吸：第二个半仓不成交就只有半仓（另一半算现金）
BLENDS = [('半仓开盘买 + 半仓挂开盘价-3%（不成交则只有半仓）', '挂开盘价-3%，不成交放弃'),
          ('半仓开盘买 + 半仓挂开盘价-5%（不成交则只有半仓）', '挂开盘价-5%，不成交放弃'),
          ('半仓开盘买 + 半仓挂昨收-3%（不成交则只有半仓）', '挂昨收-3%，不成交放弃'),
          ('半仓开盘买 + 半仓挂昨收-5%（不成交则只有半仓）', '挂昨收-5%，不成交放弃'),
          ('半仓开盘买 + 半仓挂跌停价（不成交则只有半仓）', '挂跌停价，不成交放弃')]
# 卖出分两半：各一半按两条卖法
SELL_BLENDS = [('半仓挂昨收+7%（14:50 兜底）+ 半仓到+7%后移动止盈2%（14:50 兜底）', '挂昨收+7% 卖，14:50 没到就卖', '到昨收+7% 后移动止盈2%，没到 14:50 卖'),
               ('半仓挂昨收+7%（14:50 兜底）+ 半仓到+7%后移动止盈3%（14:50 兜底）', '挂昨收+7% 卖，14:50 没到就卖', '到昨收+7% 后移动止盈3%，没到 14:50 卖'),
               ('半仓挂昨收+7%（14:50 兜底）+ 半仓到+5%后移动止盈2%（14:50 兜底）', '挂昨收+7% 卖，14:50 没到就卖', '到昨收+5% 后移动止盈2%，没到 14:50 卖'),
               ('半仓挂昨收+5%（14:50 兜底）+ 半仓到+5%后移动止盈2%（14:50 兜底）', '挂昨收+5% 卖，14:50 没到就卖', '到昨收+5% 后移动止盈2%，没到 14:50 卖'),
               ('半仓挂昨收+7%（14:50 兜底）+ 半仓挂涨停价（14:50 兜底）', '挂昨收+7% 卖，14:50 没到就卖', '挂涨停价卖，14:50 没到就卖'),
               ('半仓挂昨收+5%（14:50 兜底）+ 半仓挂昨收+9%（14:50 兜底）', '挂昨收+5% 卖，14:50 没到就卖', '挂昨收+9% 卖，14:50 没到就卖'),
               ('半仓 14:50 卖 + 半仓到+5%后移动止盈2%（14:50 兜底）', '定时 14:50 卖', '到昨收+5% 后移动止盈2%，没到 14:50 卖')]


def buy_rules(b, T):
    """{rule name: (entry price or nan, fill bar)} for one buy day. T has ref, lu, ld of the day."""
    ref, lu, ld = T['ref_b'], T['lu_b'], T['ld_b']
    o = b.open
    R = {'开盘买（现行）': market_buy(b, b.f, lu)}
    for t in TIMES_BUY:
        R[f'定时 {t} 买'] = market_buy(b, bar(t), lu)
    for d in (1, 2, 3, 4, 5, 6, 8, 10):
        for name, D in DEADLINES.items():
            R[f'挂开盘价-{d}%，{name} 未成交改市价'] = limit_buy(b, o * (1 - d / 100), b.f, D, lu, ld, True)
        R[f'挂开盘价-{d}%，不成交放弃'] = limit_buy(b, o * (1 - d / 100), b.f, LAST, lu, ld, False)
    for x in (0, 1, 2, 3, 5, 7):
        for name, D in DEADLINES.items():
            R[f'挂昨收-{x}%，{name} 未成交改市价'] = limit_buy(b, ref * (1 - x / 100), b.f, D, lu, ld, True)
        R[f'挂昨收-{x}%，不成交放弃'] = limit_buy(b, ref * (1 - x / 100), b.f, LAST, lu, ld, False)
    R['挂跌停价，不成交放弃'] = limit_buy(b, ld, b.f, LAST, lu, ld, False)
    R['挂跌停价，14:50 未成交改市价'] = limit_buy(b, ld, b.f, DEADLINES['14:50'], lu, ld, True)
    for name, D in DEADLINES.items():
        R[f'回踩均价线买，{name} 未成交改市价'] = vwap_pullback(b, D, lu, True)
    R['回踩均价线买，不成交放弃'] = vwap_pullback(b, LAST, lu, False)
    for dip in (2, 3, 5, 7):
        for bounce in (1, 2, 3):
            for name in ('11:30', '14:50'):
                R[f'先跌{dip}%再弹{bounce}%买，{name} 未触发改市价'] = dip_rebound(b, dip, bounce, DEADLINES[name], lu, True)
            R[f'先跌{dip}%再弹{bounce}%买，不触发放弃'] = dip_rebound(b, dip, bounce, LAST, lu, False)
    for n in (5, 15, 30):
        for name in ('11:30', '14:50'):
            R[f'突破前{n}分钟高点买，{name} 未触发改市价'] = breakout(b, n, DEADLINES[name], lu, True)
        R[f'突破前{n}分钟高点买，不触发放弃'] = breakout(b, n, LAST, lu, False)
    for name in ('11:30', '14:50'):
        R[f'低开等收复昨收买（高开开盘买），{name} 未触发改市价'] = reclaim(b, ref, DEADLINES[name], lu, True)
    R['低开等收复昨收买（高开开盘买），不触发放弃'] = reclaim(b, ref, LAST, lu, False)
    return R


def sell_rules(b, T, entry_adj):
    """{rule name: (sell price, bar)} for one sell day. entry_adj = entry price on the sell day's price basis."""
    ref, lu, ld = T['ref_s'], T['lu_s'], T['ld_s']
    o = b.open
    close = (float(b.c[LAST]), LAST)
    at_1450 = market_sell(b, DEADLINES['14:50'])
    R = {'开盘卖（现行）': market_sell(b, b.f), '收盘卖': close}
    for t in TIMES_SELL:
        R[f'定时 {t} 卖'] = market_sell(b, bar(t))
    for x in (1, 2, 3, 4, 5, 6, 7, 8, 9):
        hit = limit_sell(b, ref * (1 + x / 100), b.f, LAST, lu)
        R[f'挂昨收+{x}% 卖，没到收盘卖'] = hit or close
        hit2 = limit_sell(b, ref * (1 + x / 100), b.f, DEADLINES['10:00'], lu)
        R[f'挂昨收+{x}% 卖，10:00 没到就卖'] = hit2 or market_sell(b, DEADLINES['10:00'])
        hit3 = limit_sell(b, ref * (1 + x / 100), b.f, DEADLINES['11:30'], lu)
        R[f'挂昨收+{x}% 卖，11:30 没到就卖'] = hit3 or market_sell(b, DEADLINES['11:30'])
        hit4 = limit_sell(b, ref * (1 + x / 100), b.f, DEADLINES['14:50'], lu)
        R[f'挂昨收+{x}% 卖，14:50 没到就卖'] = hit4 or at_1450
    R['挂涨停价卖，没到收盘卖'] = limit_sell(b, lu, b.f, LAST, lu) or close
    R['挂涨停价卖，14:50 没到就卖'] = limit_sell(b, lu, b.f, DEADLINES['14:50'], lu) or at_1450
    for g in (-3, -1, 0, 1, 2, 3, 5, 7):
        strong = o >= ref * (1 + g / 100) - 1e-9
        R[f'高开≥{g:+d}% 开盘卖，否则收盘卖'] = market_sell(b, b.f) if strong else close
        R[f'高开≥{g:+d}% 开盘卖，否则 10:00 卖'] = market_sell(b, b.f) if strong else market_sell(b, DEADLINES['10:00'])
        R[f'高开≥{g:+d}% 开盘卖，否则移动止盈3%'] = market_sell(b, b.f) if strong else trailing_sell(b, 3, b.f)
        R[f'高开≥{g:+d}% 开盘卖，否则跌破均价线卖（10:00 起）'] = market_sell(b, b.f) if strong else vwap_sell(b, 30)
        R[f'低开≤{g:+d}% 开盘卖，否则移动止盈3%'] = market_sell(b, b.f) if not strong else trailing_sell(b, 3, b.f)
        R[f'高开≥{g:+d}% 开盘卖，否则挂昨收+7% 卖、没到 14:50 卖'] = (market_sell(b, b.f) if strong else
                                                               (limit_sell(b, ref * 1.07, b.f, DEADLINES['14:50'], lu) or at_1450))
    for z in (1, 1.5, 2, 3, 4, 5, 7):
        R[f'移动止盈{z}%（开盘起算），否则收盘卖'] = trailing_sell(b, z, b.f)
        R[f'移动止盈{z}%（09:45 起算），否则收盘卖'] = trailing_sell(b, z, bar('09:45'))
        R[f'移动止盈{z}%，且跌破开盘价即卖'] = trailing_sell(b, z, b.f, floor=o)
    for skip in (5, 15, 30, 60):
        R[f'跌破均价线卖（开盘{skip}分钟后起算），否则收盘卖'] = vwap_sell(b, skip)
    for s in (2, 3, 5):
        R[f'止损：跌破买价-{s}% 卖，否则收盘卖'] = stop_sell(b, entry_adj * (1 - s / 100))
        R[f'止损{s}% + 移动止盈3%'] = trailing_sell(b, 3, b.f, floor=entry_adj * (1 - s / 100))
    for x in (3, 5, 7):
        hit = limit_sell(b, ref * (1 + x / 100), b.f, LAST, lu)
        R[f'挂昨收+{x}% 卖，否则移动止盈3%'] = first_of(hit, trailing_sell(b, 3, b.f))
        R[f'挂昨收+{x}% 卖，否则跌破均价线卖（10:00 起）'] = first_of(hit, vwap_sell(b, 30))
    for s in (3, 5, 7):
        hit = limit_sell(b, ref * 1.07, b.f, LAST, lu)
        R[f'挂昨收+7% 卖，跌破昨收-{s}% 止损，否则收盘卖'] = first_of(hit, stop_sell(b, ref * (1 - s / 100)))
        R[f'挂昨收+7% 卖，跌破买价-{s}% 止损，否则收盘卖'] = first_of(hit, stop_sell(b, entry_adj * (1 - s / 100)))
        hit_lu = limit_sell(b, lu, b.f, LAST, lu)
        R[f'挂涨停价卖，跌破昨收-{s}% 止损，否则收盘卖'] = first_of(hit_lu, stop_sell(b, ref * (1 - s / 100)))
    # 追最高点：冲到昨收 +a% 后才开始跟踪高点
    for a in (2, 3, 5, 7, 9):
        arm = ref * (1 + a / 100)
        for z in (0.5, 1, 1.5, 2, 3, 5):
            R[f'到昨收+{a}% 后移动止盈{z}%，没到 14:50 卖'] = armed_trailing(b, arm, z, at_1450)
            if a in (5, 7) and z in (1, 2, 3):
                R[f'到昨收+{a}% 后移动止盈{z}%，没到收盘卖'] = armed_trailing(b, arm, z, close)
        for n in (1, 2, 3, 5):
            R[f'到昨收+{a}% 后跌破前{n}分钟低点卖，没到 14:50 卖'] = armed_swing(b, arm, n, at_1450)
        R[f'到昨收+{a}% 后跌破均价线卖，没到 14:50 卖'] = armed_vwap(b, arm, at_1450)
    # 到 +7% 先不卖、改跟踪，但封涨停就持有到收盘 —— 等同于 armed_trailing（封板时最高价=现价，不会触发）
    return R


# ----------------------------------------------------------------------------------------------- study
def ret_pct(sell_px, entry, T):
    """Return % the engine's way: the sell-day price is put on the buy day's price basis (ex-rights between the days)."""
    return (sell_px * T['close_b'] / T['ref_s'] / entry - 1) * 100


def agg(rets, filled, label):
    r = np.asarray(rets, float)
    f = np.asarray(filled, bool)
    n = len(r)
    rf = r[f]
    allr = np.where(f, r, 0.0)
    half = n // 2
    cum = np.cumsum(allr)
    dd = float((cum - np.maximum.accumulate(cum)).min()) if n else np.nan
    return {'规则': label, '笔数': n, '成交笔数': int(f.sum()), '成交率': f.mean() if n else np.nan,
            '每笔平均%（成交的）': rf.mean() if len(rf) else np.nan, '每笔平均%（未成交算0）': allr.mean() if n else np.nan,
            '中位数%': float(np.median(rf)) if len(rf) else np.nan, '胜率': (rf > 0).mean() if len(rf) else np.nan,
            '合计%': allr.sum(), '标准误%': rf.std(ddof=1) / np.sqrt(len(rf)) if len(rf) > 1 else np.nan,
            '前半段平均%': allr[:half].mean() if half else np.nan, '后半段平均%': allr[half:].mean() if n - half else np.nan,
            '最差一笔%': rf.min() if len(rf) else np.nan, '最好一笔%': rf.max() if len(rf) else np.nan, '最大回撤%': dd}


def regret_stats(pairs):
    """pairs = [(距最高%, 区间捕获率)] -> summary."""
    reg = np.array([p[0] for p in pairs], float)
    cap = np.array([p[1] for p in pairs], float)
    return {'平均距最高%': reg.mean(), '中位距最高%': float(np.median(reg)), '距最高≤1%占比': (reg <= 1).mean(),
            '距最高≤2%占比': (reg <= 2).mean(), '距最高≤3%占比': (reg <= 3).mean(), '距最高>5%占比': (reg > 5).mean(),
            '区间捕获率': np.nanmean(cap)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(TRADES, dtype={'code': str})
    t['filled'] = t['filled'].astype(str).str.lower().isin(['true', '1', '1.0'])
    settled = t[t['result'].isin(['赚', '亏'])].copy().reset_index(drop=True)
    pending = t[t['result'] == '待卖出'].copy()
    print(f'settled trades {len(settled)}, pending {len(pending)}', flush=True)

    panel = store.build_panel()
    rows = []
    for r in pd.concat([settled, pending]).to_dict('records'):
        tb, j = panel.index_of(r['buy_date']), panel.code_index[r['code']]
        ts = tb + 1
        has_sell = ts < len(panel.dates) and bool(panel.has[ts, j]) and r['result'] in ('赚', '亏')
        rows.append({**r, 'tb': tb, 'ts': ts, 'j': j, 'has_sell': has_sell,
                     'ref_b': float(panel.ref[tb, j]), 'lu_b': float(panel.limit_up[tb, j]), 'ld_b': float(panel.limit_down[tb, j]),
                     'close_b': float(panel.close[tb, j]), 'open_b': float(panel.open[tb, j]),
                     'ref_s': float(panel.ref[ts, j]) if has_sell else np.nan,
                     'lu_s': float(panel.limit_up[ts, j]) if has_sell else np.nan,
                     'ld_s': float(panel.limit_down[ts, j]) if has_sell else np.nan,
                     'open_s': float(panel.open[ts, j]) if has_sell else np.nan,
                     'high_s': float(panel.high[ts, j]) if has_sell else np.nan,
                     'low_s': float(panel.low[ts, j]) if has_sell else np.nan,
                     'close_s': float(panel.close[ts, j]) if has_sell else np.nan,
                     'sell_day': panel.dates[ts] if has_sell else ''})
    T = pd.DataFrame(rows)

    need = {}
    for r in T.to_dict('records'):
        need.setdefault(r['buy_date'], set()).add(r['code'])
        if r['has_sell']:
            need.setdefault(r['sell_day'], set()).add(r['code'])
    bars = {}
    for i, (date, codes) in enumerate(sorted(need.items()), 1):
        bars[date] = load_minutes(date, codes)
        if i % 10 == 0:
            print(f'  minutes loaded {i}/{len(need)}', flush=True)

    per_trade, structure = [], []
    buy_cols, sell_cols, sell_extra, combo_cols = {}, {}, {}, {}
    best_case = []
    for r in T.to_dict('records'):
        bb = bars[r['buy_date']].get(r['code'])
        if bb is None:
            print(f'  no buy-day bars for {r["code"]} {r["buy_date"]}', flush=True)
            continue
        R = buy_rules(bb, r)
        e0, k0 = R['开盘买（现行）']
        if r['filled'] and abs(e0 - r['entry']) > 0.0051:
            print(f'  WARNING entry mismatch {r["code"]} {r["buy_date"]}: engine {r["entry"]} vs study {e0}', flush=True)
        low_m = int(np.argmin(np.where(np.arange(240) >= bb.f, bb.l, np.inf)))
        high_m = int(np.argmax(np.where(np.arange(240) >= bb.f, bb.h, -np.inf)))
        st = {'code': r['code'], 'name': r['name'], 'buy_date': r['buy_date'], 'sell_date': r['sell_day'],
              '买入日开盘涨幅%': (bb.open / r['ref_b'] - 1) * 100,
              '买入日最低相对开盘%': (bb.l[bb.f:].min() / bb.open - 1) * 100, '买入日最低时间': TIMES[low_m],
              '买入日最高相对开盘%': (bb.h[bb.f:].max() / bb.open - 1) * 100, '买入日最高时间': TIMES[high_m],
              '买入日收盘相对开盘%': (bb.c[LAST] / bb.open - 1) * 100,
              '买入日触及跌停': bool(profile.at_or_below(bb.l[bb.f:].min(), r['ld_b'])),
              '买入日触及涨停': bool(profile.at_or_above(bb.h[bb.f:].max(), r['lu_b'])),
              '买入日收盘涨停': bool(profile.at_or_above(bb.c[LAST], r['lu_b']))}
        row = {'code': r['code'], 'name': r['name'], 'buy_date': r['buy_date'], 'sell_date': r['sell_day']}
        for k, (e, m) in R.items():
            row[f'买:{k}'] = e
            row[f'买:{k}@'] = TIMES[m] if m >= 0 else ''
        if not r['has_sell']:
            structure.append(st)
            per_trade.append(row)
            continue
        sb = bars[r['sell_day']].get(r['code'])
        if sb is None:
            print(f'  no sell-day bars for {r["code"]} {r["sell_day"]}', flush=True)
            continue
        adj = r['ref_s'] / r['close_b']                      # entry on the sell day's price basis
        S = sell_rules(sb, r, r['entry'] * adj)
        for label, k1, k2 in SELL_BLENDS:
            (p1, m1), (p2, m2) = S[k1], S[k2]
            S[label] = (0.5 * (p1 + p2), max(m1, m2))
        s0 = S['开盘卖（现行）'][0]
        hi_day, lo_day = float(sb.h[sb.f:].max()), float(sb.l[sb.f:].min())
        low_s = int(np.argmin(np.where(np.arange(240) >= sb.f, sb.l, np.inf)))
        high_s = int(np.argmax(np.where(np.arange(240) >= sb.f, sb.h, -np.inf)))
        st.update({'卖出日开盘涨幅%': (sb.open / r['ref_s'] - 1) * 100,
                   '卖出日最高相对开盘%': (hi_day / sb.open - 1) * 100, '卖出日最高时间': TIMES[high_s],
                   '卖出日最高涨幅%': (hi_day / r['ref_s'] - 1) * 100,
                   '卖出日最低相对开盘%': (lo_day / sb.open - 1) * 100, '卖出日最低时间': TIMES[low_s],
                   '卖出日收盘相对开盘%': (sb.c[LAST] / sb.open - 1) * 100,
                   '卖出日收盘涨幅%': (sb.c[LAST] / r['ref_s'] - 1) * 100,
                   '卖出日触及涨停': bool(profile.at_or_above(hi_day, r['lu_s'])),
                   '卖出日收盘涨停': bool(profile.at_or_above(sb.c[LAST], r['lu_s'])),
                   '现行收益%（开盘买开盘卖）': ret_pct(s0, r['entry'], r) if r['filled'] else np.nan,
                   '开盘买收盘卖%': ret_pct(float(sb.c[LAST]), r['entry'], r),
                   '均价卖收益%': r['t1_mid'], '最好%': r['t1_best'], '最差%': r['t1_worst']})
        structure.append(st)
        best_case.append(ret_pct(hi_day, r['entry'], r))
        mid = (r['high_s'] + r['low_s']) / 2
        bench = (('均价卖', mid), ('开盘卖', r['open_s']), ('收盘卖', r['close_s']))
        for k, (e, _) in R.items():
            ok = np.isfinite(e)
            for name, px in bench:
                buy_cols.setdefault((k, name), []).append((ret_pct(px, e, r) if ok else 0.0, ok))
                row[f'买:{k}|{name}%'] = ret_pct(px, e, r) if ok else np.nan
        e_open = R['开盘买（现行）'][0]
        for label, second in BLENDS:
            e2 = R[second][0]
            for name, px in bench:
                r1 = ret_pct(px, e_open, r) if np.isfinite(e_open) else 0.0
                r2 = ret_pct(px, e2, r) if np.isfinite(e2) else 0.0
                buy_cols.setdefault((label, name), []).append((0.5 * r1 + 0.5 * r2, np.isfinite(e_open)))
                row[f'买:{label}|{name}%'] = 0.5 * r1 + 0.5 * r2
        for k, (px, m) in S.items():
            sell_cols.setdefault(k, []).append((ret_pct(px, r['entry'], r), True))
            regret = (hi_day - px) / hi_day * 100
            capture = (px - lo_day) / (hi_day - lo_day) if hi_day > lo_day else 1.0
            sell_extra.setdefault(k, []).append((regret, capture))
            row[f'卖:{k}%'] = ret_pct(px, r['entry'], r)
            row[f'卖:{k}@'] = TIMES[m]
            row[f'卖:{k}|距最高%'] = regret
        for kb in ('开盘买（现行）', '挂开盘价-3%，10:00 未成交改市价', '挂开盘价-3%，14:50 未成交改市价',
                   '挂昨收-3%，14:50 未成交改市价', '回踩均价线买，11:30 未成交改市价', '先跌3%再弹1%买，14:50 未触发改市价',
                   '定时 10:00 买', '定时 14:50 买'):
            e, _ = R[kb]
            ok = np.isfinite(e)
            for ks, (px, _m) in S.items():
                combo_cols.setdefault((kb, ks), []).append((ret_pct(px, e, r) if ok else 0.0, ok))
        per_trade.append(row)

    pt = pd.DataFrame(per_trade)
    pt.to_csv(OUT / 'per_trade.csv', index=False, encoding='utf-8-sig')
    S_df = pd.DataFrame(structure)
    S_df.to_csv(OUT / 'structure.csv', index=False, encoding='utf-8-sig')

    buy_df = pd.DataFrame([{**agg([x[0] for x in v], [x[1] for x in v], k), '卖出基准': bench}
                           for (k, bench), v in buy_cols.items()])
    sell_df = pd.DataFrame([{**agg([x[0] for x in v], [x[1] for x in v], k), **regret_stats(sell_extra[k])}
                            for k, v in sell_cols.items()])
    combo_df = pd.DataFrame([{**agg([x[0] for x in v], [x[1] for x in v], f'{kb}  ×  {ks}'), '买法': kb, '卖法': ks}
                             for (kb, ks), v in combo_cols.items()])
    for df, name, key in ((buy_df, 'buy_rules.csv', '每笔平均%（未成交算0）'), (sell_df, 'sell_rules.csv', '每笔平均%（成交的）'),
                          (combo_df, 'combo_rules.csv', '每笔平均%（未成交算0）')):
        df.sort_values(key, ascending=False).to_csv(OUT / name, index=False, encoding='utf-8-sig', float_format='%.3f')
    write_report(buy_df, sell_df, combo_df, S_df, T, pt, float(np.mean(best_case)))
    print(f'written {OUT}', flush=True)


def fmt(df, cols, n=None):
    d = df if n is None else df.head(n)
    head = '| ' + ' | '.join(str(c) for c in cols) + ' |\n|' + '---|' * len(cols) + '\n'
    body = ''
    for r in d.to_dict('records'):
        cells = []
        for c in cols:
            v = r.get(c, '')
            if isinstance(v, (float, np.floating)):
                cells.append('—' if not np.isfinite(v) else (f'{v:.0%}' if (c in ('成交率', '胜率', '区间捕获率') or '占比' in str(c)) else f'{v:+.2f}'))
            elif isinstance(v, (bool, np.bool_)):
                cells.append('是' if v else '否')
            else:
                cells.append(str(v))
        body += '| ' + ' | '.join(cells) + ' |\n'
    return head + body


def group_table(m, gap_col, bins, labels, cols, prefix, suffix):
    m = m.copy()
    m['组'] = pd.cut(m[gap_col], bins=bins, labels=labels)
    rows = []
    for g, d in m.groupby('组', observed=True):
        row = {'分组': str(g), '笔数': len(d)}
        for c in cols:
            row[c] = d[f'{prefix}{c}{suffix}'].fillna(0).mean()
        rows.append(row)
    row = {'分组': '全部', '笔数': len(m)}
    for c in cols:
        row[c] = m[f'{prefix}{c}{suffix}'].fillna(0).mean()
    rows.append(row)
    return pd.DataFrame(rows)


def write_report(buy_df, sell_df, combo_df, S, T, pt, best_mean):
    n = int(sell_df['笔数'].iloc[0])
    L = ['# 买卖点研究报告（自动生成，timing_study.py）', '',
         f'样本：最优组合已结算的 {n} 笔（{T["buy_date"].min()} 至 {T[T["has_sell"]]["sell_day"].max()}），'
         '买入日 = 复盘次日，卖出日 = 买入次日。收益口径同引擎（卖出日价格换算到买入日价格基准；未扣费用）。',
         '“未成交算0”= 那一笔当现金；“前半段/后半段”= 按时间顺序前一半和后一半的每笔平均，两段差得远的规则不可信。', '']
    settled = S[S['sell_date'] != '']
    L += ['## 一、买入日和卖出日的日内结构', '',
          f'- 买入日开盘涨幅：平均 {settled["买入日开盘涨幅%"].mean():+.2f}%，中位数 {settled["买入日开盘涨幅%"].median():+.2f}%；'
          f'高开（>0）{(settled["买入日开盘涨幅%"] > 0).mean():.0%}',
          f'- 买入日最低点相对开盘：平均 {settled["买入日最低相对开盘%"].mean():+.2f}%，中位数 {settled["买入日最低相对开盘%"].median():+.2f}%；'
          f'最低点出现在 10:00 前的占 {(settled["买入日最低时间"] <= "10:00").mean():.0%}，11:30 前 {(settled["买入日最低时间"] <= "11:30").mean():.0%}',
          f'- 买入日最低点比开盘低 ≥3% 的占 {(settled["买入日最低相对开盘%"] <= -3).mean():.0%}，≥5% 的占 {(settled["买入日最低相对开盘%"] <= -5).mean():.0%}',
          f'- 买入日触及跌停 {int(settled["买入日触及跌停"].sum())} 笔，触及涨停 {int(settled["买入日触及涨停"].sum())} 笔，收盘涨停 {int(settled["买入日收盘涨停"].sum())} 笔',
          f'- 卖出日开盘涨幅：平均 {settled["卖出日开盘涨幅%"].mean():+.2f}%，中位数 {settled["卖出日开盘涨幅%"].median():+.2f}%',
          f'- 卖出日最高点相对开盘：平均 {settled["卖出日最高相对开盘%"].mean():+.2f}%；最高点出现在 10:00 前的占 {(settled["卖出日最高时间"] <= "10:00").mean():.0%}，'
          f'11:30 前 {(settled["卖出日最高时间"] <= "11:30").mean():.0%}',
          f'- 卖出日最高涨幅（相对昨收）：平均 {settled["卖出日最高涨幅%"].mean():+.2f}%，中位数 {settled["卖出日最高涨幅%"].median():+.2f}%；'
          f'≥+5% 的占 {(settled["卖出日最高涨幅%"] >= 5).mean():.0%}，≥+7% 的占 {(settled["卖出日最高涨幅%"] >= 7).mean():.0%}，≥+9% 的占 {(settled["卖出日最高涨幅%"] >= 9).mean():.0%}',
          f'- 卖出日最低点相对开盘：平均 {settled["卖出日最低相对开盘%"].mean():+.2f}%；收盘相对开盘：平均 {settled["卖出日收盘相对开盘%"].mean():+.2f}%，'
          f'收盘高于开盘的占 {(settled["卖出日收盘相对开盘%"] > 0).mean():.0%}',
          f'- 卖出日触及涨停 {int(settled["卖出日触及涨停"].sum())} 笔，收盘涨停 {int(settled["卖出日收盘涨停"].sum())} 笔',
          f'- 最好情况（卖出日最高价卖出）：每笔平均 {best_mean:+.2f}%', '']
    cols_b = ['规则', '成交率', '每笔平均%（未成交算0）', '每笔平均%（成交的）', '胜率', '合计%', '前半段平均%', '后半段平均%', '最差一笔%', '最大回撤%']
    for bench in ('均价卖', '收盘卖', '开盘卖'):
        d = buy_df[buy_df['卖出基准'] == bench].sort_values('每笔平均%（未成交算0）', ascending=False)
        base = d[d['规则'] == '开盘买（现行）'].iloc[0]
        L += [f'## 二、买法比较（卖出按{bench}）', '',
              f'现行开盘买：每笔 {base["每笔平均%（未成交算0）"]:+.2f}%，胜率 {base["胜率"]:.0%}，合计 {base["合计%"]:+.2f}%。', '',
              '前 25 名：', '', fmt(d, cols_b, 25), '',
              '定时买：', '', fmt(d[d['规则'].str.startswith('定时') | (d['规则'] == '开盘买（现行）')], cols_b), '',
              '挂跌停价 / 挂昨收下方（不成交放弃）：', '',
              fmt(d[d['规则'].str.contains('放弃') & (d['规则'].str.contains('跌停') | d['规则'].str.startswith('挂昨收'))], cols_b), '',
              '半仓开盘 + 半仓低吸：', '', fmt(d[d['规则'].str.startswith('半仓') | (d['规则'] == '开盘买（现行）')], cols_b), '']
    cols_s = ['规则', '每笔平均%（成交的）', '中位数%', '胜率', '合计%', '前半段平均%', '后半段平均%', '最差一笔%', '最大回撤%', '平均距最高%']
    d = sell_df.sort_values('每笔平均%（成交的）', ascending=False)
    base = d[d['规则'] == '开盘卖（现行）'].iloc[0]
    L += ['## 三、卖法比较（买入按现行开盘价）', '',
          f'现行开盘卖：每笔 {base["每笔平均%（成交的）"]:+.2f}%，胜率 {base["胜率"]:.0%}，合计 {base["合计%"]:+.2f}%。', '',
          '前 40 名：', '', fmt(d, cols_s, 40), '',
          '定时卖：', '', fmt(d[d['规则'].str.startswith('定时') | d['规则'].isin(['开盘卖（现行）', '收盘卖'])], cols_s), '',
          '挂高卖（没到收盘卖 / 14:50 卖）：', '', fmt(d[d['规则'].str.contains('没到收盘卖') | d['规则'].str.contains('14:50 没到就卖')], cols_s), '',
          '追最高点（到昨收 +a% 后跟踪高点）：', '', fmt(d[d['规则'].str.startswith('到昨收')], cols_s), '',
          '卖出分两半：', '', fmt(d[d['规则'].str.startswith('半仓')], cols_s), '',
          '挂高卖 + 止损：', '', fmt(d[d['规则'].str.contains('止损，否则收盘卖')], cols_s), '',
          '高开直接卖：', '', fmt(d[d['规则'].str.startswith('高开') | d['规则'].str.startswith('低开')], cols_s), '',
          '移动止盈：', '', fmt(d[d['规则'].str.startswith('移动止盈')], cols_s), '',
          '均价线 / 止损：', '', fmt(d[d['规则'].str.contains('均价线') & ~d['规则'].str.startswith('到昨收') | d['规则'].str.startswith('止损')], cols_s), '']
    d = combo_df.sort_values('每笔平均%（未成交算0）', ascending=False)
    L += ['## 四、买法 × 卖法 组合（前 30）', '', fmt(d, ['规则', '成交率', '每笔平均%（未成交算0）', '胜率', '合计%', '前半段平均%', '后半段平均%', '最大回撤%'], 30), '']

    m = settled.merge(pt, on=['code', 'buy_date'], suffixes=('', '_pt'))
    buy_sel = ['开盘买（现行）', '挂开盘价-1%，14:50 未成交改市价', '挂开盘价-3%，14:50 未成交改市价', '挂开盘价-3%，不成交放弃',
               '挂昨收-2%，14:50 未成交改市价', '回踩均价线买，11:30 未成交改市价', '先跌3%再弹1%买，14:50 未触发改市价',
               '低开等收复昨收买（高开开盘买），不触发放弃', '定时 10:00 买', '半仓开盘买 + 半仓挂跌停价（不成交则只有半仓）']
    g = group_table(m, '买入日开盘涨幅%', [-100, -3, 0, 3, 7, 100], ['低开≤-3%', '-3%~0', '0~+3%', '+3%~+7%', '≥+7%（含一字排队）'],
                    buy_sel, '买:', '|均价卖%')
    L += ['## 五、按买入日开盘涨幅分组看买法（卖出按均价；每笔平均%，未成交算 0）', '', fmt(g, ['分组', '笔数'] + buy_sel), '']
    sell_sel = ['开盘卖（现行）', '收盘卖', '定时 14:50 卖', '挂昨收+7% 卖，没到收盘卖', '挂昨收+7% 卖，14:50 没到就卖', '挂涨停价卖，没到收盘卖',
                '到昨收+5% 后移动止盈2%，没到 14:50 卖', '到昨收+7% 后移动止盈2%，没到 14:50 卖', '到昨收+3% 后移动止盈1%，没到 14:50 卖',
                '半仓挂昨收+7%（14:50 兜底）+ 半仓到+7%后移动止盈2%（14:50 兜底）']
    g = group_table(m, '卖出日开盘涨幅%', [-100, -3, 0, 3, 7, 100], ['低开≤-3%', '-3%~0', '0~+3%', '+3%~+7%', '≥+7%'],
                    sell_sel, '卖:', '%')
    L += ['## 六、按卖出日开盘涨幅分组看卖法（买入按现行开盘价；每笔平均%）', '', fmt(g, ['分组', '笔数'] + sell_sel), '']
    g = group_table(m, '卖出日最高涨幅%', [-100, 0, 3, 5, 7, 9.5, 100], ['最高≤0', '0~3%', '3~5%', '5~7%', '7~9.5%', '≥9.5%（涨停）'],
                    sell_sel, '卖:', '%')
    L += ['## 六b、按卖出日最高涨幅分组看卖法（事后分组，只为看每条卖法在什么样的日子吃亏；每笔平均%）', '', fmt(g, ['分组', '笔数'] + sell_sel), '']

    # 十、离最高点有多远
    cols_r = ['规则', '每笔平均%（成交的）', '胜率', '平均距最高%', '中位距最高%', '距最高≤1%占比', '距最高≤2%占比', '距最高≤3%占比', '距最高>5%占比', '区间捕获率', '前半段平均%', '后半段平均%']
    pick = ['开盘卖（现行）', '定时 10:00 卖', '收盘卖', '定时 14:50 卖', '挂昨收+5% 卖，14:50 没到就卖', '挂昨收+7% 卖，14:50 没到就卖',
            '挂昨收+9% 卖，14:50 没到就卖', '挂涨停价卖，14:50 没到就卖', '移动止盈3%（开盘起算），否则收盘卖', '移动止盈5%（09:45 起算），否则收盘卖',
            '跌破均价线卖（开盘60分钟后起算），否则收盘卖']
    armed = sell_df[sell_df['规则'].str.startswith('到昨收') | sell_df['规则'].str.startswith('半仓')]
    top_armed = armed.sort_values('每笔平均%（成交的）', ascending=False).head(12)['规则'].tolist()
    closest = sell_df.sort_values('平均距最高%').head(12)['规则'].tolist()
    sel = sell_df[sell_df['规则'].isin(pick + top_armed + closest)].sort_values('每笔平均%（成交的）', ascending=False)
    best_row = pd.DataFrame([{'规则': '最好情况（卖在当天最高价，事后才知道）', '每笔平均%（成交的）': best_mean, '胜率': np.nan, '平均距最高%': 0.0,
                              '中位距最高%': 0.0, '距最高≤1%占比': 1.0, '距最高≤2%占比': 1.0, '距最高≤3%占比': 1.0, '距最高>5%占比': 0.0,
                              '区间捕获率': 1.0, '前半段平均%': np.nan, '后半段平均%': np.nan}])
    L += ['## 七、离当天最高点有多远（卖出日；距最高% = (最高价 − 卖出价) / 最高价）', '',
          '区间捕获率 = (卖出价 − 最低价) / (最高价 − 最低价)，1 = 卖在最高、0 = 卖在最低。', '',
          fmt(pd.concat([best_row, sel]), cols_r), '',
          '按“平均距最高%”最小排的前 12 条：', '', fmt(sell_df.sort_values('平均距最高%').head(12), cols_r), '']

    ldm = S[S['买入日触及跌停']].merge(pt, on=['code', 'buy_date'], suffixes=('', '_pt'))
    ldm['跌停买成交价'] = ldm['买:挂跌停价，不成交放弃']
    ldm['跌停买成交时间'] = ldm['买:挂跌停价，不成交放弃@']
    ldm['开盘买'] = ldm['买:开盘买（现行）']
    for c in ('买:挂跌停价，不成交放弃|收盘卖%', '买:开盘买（现行）|收盘卖%', '买:挂跌停价，不成交放弃|均价卖%'):
        if c not in ldm:
            ldm[c] = np.nan
    L += ['## 八、买入日触及跌停的信号（“跌停买”）', '',
          fmt(ldm, ['buy_date', 'name', '买入日开盘涨幅%', '开盘买', '跌停买成交价', '跌停买成交时间', '买入日收盘相对开盘%', '买入日收盘涨停',
                    '现行收益%（开盘买开盘卖）', '开盘买收盘卖%', '买:挂跌停价，不成交放弃|收盘卖%', '买:挂跌停价，不成交放弃|均价卖%']), '']

    L += ['## 九、逐笔', '', fmt(settled,
          ['buy_date', 'name', '买入日开盘涨幅%', '买入日最低相对开盘%', '买入日最低时间', '买入日收盘相对开盘%', '卖出日开盘涨幅%', '卖出日最高涨幅%',
           '卖出日最高时间', '卖出日收盘涨幅%', '现行收益%（开盘买开盘卖）', '开盘买收盘卖%', '均价卖收益%', '最好%', '最差%']), '']
    pend = S[S['sell_date'] == '']
    if len(pend):
        L += ['## 十、待卖出（买入日已知，卖出日未到）', '', fmt(pend, ['buy_date', 'name', '买入日开盘涨幅%', '买入日最低相对开盘%', '买入日最低时间',
              '买入日最高相对开盘%', '买入日最高时间', '买入日收盘相对开盘%', '买入日触及跌停', '买入日收盘涨停']), '']
    (OUT / 'report.md').write_text('\n'.join(L), encoding='utf-8')


if __name__ == '__main__':
    main()
