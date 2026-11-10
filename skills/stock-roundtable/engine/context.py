"""Decision context for one day and one meeting time, built only from data visible at the cutoff.

A decision on day T at cutoff bar k may read:
  * panel rows [:t]            daily history up to the previous trading day
  * panel row t pre-open facts reference price, price band, limit prices (published before 09:30)
  * minute bars [0, k) of T    intraday facts up to the cutoff
  * History                    summaries and market metrics of earlier, finished days
It never reads bar k or later, panel row t's close/high/low/amount/has, or any later day.
The Context keeps no reference to the minute arrays or the panel, so debaters cannot reach them.
"""
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from . import config, minutes, profile

TAIL_START = minutes.TIME_INDEX['14:30']


class History:
    """Memory filled strictly in date order by the runner, after each day's decisions are recorded."""

    def __init__(self):
        self.summaries = {}               # date -> whole-day facts of a finished day (index: panel j)
        self.market = defaultdict(list)   # meeting time -> [(date, metrics)] of earlier full-market days
        self.stages = defaultdict(list)   # meeting time -> [(date, stage)]

    def recent_summaries(self, before, n):
        dates = sorted(d for d in self.summaries if d < before)[-n:]
        return [self.summaries[d] for d in dates]

    def prune(self, keep=8):
        for d in sorted(self.summaries)[:-keep]:
            del self.summaries[d]


class Context:
    pass


def _boards(panel, t, lookback=40):
    """Consecutive sealed limit-ups ending at row t-1 (a suspension does not break a run)."""
    n = len(panel.codes)
    boards = np.zeros(n, int)
    alive = np.ones(n, bool)
    for i in range(t - 1, max(t - lookback, 0) - 1, -1):
        trade = panel.has[i]
        sealed = panel.lu_seal[i]
        boards += alive & trade & sealed
        alive &= ~(trade & ~sealed)
    return boards


def _max_run(panel, t, window=20):
    n = len(panel.codes)
    run = np.zeros(n, int)
    best = np.zeros(n, int)
    for i in range(max(t - window, 0), t):
        trade = panel.has[i]
        sealed = panel.lu_seal[i]
        run = np.where(trade & sealed, run + 1, np.where(trade, 0, run))
        best = np.maximum(best, run)
    return best


def history_features(panel, t):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        closes = pd.DataFrame(panel.adjusted('close', t)).ffill().to_numpy()
        adj_high = panel.adjusted('high', t)
        adj_low = panel.adjusted('low', t)
        last = closes[-1]

        def back(n):
            return closes[-1 - n] if t > n else np.full(len(last), np.nan)

        high20 = np.nanmax(adj_high[-20:], axis=0)
        low20 = np.nanmin(adj_low[-20:], axis=0)
        w60 = slice(max(t - 60, 0), t)
        touch60 = panel.lu_touch[w60].sum(axis=0)
        seal60 = panel.lu_seal[w60].sum(axis=0)
        sealed_rows = panel.lu_seal[w60]
        days_since = np.where(sealed_rows.any(axis=0), np.argmax(sealed_rows[::-1], axis=0) + 1, 99)
        feats = pd.DataFrame({
            'j': np.arange(len(panel.codes)),
            'n_hist': panel.has[:t].sum(axis=0),
            'prev_close_adj': last,
            'ma5': np.nanmean(closes[-5:], axis=0),
            'ma10': np.nanmean(closes[-10:], axis=0),
            'ma20': np.nanmean(closes[-20:], axis=0),
            'ma20_prev': np.nanmean(closes[-25:-5], axis=0),
            'ret5': (last / back(5) - 1.0) * 100.0,
            'ret10': (last / back(10) - 1.0) * 100.0,
            'ret20': (last / back(20) - 1.0) * 100.0,
            'high20': high20,
            'low20': low20,
            'pos20': (last - low20) / (high20 - low20),
            'amt_avg5': np.nanmean(panel.amount[max(t - 5, 0):t], axis=0),
            'lu10': panel.lu_seal[max(t - 10, 0):t].sum(axis=0),
            'lu20': panel.lu_seal[max(t - 20, 0):t].sum(axis=0),
            'touch60': touch60,
            'seal_rate60': np.where(touch60 > 0, seal60 / np.maximum(touch60, 1), np.nan),
            'days_since_lu': days_since,
            'boards_y': _boards(panel, t),
            'max_run20': _max_run(panel, t),
            'ret_y': (panel.close[t - 1] / panel.ref[t - 1] - 1.0) * 100.0,
            'ret_s_y': panel.ret_scaled[t - 1],
            'lu_seal_y': panel.lu_seal[t - 1],
            'lu_touch_y': panel.lu_touch[t - 1],
            'ld_seal_y': panel.ld_seal[t - 1],
            'ld_seal_y2': panel.ld_seal[t - 2],
        })
    return feats


def _cluster(day, rows, k, threshold):
    """Average-linkage clusters of 1-minute return series (bars [0, k) only). Returns labels, 0 = none."""
    labels = np.zeros(len(rows), int)
    if len(rows) < config.CLUSTER_MIN_STRONG or k < 6:
        return labels
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.nan_to_num(np.diff(np.log(day.close[rows, :k]), axis=1))
    keep = r.std(axis=1) > 0
    if keep.sum() < config.CLUSTER_MIN_STRONG:
        return labels
    corr = np.nan_to_num(np.corrcoef(r[keep]))
    dist = np.clip(1.0 - corr, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    z = linkage(squareform(dist, checks=False), method='average')
    labels[np.nonzero(keep)[0]] = fcluster(z, t=1.0 - threshold, criterion='distance')
    return labels


def _percentile(history, key, value):
    vals = [m[key] for _, m in history if m.get(key) is not None and np.isfinite(m[key])]
    if len(vals) < config.STAGE_MIN_HISTORY or value is None or not np.isfinite(value):
        return None
    return float((np.asarray(vals) < value).mean())


def _stage_yangjia(mode, market, hist_market, prev_stage):
    """Emotion stage in the market-generic cycle the 炒股养家 skill uses (not his own wording)."""
    d = config.STAGE_DEFAULTS[mode]
    recent = hist_market[-config.STAGE_LOOKBACK:]
    lu, ld, zb = market['lu_touch'], market['ld_now'], market['zb_rate']
    prem = market['prem'] if market['prem'] is not None else 0.0
    p_lu = _percentile(recent, 'lu_touch', lu)
    p_ld = _percentile(recent, 'ld_now', ld)
    p_zb = _percentile(recent, 'zb_rate', zb)
    if p_lu is not None:
        basis = f'按前{len(recent)}个同口径交易日分位'
        hot, warm, cold = p_lu >= 0.8, p_lu >= 0.5, p_lu <= 0.2
        many_ld, high_zb = p_ld >= 0.7, p_zb >= 0.7
    else:
        basis = '历史样本不足10天，按固定阈值'
        hot, warm, cold = lu >= d['lu_hot'], lu >= (d['lu_hot'] + d['lu_cold']) / 2, lu <= d['lu_cold']
        many_ld, high_zb = ld >= d['ld_cold'], zb >= d['zb_high']
    height_up = market['height'] >= market['height_3d_ago']
    ev = [f"触板{lu}家（{'' if p_lu is None else f'分位{p_lu:.0%}，'}{basis}）", f"跌停{ld}家", f"炸板率{zb:.0%}",
          f"昨日涨停今日均{market['prem_raw']:+.2f}%" if market['prem_raw'] is not None else '昨日无涨停样本',
          f"最高{market['height']}板（3日前{market['height_3d_ago']}板）"]
    if cold and (many_ld or prem <= d['prem_cold']):
        stage = '冰点'
    elif prem <= d['prem_cold'] and many_ld and not height_up:
        stage = '退潮'
    elif hot and prem >= d['prem_hot'] and not high_zb:
        stage = '高潮'
    elif hot and prem >= d['prem_hot'] * 0.6 and height_up:
        stage = '加速'
    elif high_zb and -1.0 <= prem <= 2.0:
        stage = '分歧'
    elif prev_stage in ('冰点', '退潮') and prem >= 0:
        stage = '修复'
    elif warm and prem >= 0 and (market['jj1'] or 0) >= 0.2:
        stage = '发酵'
    else:
        stage = '混合'
    return stage, ev


def _stage_kobe(market):
    """92科比 2020 帖的四阶段（据转帖整理）：低位震荡 / 主升 / 高位震荡 / 主跌。"""
    h = market['height']
    lead = market['leader_ret_s']
    prem = market['prem'] if market['prem'] is not None else 0.0
    if prem <= -2.0 or (lead is not None and lead <= -5.0 and h >= 4):
        return '主跌'
    if h >= 4 and h >= market['height_5d_ago'] and (market['leader_sealed'] or (lead or 0) >= 5):
        return '主升'
    if h >= 4:
        return '高位震荡'
    return '低位震荡'


def _flag(series):
    return series.fillna(False).astype(bool)


def _attach_position_gate(pool, panel, t, date):
    """F3 位置体检：给候选池加 qc_pass / qc_note，辩手只看得见 qc_pass=True 的票。

    必须在辩论之前跑 —— 先投票再核位置等于先射箭再画靶（2026-10-07 三次翻转的根源）。
    数据不足（次新股等）按"不进池"处理：没把握确认位置就不投。
    """
    from pick_qc import inspect
    # 前复权只算一次，整池复用 —— 否则每只票都重算，几百只票会慢到跑不动
    pre = tuple(panel.adjusted(k, t) for k in ('close', 'high', 'low'))
    flags, notes = [], []
    for raw in pool['code'].tolist():
        code = str(raw).zfill(6)
        rec = inspect(panel, date, code, pre=pre)
        if rec is None:
            flags.append(False)
            notes.append('无数据/历史不足')
            continue
        gates = rec['gates']
        bad = [k for k in config.POSITION_GATE_KEYS if not gates.get(k, '').startswith('PASS')]
        flags.append(not bad)
        notes.append('过闸' if not bad else '未过:' + '、'.join(bad))
    pool['qc_pass'] = flags
    pool['qc_note'] = notes


def build(panel, t, date, spec, day, hist, float_shares):
    """spec: a meeting from config.meeting('HH:MM'), or the backtest mode name 'intraday' / 'eod'."""
    if isinstance(spec, str):
        spec = config.MODES[spec]
    mode, mkey, k = spec['mode'], spec['key'], spec['cutoff_index']
    j = day.panel_j
    cut = minutes.cut_features(day, k, panel.ref[t, j], panel.pct[t, j], panel.limit_up[t, j], panel.limit_down[t, j])
    cut = cut[cut['trades_seen'] > 0]  # no trade before the cutoff -> the stock is not visible yet (TDX writes flat
    #                                    zero-volume bars for a suspended stock: bars alone do not mean it traded)
    hf = history_features(panel, t)
    df = cut.merge(hf, on='j', how='left', validate='one_to_one').reset_index(drop=True)
    jj = df['j'].to_numpy()
    df['code'] = np.asarray(panel.codes, dtype=object)[jj]
    df['name'] = panel.names[jj]
    df['industry'] = [x if x else '未分类' for x in panel.industries[jj]]
    df['float_shares'] = float_shares[jj]
    for col in ('open', 'low', 'high'):
        df[f'{col}_ret'] = df[f'{col}_s'] * df['pct'] / 10.0
    with np.errstate(invalid='ignore', divide='ignore'):
        df['turnover'] = df['volume'] / df['float_shares'] * 100.0
        df['price_vs_ma5'] = (df['price'] / df['ma5'] - 1.0) * 100.0
        df['price_vs_ma10'] = (df['price'] / df['ma10'] - 1.0) * 100.0
    for col in ('lu_seal_y', 'lu_touch_y', 'ld_seal_y', 'ld_seal_y2', 'lu_touched', 'lu_now', 'lu_reseal', 'ld_now',
                'ld_touched', 'one_word'):
        df[col] = _flag(df[col])

    # Amount so far versus the same cutoff on recent finished days.
    if mode == 'eod':
        base = df['amt_avg5'].to_numpy()
    else:
        prior = hist.recent_summaries(date, 5)
        if prior:
            base = pd.concat([s[f'amt_k{k}'] for s in prior], axis=1).mean(axis=1).reindex(jj).to_numpy()
        else:
            base = np.full(len(df), np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        df['amt_ratio'] = df['amount'].to_numpy() / base

    # Yesterday's intraday facts (a finished day).
    yday = hist.summaries.get(panel.dates[t - 1])
    if yday is not None:
        y = yday.reindex(jj)
        df['opens_y'] = y['lu_opens'].to_numpy(dtype=float)
        df['first_y'] = y['lu_first'].to_numpy(dtype=float)
    else:
        df['opens_y'] = np.nan
        df['first_y'] = np.nan
    df['weak_y'] = (df['lu_touch_y'] & ~df['lu_seal_y']) | (df['lu_seal_y'] & (df['opens_y'] >= 2))
    df['board_now'] = np.where(df['lu_now'], df['boards_y'] + 1, 0)

    has_band = np.isfinite(df['pct']) & (df['ref'] > 0)
    df['has_band'] = has_band
    # ST names trade in their own ecosystem (and on a 5% band before 2026-07-06): short-term practice
    # keeps them out of the streak ladder, so they never count as a market or group leader.
    if not hasattr(panel, 'is_st'):
        panel.is_st = np.array([profile.is_risk_warning(n) for n in panel.names])
    df['is_st'] = panel.is_st[jj]
    m = df[has_band]
    ml = df[has_band & ~df['is_st']]
    ladder = np.isfinite(panel.pct[t]) & ~panel.is_st

    # ---------------- market ----------------
    up, down = int((m['ret'] > 0).sum()), int((m['ret'] < 0).sum())
    lu_touch, lu_now = int(m['lu_touched'].sum()), int(m['lu_now'].sum())
    y_lu = m[m['lu_seal_y']]
    y_lb = m[m['boards_y'] >= 2]
    y_fb = m[m['boards_y'] == 1]
    height = int(ml['boards_y'].max()) if len(ml) else 0
    leaders = ml[ml['boards_y'] == height].sort_values(['lu_now', 'ret_s', 'amount'], ascending=False) \
        if height >= 2 else ml.iloc[0:0]
    ew_ret = float(m['ret'].mean()) if len(m) else 0.0
    n_idx = 30
    with np.errstate(invalid='ignore', divide='ignore'):
        r = panel.close[t - n_idx:t] / panel.ref[t - n_idx:t] - 1.0
    ok = panel.has[t - n_idx:t] & np.isfinite(panel.pct[t - n_idx:t]) & np.isfinite(r)
    eq = np.where(ok, r, 0.0).sum(axis=1) / np.maximum(ok.sum(axis=1), 1)
    series = np.cumprod(np.append(1.0 + eq, 1.0 + ew_ret / 100.0))

    def mean_or_none(s):
        return float(s.mean()) if len(s) else None

    market = {
        'date': date, 'mode': mode, 'time': mkey, 'n_trading': int(len(m)), 'up': up, 'down': down,
        'breadth': up / max(up + down, 1), 'ew_ret': ew_ret,
        'idx_above_ma5': bool(series[-1] > series[-5:].mean()),
        'idx_above_ma10': bool(series[-1] > series[-10:].mean()),
        'idx_ret5': float((series[-1] / series[-6] - 1.0) * 100.0),
        'lu_touch': lu_touch, 'lu_now': lu_now,
        'zb_rate': (lu_touch - lu_now) / lu_touch if lu_touch else 0.0,
        'ld_now': int(m['ld_now'].sum()),
        'prem': mean_or_none(y_lu['ret_s']), 'prem_raw': mean_or_none(y_lu['ret']),
        'prem_lb': mean_or_none(y_lb['ret_s']), 'prem_lb_raw': mean_or_none(y_lb['ret']),
        'fb_prem': mean_or_none(y_fb['ret_s']),
        'fb_prem_raw': mean_or_none(y_fb['ret']),
        'jj1': mean_or_none(y_fb['lu_now'].astype(float)), 'jj2': mean_or_none(y_lb['lu_now'].astype(float)),
        'n_yday_lu': int(len(y_lu)), 'n_yday_lb': int(len(y_lb)),
        'height': height,
        'height_3d_ago': int(_boards(panel, t - 3)[ladder].max()),
        'height_5d_ago': int(_boards(panel, t - 5)[ladder].max()),
        'leaders': [(c, n) for c, n in zip(leaders['code'].tolist()[:3], leaders['name'].tolist()[:3])],
        'leader_sealed': bool(leaders['lu_now'].any()) if len(leaders) else False,
        'leader_ret_s': float(leaders['ret_s'].iloc[0]) if len(leaders) else None,
        'leader_ret': float(leaders['ret'].iloc[0]) if len(leaders) else None,
        'leader_industry': leaders['industry'].iloc[0] if len(leaders) else None,
        'amount_total': float(m['amount'].sum()),
    }
    prior_market = hist.market[mkey]
    prev_amounts = [x['amount_total'] for _, x in prior_market[-5:]]
    market['amount_vs5'] = market['amount_total'] / np.mean(prev_amounts) if prev_amounts else None

    # ---------------- industries ----------------
    m2 = m.assign(strong_i=(m['ret_s'] >= config.STRONG_SCALED).astype(int), lut_i=m['lu_touched'].astype(int),
                  lun_i=m['lu_now'].astype(int), ylu_i=m['lu_seal_y'].astype(int))
    g = m2.groupby('industry')
    ind = pd.DataFrame({'n': g.size(), 'strong': g['strong_i'].sum(), 'lu_touch': g['lut_i'].sum(),
                        'lu_now': g['lun_i'].sum(), 'mean_ret_s': g['ret_s'].mean(), 'amount': g['amount'].sum(),
                        'yday_lu': g['ylu_i'].sum(), 'max_boards': g['boards_y'].max()})
    ind = ind[ind.index != '未分类']
    ind['score'] = 2 * ind['lu_touch'] + ind['strong'] + 0.5 * ind['yday_lu']
    ind = ind.sort_values(['score', 'amount'], ascending=False)
    ind['rank'] = np.arange(1, len(ind) + 1)
    mainline = list(ind[(ind['lu_touch'] >= 2) | (ind['strong'] >= 4)].index[:3])
    df['ind_rank'] = df['industry'].map(ind['rank']).fillna(999).astype(int)
    df['ind_lu'] = df['industry'].map(ind['lu_touch']).fillna(0).astype(int)
    df['ind_strong'] = df['industry'].map(ind['strong']).fillna(0).astype(int)
    df['ind_mean_s'] = df['industry'].map(ind['mean_ret_s']).fillna(0.0)
    df['ind_yday_lu'] = df['industry'].map(ind['yday_lu']).fillna(0).astype(int)
    df['ind_max_boards'] = df['industry'].map(ind['max_boards']).fillna(0).astype(int)
    df['mainline'] = df['industry'].isin(mainline)
    df['ind_first'] = df['industry'].map(m[m['lu_touched']].groupby('industry')['lu_first'].min())

    # Leaders: highest streak (counting today's seal), sealed now, earliest touch, largest amount.
    key = ml.assign(first=np.where(ml['lu_first'] >= 0, ml['lu_first'], 999))
    key = key[(key['industry'] != '未分类') & ((key['boards_y'] >= 1) | key['lu_now'] | (key['ret_s'] >= config.STRONG_SCALED))]
    key = key.assign(streak=np.maximum(key['board_now'], key['boards_y']))
    order = key.sort_values(['industry', 'streak', 'lu_now', 'first', 'amount'],
                            ascending=[True, False, False, True, False])
    df['ind_leader'] = df.index.isin(order.groupby('industry').head(1).index)
    df['is_top_leader'] = has_band & ~df['is_st'] & (df['boards_y'] == height) & (height >= 2)
    strong5 = ml[ml['ret_s'] >= config.STRONG_SCALED]
    df['cap_core'] = df.index.isin(strong5[strong5['industry'] != '未分类'].sort_values('amount', ascending=False)
                                   .groupby('industry').head(1).index)
    s2 = ml[(ml['boards_y'] == 1) & ml['lu_now']]
    df['second_board_first'] = df.index.isin(s2.sort_values(['lu_first', 'amount'], ascending=[True, False])
                                             .groupby('industry').head(1).index)
    df['amount_rank'] = df['amount'].rank(ascending=False, method='first').astype(int)
    df['ret10_rank'] = df['ret10'].rank(ascending=False, method='first', na_option='bottom').astype(int)

    # ---------------- co-moving clusters (themes without an external concept list) ----------------
    strong_mask = (has_band & ((df['ret_s'] >= config.STRONG_SCALED) | df['lu_touched'])
                   & (df['bars_seen'] >= max(k // 2, 1))).to_numpy()
    strong_pos = np.nonzero(strong_mask)[0]
    labels = _cluster(day, df['row'].to_numpy()[strong_pos], k, config.CLUSTER_CORR[mode])
    for col, val in (('cluster', 0), ('cluster_size', 0), ('cluster_lu', 0)):
        df[col] = val
    df['cluster_new'] = np.nan
    df['cluster_leader'] = False
    clusters = []
    for lab in sorted(set(labels.tolist()) - {0}):
        pos = strong_pos[labels == lab]
        if len(pos) < config.CLUSTER_MIN_SIZE:
            continue
        sub = df.iloc[pos].assign(first=lambda x: np.where(x['lu_first'] >= 0, x['lu_first'], 999))
        sub = sub.sort_values(['is_st', 'lu_now', 'boards_y', 'first', 'amount'],
                              ascending=[True, False, False, True, False])
        clusters.append({
            'index': sub.index.to_numpy(), 'size': int(len(sub)), 'lu': int(sub['lu_touched'].sum()),
            'industries': sub['industry'].value_counts().head(3).index.tolist(),
            'new_share': float((~sub['lu_seal_y'] & (sub['ret_s_y'].fillna(0) < config.STRONG_SCALED)).mean()),
            'members': list(zip(sub['code'], sub['name'])),
        })
    clusters.sort(key=lambda c: (c['lu'], c['size']), reverse=True)
    for rank, c in enumerate(clusters, 1):
        c['rank'] = rank
        df.loc[c['index'], 'cluster'] = rank
        df.loc[c['index'], 'cluster_size'] = c['size']
        df.loc[c['index'], 'cluster_lu'] = c['lu']
        df.loc[c['index'], 'cluster_new'] = c['new_share']
        if not df.at[c['index'][0], 'is_st']:
            df.loc[c['index'][0], 'cluster_leader'] = True
        del c['index']
    df['theme'] = df['mainline'] | (df['cluster'].between(1, 2) & (df['cluster_lu'] >= 3))
    df['in_cluster'] = df['cluster'] > 0

    # Back row: strong today inside an active group, but neither its leader nor its capacity core,
    # and either sealed at least 30 minutes after the group's first seal or not sealed at all.
    strong_today = (df['ret_s'] >= 3) | df['lu_touched']
    late = df['lu_now'] & ((df['lu_first'] - df['ind_first']) >= 30)
    unsealed = ~df['lu_now'] & (df['ret_s'] < 7)
    df['back_row'] = (strong_today & ~df['ind_leader'] & ~df['cluster_leader'] & ~df['is_top_leader'] & ~df['cap_core']
                      & (df['ind_lu'] >= 2) & (late | unsealed))

    # ---------------- eligibility ----------------
    min_bars = max(1, int(round(config.scaled(k, 10, 120))))
    df['eligible'] = (has_band & (df['n_hist'] >= config.MIN_HISTORY_DAYS) & (df['bars_seen'] >= min_bars)
                      & (df['amount'] >= spec['min_amount']))
    # Intraday, a board sealed since the open has had no trading to judge and is very unlikely to fill.
    df['nominable'] = (df['eligible'] & ~df['one_word']) if mode == 'intraday' else df['eligible']

    prev_stage = hist.stages[mkey][-1][1] if hist.stages[mkey] else None
    stage, stage_ev = _stage_yangjia(mode, market, prior_market, prev_stage)

    ctx = Context()
    ctx.date, ctx.mode, ctx.key, ctx.k, ctx.t = date, mode, mkey, k, t
    ctx.spec = spec
    ctx.label, ctx.cutoff = spec['label'], spec['cutoff']
    ctx.amt = lambda at_1000, at_close: config.scaled(k, at_1000, at_close)
    ctx.stocks = df
    ctx.pool = df[df['nominable']].copy()   # copy：下面要给池子加 qc_pass 列
    if config.POSITION_GATE:
        _attach_position_gate(ctx.pool, panel, t, date)
    ctx.market = market
    ctx.industries = ind
    ctx.mainline = mainline
    ctx.clusters = clusters
    ctx.stage, ctx.stage_evidence, ctx.prev_stage = stage, stage_ev, prev_stage
    ctx.kobe_stage = _stage_kobe(market)
    ctx.history_last_date = panel.dates[t - 1]
    ctx.minute_last_bar = minutes.TIMES[k - 1]
    return ctx
