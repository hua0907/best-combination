"""Markdown record of one roundtable and the backtest summary."""
import numpy as np

from . import config
from .debaters import label, now, num, yi


def _pct(x, digits=0):
    return f'{x:.{digits}%}' if x is not None and np.isfinite(x) else '—'


def _signed(x):
    return f'{x:+.2f}%' if x is not None and np.isfinite(x) else '—'


def _row(ctx, idx):
    return next(ctx.stocks.loc[[idx]].itertuples())


def _lab(ctx, idx):
    return label(_row(ctx, idx))


def briefing_lines(ctx):
    """The moderator's market briefing, one fact per line (shared by records and the dialog window)."""
    m = ctx.market
    a5 = '站上' if m['idx_above_ma5'] else '跌破'
    a10 = '站上' if m['idx_above_ma10'] else '跌破'
    av = '' if m['amount_vs5'] is None else f'，为前5日同时点的 {m["amount_vs5"]:.2f} 倍'
    leaders = '、'.join(f'{n}({c})' for c, n in m['leaders']) or '无'
    ind = ctx.industries
    mainline = '；'.join(f'{x}（触板{int(ind.at[x, "lu_touch"])}、强势{int(ind.at[x, "strong"])}、昨日涨停'
                        f'{int(ind.at[x, "yday_lu"])}）' for x in ctx.mainline) or '无（没有行业触板≥2家或强势≥4家）'
    lines = [f'全市场 {m["n_trading"]} 只：上涨 {m["up"]} / 下跌 {m["down"]}（上涨占比 {m["breadth"]:.0%}），'
             f'等权平均 {m["ew_ret"]:+.2f}%；全A等权代理指数{a5}5日线、{a10}10日线，近5日 {m["idx_ret5"]:+.1f}%',
             f'成交 {yi(m["amount_total"])}{av}；触板 {m["lu_touch"]} 家、现封 {m["lu_now"]} 家，炸板率 {m["zb_rate"]:.0%}；'
             f'跌停 {m["ld_now"]} 家',
             f'昨日涨停 {m["n_yday_lu"]} 家今日均 {num(m["prem_raw"])}%；昨日首板晋级率 {_pct(m["jj1"])}；'
             f'昨日连板 {m["n_yday_lb"]} 家今日均 {num(m["prem_lb_raw"])}%、晋级率 {_pct(m["jj2"])}',
             f'最高 {m["height"]} 板：{leaders}，今日 {num(m["leader_ret"])}%',
             f'情绪阶段（炒股养家 Skill 用的市场通用分段）：**{ctx.stage}**（前一日 {ctx.prev_stage or "—"}）；'
             f'依据：{"；".join(ctx.stage_evidence)}',
             f'周期（92科比四阶段）：**{ctx.kobe_stage}**',
             f'主线行业（通达信行业，按触板×2+强势家数排序）：{mainline}']
    for c in ctx.clusters[:3]:
        members = '、'.join(n for _, n in c['members'][:6]) + ('…' if c['size'] > 6 else '')
        lines.append(f'盘中联动簇 #{c["rank"]}：{c["size"]} 只、触板 {c["lu"]}，主要行业 {"/".join(c["industries"])}，'
                     f'新面孔 {c["new_share"]:.0%}：{members}')
    if not ctx.clusters:
        lines.append('盘中联动簇：无（强势股不足，或分时走势互不相关）')
    return lines


def rules_text():
    return (f'每位辩手只按自己的框架选票（最多 {config.NOMINATIONS_PER_DEBATER} 只，也可以不选），不互相投票；主持人只统计票数：'
            f'被 {config.CONSENSUS_MIN} 位及以上选中的是共识票，按票数从多到少排；没有共识票就空仓')


def buy_way(ctx, ch):
    if ctx.mode != 'intraday':
        return '次日开盘价买入（一字开盘则按涨停价排队）'
    return f'{ctx.cutoff} 已封涨停，按涨停价排队（开板才成交）' if ch.sealed else f'{ctx.cutoff} 价格买入'


def transcript(db, outcomes=None, baseline=None, outcome_note=''):
    ctx = db.ctx
    names = {d.key: d.name for d in db.debaters}
    auction = '（15:00 那根含收盘集合竞价，收盘价已定）' if ctx.mode == 'eod' else ''
    L = [f'# {ctx.date} · {ctx.label} 圆桌', '',
         f'> 决策时点 {ctx.date} {ctx.cutoff}。可见数据：当日 09:31–{ctx.minute_last_bar} 的1分钟线{auction}、截至 '
         f'{ctx.history_last_date} 的日线、开盘前已知的昨收与涨跌停价。流程：{rules_text()}。买入规则：{ctx.spec["entry"]}。'
         f'辩手是按各自 Skill 规则化的风格模型，不是本人；文末“事后结果”在开会时不可见。', '']
    L += ['## 主持人盘面简报', ''] + [f'- {x}' for x in briefing_lines(ctx)] + ['']

    L += ['## 各位辩手的选票', '']
    for d, noms, abstain in db.opening:
        v = db.views[d.key]
        L.append(f'### {d.name}（{d.school}）· {"本场不参加" if v.mode_out else f"买入意愿 {v.exposure:.0%}"}')
        L.append(f'- 判断：{v.stance}')
        if noms:
            for i, (idx, model, why, _) in enumerate(noms):
                L.append(f'- 选票{"①②③"[i]} **{_lab(ctx, idx)}**｜{model}：{why}')
        else:
            L.append(f'- 不选：{abstain}')
        L.append('')

    L += ['## 主持人统计（唱票）', '']
    ranked = sorted(db.choices.values(), key=lambda ch: (-ch.count, -ch.firsts, ch.rank or 999, ch.code))
    if ranked:
        L += ['| 股票 | 票数 | 选它的辩手（★=第一选择） | 开会时 |', '|---|---|---|---|']
        for ch in ranked:
            who = '、'.join(f'{names[k]}{"★" if order == 1 else ""}' for k, _, _, _, order in ch.choosers)
            L.append(f'| {_lab(ctx, ch.idx)} | {ch.count} | {who} | {now(_row(ctx, ch.idx))} |')
    else:
        L.append('（没有辩手选票）')
    L.append('')

    L += [f'## 共识票（{config.CONSENSUS_MIN} 位及以上选中，按票数排）', '']
    if db.picks:
        L += ['| 名次 | 股票 | 票数 | 选它的辩手 | 买入方式 |', '|---|---|---|---|---|']
        for ch in db.picks:
            L.append(f'| {ch.rank} | {_lab(ctx, ch.idx)} | {ch.count} | {"、".join(names[k] for k in ch.keys)} | '
                     f'{buy_way(ctx, ch)} |')
    else:
        L.append(f'- {db.note}')
    L.append(f'- 圆桌仓位意见：{db.voters} 位辩手平均买入意愿 {db.position:.0%}，{db.flat_votes}/{db.voters} 位不选')
    if db.picks:
        L += ['', '### 共识票的理由', '']
        for ch in db.picks:
            L.append(f'{ch.rank}. **{_lab(ctx, ch.idx)}**（{ch.count} 票）')
            L += [f'   - {names[k]}「{model}」：{why}' for k, model, why, _, _ in ch.choosers]
    L.append('')

    L += ['---', '', '## 事后结果（开会时不可见，仅供回测统计）', '']
    if outcomes is None:
        L.append(outcome_note or '待下一交易日数据。')
    else:
        day_txt = f'当日（{ctx.date}）' if ctx.mode == 'intraday' else f'次日（{outcome_note}）'
        L += [f'买入日：{day_txt}；卖出：买入后的下一个交易日（A股 T+1），按当天最高价（最佳）、高低均价（平均）、最低价（最差）三种算。'
              '涨幅相对前一日收盘，收益相对买入价。', '']
        if not db.picks:
            L.append('本场没有共识票，空仓。')
        else:
            L += ['| 名次 | 股票 | 票数 | 成交 | 买入价 | 买入时涨幅 | 买入日收盘涨幅 | 买入日收盘收益 | 卖出日 | 次日最高涨幅 | '
                  '次日最低涨幅 | 次日均价涨幅 | 最高价卖出收益 | 均价卖出收益 | 最低价卖出收益 |', '|---' * 15 + '|']
        for ch in db.picks:
            o = outcomes.loc[ch.idx] if ch.idx in outcomes.index else None
            if o is None or not o['filled']:
                why = '涨停排队未开板' if (o is not None and o['queued']) else '无成交数据'
                L.append(f'| {ch.rank} | {_lab(ctx, ch.idx)} | {ch.count} | 未成交（{why}） |' + ' — |' * 11)
                continue
            fill = '排板成交' if o['queued'] else '成交'
            sell = o['sell_date'] if isinstance(o['sell_date'], str) else '待数据'
            if o.get('t1_stuck') == True:  # noqa: E712
                sell += '（一字跌停，实际卖不掉，按跌停价计）'
            L.append(f'| {ch.rank} | {_lab(ctx, ch.idx)} | {ch.count} | {fill} | {o["entry"]:.2f} | {_signed(o["buy_chg"])} | '
                     f'{_signed(o["close_chg"])} | {_signed(o["ret_close"])} | {sell} | {_signed(o["next_high_chg"])} | '
                     f'{_signed(o["next_low_chg"])} | {_signed(o["next_mid_chg"])} | {_signed(o["t1_best"])} | '
                     f'{_signed(o["t1_mid"])} | {_signed(o["t1_worst"])} |')
        if baseline:
            L += ['', f'对照：同日全部可选股 {baseline["all_n"]} 只，买入日收盘收益 {_signed(baseline["all_ret"])}、'
                      f'次日均价卖出 {_signed(baseline["all_t1"])}；被选中的 {baseline["pool_n"]} 只，买入日收盘收益 '
                      f'{_signed(baseline["pool_ret"])}、次日均价卖出 {_signed(baseline["pool_t1"])}']
    L.append('')
    return '\n'.join(L)


# ---------------------------------------------------------------- summary

STAT_ROWS = [('会议数（T+1 卖出日已有数据）', 'days', '{:.0f}'), ('其中有票的会议', 'days_with', '{:.0f}'),
             ('股票-次数（含待卖出的）', 'n', '{:.0f}'),
             ('成交数', 'filled', '{:.0f}'), ('成交率', 'fill_rate', 'pct'),
             ('买入时平均涨幅', 'buy_chg', '{:+.2f}%'), ('买入日收盘收益（平均）', 'ret_close', '{:+.2f}%'),
             ('买入日收盘胜率', 'win', 'pct'),
             ('T+1 最高价卖出：胜率', 'best_win', 'pct'), ('T+1 最高价卖出：平均收益', 'best', '{:+.2f}%'),
             ('T+1 均价卖出：胜率', 'mid_win', 'pct'), ('T+1 均价卖出：平均收益', 'mid', '{:+.2f}%'),
             ('T+1 最低价卖出：胜率', 'worst_win', 'pct'), ('T+1 最低价卖出：平均收益', 'worst', '{:+.2f}%'),
             ('每场组合平均收益（T+1 均价卖出）', 'day_ret', '{:+.3f}%'), ('每场组合收益合计（单利）', 'day_sum', '{:+.2f}%'),
             ('组合收益为正的会议占比', 'day_pos', 'pct')]


def _fmt(v, f):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '—'
    return f'{v:.1%}' if f == 'pct' else f.format(v)


def stats(frame, days=None):
    """frame: one row per stock and meeting with outcome columns. days: the meetings whose T+1 sale day is in the data;
    with them a portfolio is formed per meeting (one stake split equally among its stocks, an unfilled share stays
    cash, a meeting without stocks is a cash meeting)."""
    f = frame[frame['filled'] == True]  # noqa: E712
    t = f[np.isfinite(f['t1_mid'].astype(float))] if len(f) else f
    out = {'days': frame['date'].nunique() if days is None else len(days), 'n': len(frame), 'filled': len(f),
           'fill_rate': len(f) / len(frame) if len(frame) else np.nan}
    if len(f):
        out.update({'buy_chg': f['buy_chg'].mean(), 'ret_close': f['ret_close'].mean(), 'win': f['win'].mean()})
    if len(t):
        for key, col in (('best', 't1_best'), ('mid', 't1_mid'), ('worst', 't1_worst')):
            out[key] = t[col].mean()
            out[f'{key}_win'] = (t[col] > 0).mean()
    if days is not None:
        g = frame[frame['date'].isin(days)]
        out['days_with'] = g['date'].nunique()
        r = g['t1_mid'].astype(float).where(g['filled'] == True, 0.0).fillna(0.0)  # noqa: E712
        daily = r.groupby(g['date']).mean().reindex(days, fill_value=0.0)
        out.update({'day_ret': daily.mean(), 'day_sum': daily.sum(), 'day_pos': (daily > 0).mean()})
    return out


def table(columns):
    """columns: [(title, stats dict)]."""
    L = ['| 指标 | ' + ' | '.join(t for t, _ in columns) + ' |', '|---|' + '---|' * len(columns)]
    for title, key, f in STAT_ROWS:
        if all(key not in s for _, s in columns):
            continue
        L.append(f'| {title} | ' + ' | '.join(_fmt(s.get(key), f) for _, s in columns) + ' |')
    return '\n'.join(L)


def group_table(frame, by, title, min_n=1, order=None):
    rows = []
    for key, g in frame.groupby(by, sort=False):
        s = stats(g)
        if s['n'] >= min_n:
            rows.append((key, s))
    rows.sort(key=lambda x: (order.index(x[0]) if order and x[0] in order else len(order or []), -x[1]['n']))
    L = [f'| {title} | 次数 | 成交 | 买入时涨幅 | 买入日收盘收益 | T+1 均价卖出胜率 | T+1 均价卖出收益 | T+1 最高价卖出 | '
         f'T+1 最低价卖出 |', '|---' * 9 + '|']
    for key, s in rows:
        L.append(f'| {key} | {s["n"]} | {s["filled"]} | {_fmt(s.get("buy_chg"), "{:+.2f}%")} | '
                 f'{_fmt(s.get("ret_close"), "{:+.2f}%")} | {_fmt(s.get("mid_win"), "pct")} | {_fmt(s.get("mid"), "{:+.2f}%")} | '
                 f'{_fmt(s.get("best"), "{:+.2f}%")} | {_fmt(s.get("worst"), "{:+.2f}%")} |')
    return '\n'.join(L)
