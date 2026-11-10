"""One roundtable meeting on a given trading day and time, and questions asked about it afterwards.

run_day.py writes a meeting's record; the dialog window (chat_app.py) holds meetings and answers questions
about them. Everything a debater says comes from the meeting's Context (data up to the cutoff); what
happened after the decision is kept apart and only shown when asked for.
"""
import re
import time
from dataclasses import dataclass

import numpy as np

from . import audit as pit
from . import config, context, evaluate, market_cache, minutes, report, roundtable, store
from .debaters import all_debaters, amt, yi

T1_COLUMNS = ['next_high_chg', 'next_low_chg', 'next_mid_chg', 't1_best', 't1_mid', 't1_worst', 't1_stuck']


def full_market_days(days=None):
    """D:/data days known to cover the whole market (read from the market caches); unknown days count as full."""
    days = store.data_dates() if days is None else days
    known = {}
    for path in sorted(config.CACHE_DIR.glob('market_*.json')):
        hhmm = path.stem.split('_', 1)[1]
        for d, e in market_cache.load(f'{hhmm[:2]}:{hhmm[2:]}').items():
            known[d] = e['codes']
    return [d for d in days if known.get(d, config.FULL_MARKET_MIN_CODES) >= config.FULL_MARKET_MIN_CODES]


class Data:
    """Daily panel, ex-rights records and the list of D:/data days, loaded once per process."""

    def __init__(self):
        self.days = store.data_dates()
        self.panel = store.build_panel()
        self.xdxr = store.load_xdxr()


def _summary(day, panel, t, cuts):
    j = day.panel_j
    return minutes.day_summary(day, panel.ref[t, j], panel.pct[t, j], panel.limit_up[t, j], panel.limit_down[t, j],
                               cuts)


def history_for(date, spec, data, log=print):
    """The History the backtest would hold at `date` for this meeting time (market readings are cached)."""
    panel, key = data.panel, spec['key']
    cuts = (spec['cutoff_index'],) if spec['mode'] == 'intraday' else ()
    prior = [d for d in data.days if d < date]
    recent = set(prior[-6:])
    cached = market_cache.load(key)
    stale = next((i for i, d in enumerate(prior)
                  if d not in cached or cached[d]['size'] != market_cache.file_size(d)), len(prior))
    if stale < len(prior):  # a missing or changed day invalidates itself and every later reading
        cached = {d: e for d, e in cached.items() if d < prior[stale]}
    hist = context.History()
    for d in prior[:stale]:
        if 'metrics' in cached[d]:
            hist.market[key].append((d, cached[d]['metrics']))
            hist.stages[key].append((d, cached[d]['stage']))
    todo = prior[stale:]
    if todo:
        log(f'第一次在 {key} 开会：从 {todo[0]} 起逐日计算 {len(todo)} 个交易日的盘面历史（之后会缓存）…')
    t0 = time.time()
    for n, d in enumerate(todo, 1):
        day = minutes.load_day(d, panel)
        t = panel.index_of(d)
        if day.size >= config.FULL_MARKET_MIN_CODES:
            ctx = context.build(panel, t, d, spec, day, hist, store.float_shares_asof(data.xdxr, panel.codes, d))
            hist.market[key].append((d, ctx.market))
            hist.stages[key].append((d, ctx.stage))
            cached[d] = market_cache.entry(d, day.size, ctx.market, ctx.stage)
        else:
            cached[d] = market_cache.entry(d, day.size)
        if d in recent:
            hist.summaries[d] = _summary(day, panel, t, cuts)
        if n % 10 == 0 or n == len(todo):
            log(f'  已算到 {d}（{n}/{len(todo)}，{time.time() - t0:.0f} 秒）')
            market_cache.save(key, cached)
    for d in sorted(recent - set(hist.summaries)):
        hist.summaries[d] = _summary(minutes.load_day(d, panel), panel, panel.index_of(d), cuts)
    return hist, cached


@dataclass
class Meeting:
    date: str
    spec: dict
    ctx: object
    db: object
    debaters: list
    out: object = None          # outcomes after the decision; None when blind or not available yet
    base: dict = None
    note: str = ''
    complete: bool = True
    blind: bool = False
    audit_passed: object = None
    path: object = None

    @property
    def label(self):
        return f'{self.date} {self.spec["label"]}'


def convene(data, date, time_str='10:00', blind=False, audit=False, log=print):
    spec = config.meeting(time_str)
    if date not in data.days:
        raise ValueError(f'D:/data 里没有 {date}.csv（现有 {data.days[0]} – {data.days[-1]}）')
    review = spec['mode'] == 'eod'
    early = config.too_early(date) if review else ''
    if early:  # before reading anything
        raise ValueError(f'{early}：收盘后复盘 {config.CLOSE_REVIEW} 以后再开（盘中会议可以开 09:31–14:59 的任意分钟）')
    panel = data.panel
    hist, cached = history_for(date, spec, data, log)
    day = minutes.load_day(date, panel)
    if day.size < config.FULL_MARKET_MIN_CODES:
        raise ValueError(f'{date} 的分钟数据只有 {day.size} 只股票，不是全市场数据，不能开会')
    open_ = minutes.unfinished(day)
    if review and open_:
        raise ValueError(f'{open_}：收盘后复盘只用收完的数据，换成收盘后的完整文件再开')
    last_bar = int(day.real.any(axis=0).nonzero()[0].max())
    if last_bar < spec['cutoff_index'] - 1:
        raise ValueError(f'{date} 的分钟数据只到 {minutes.TIMES[last_bar]}，不能在 {spec["cutoff"]} 开会')
    complete = not open_
    t = panel.index_of(date)
    debaters = all_debaters()
    ctx = context.build(panel, t, date, spec, day, hist, store.float_shares_asof(data.xdxr, panel.codes, date))
    db = roundtable.run(ctx, debaters)
    if complete and not any(d >= date for d in cached):  # extend the cached chain with a finished day
        cached[date] = market_cache.entry(date, day.size, ctx.market, ctx.stage)
        market_cache.save(spec['key'], cached)

    out = base = None
    note = ''
    if blind:
        note = '已按盲测省略：只看决策，不看之后发生了什么。'
    elif not complete:
        note = f'当日还没收完（{open_}），事后结果要等收盘后（{config.CLOSE_REVIEW} 以后）的完整数据。'
    elif spec['mode'] == 'intraday':
        out = evaluate.outcomes_for(ctx, panel, day, 'intraday')
    else:
        nxt = panel.dates[t + 1] if t + 1 < len(panel.dates) else None
        nday = minutes.load_day(nxt, panel) if nxt in data.days else None
        nxt_open = minutes.unfinished(nday) if nday is not None else ''
        if nday is None:
            note = f'待 {date} 的下一个交易日数据（D:/data 截至 {data.days[-1]}）。'
        elif nxt_open:
            note = f'买入日 {nxt} 还没收完（{nxt_open}），事后结果要等它收盘后（{config.CLOSE_REVIEW} 以后）的完整数据。'
        else:
            out = evaluate.outcomes_for(ctx, panel, nday, 'eod')
            note = nxt
    if out is not None:  # add the T+1 sale: the trading day after the buy
        t_buy = t if spec['mode'] == 'intraday' else t + 1
        t1 = evaluate.next_day(panel, t_buy, ctx.stocks.loc[out.index, 'j'].to_numpy(), out['entry'].to_numpy())
        t1.index = out.index
        sale = panel.dates[t_buy + 1] if t_buy + 1 < len(panel.dates) else None
        # Only the latest D:/data file can still be filling up; a sale day that has not closed is not a sale yet.
        if sale is not None and sale == data.days[-1] and minutes.unfinished(minutes.load_day(sale, panel)):
            t1[T1_COLUMNS] = np.nan
            t1['sell_date'] = None
        out = out.join(t1)
        base = evaluate.baseline(ctx, db, out)
    m = Meeting(date=date, spec=spec, ctx=ctx, db=db, debaters=debaters, out=out, base=base, note=note,
                complete=complete, blind=blind)
    if audit:
        m.audit_passed = pit.check(panel, t, date, spec, day, hist, data.xdxr, debaters, db)
    return m


def write_record(m, out_dir):
    path = out_dir / f'{m.date}_{m.spec["key"].replace(":", "")}.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.transcript(m.db, m.out, m.base, outcome_note=m.note), encoding='utf-8')
    m.path = path
    return path


# ---------------------------------------------------------------- questions about a meeting

def row(m, idx):
    return next(m.ctx.stocks.loc[[idx]].itertuples())


def find_stocks(m, text):
    """Stocks named in `text`, by 6-digit code or by name (longest names first), as Context indexes."""
    s = m.ctx.stocks
    found = []
    for code in re.findall(r'(?<!\d)(\d{6})(?!\d)', text):
        found += [i for i in s.index[s['code'] == code] if i not in found]
    names = sorted(((i, n) for i, n in zip(s.index, s['name']) if isinstance(n, str) and len(n) >= 2 and n in text),
                   key=lambda x: -len(x[1]))
    taken = []
    for i, n in names:
        if i not in found and not any(n in other for other in taken):
            found.append(i)
            taken.append(n)
    return found


def pool_status(m, idx):
    """Why a visible stock is not in the nomination pool ('' when it is)."""
    r, ctx = row(m, idx), m.ctx
    if r.nominable:
        return ''
    if not r.has_band:
        return '没有涨跌幅限制（新股前5个交易日等），不进入可选池'
    if r.n_hist < config.MIN_HISTORY_DAYS:
        return f'上市不足 {config.MIN_HISTORY_DAYS} 个交易日，不进入可选池'
    if r.amount < ctx.spec['min_amount']:
        return f'截至 {ctx.cutoff} {amt(r)}，低于可选池门槛 {yi(ctx.spec["min_amount"])}'
    if r.one_word:
        return f'{ctx.cutoff} 前一直封在涨停价（一字），盘中无法判断承接，不进入可选池'
    return '截止前成交的分钟数太少，不进入可选池'


def opinion(m, debater, idx):
    """What one debater says about one stock at this meeting: (kind, model or tag, text).

    kind: nominate (one of its own picks) / support (also fits its framework) / hard / soft (its risk rules) /
    neutral. This is only the debater's view of the stock; since v2.0 nobody's view changes the verdict.
    """
    db = m.db
    view = db.views[debater.key]
    ch = db.choices.get(idx)
    if ch is not None:
        mine = next((x for x in ch.choosers if x[0] == debater.key), None)
        if mine is not None:
            return 'nominate', mine[1], mine[2]
    cands = db.cands[debater.key]
    if idx in cands.index:
        return 'support', cands.at[idx, 'model'], cands.at[idx, 'why']
    rv = debater.objection(m.ctx, row(m, idx), view)
    if rv is not None:
        return rv.kind, rv.tag, rv.text
    if view.mode_out:
        return 'neutral', '', f'这一场不参加：{view.flat_reason}'
    if view.flat_reason:
        return 'neutral', '', f'今天空仓：{view.flat_reason}'
    if view.max_nominations == 0:
        return 'neutral', '', '本阶段不新增'
    return 'neutral', '', '不在本框架的条件内（本框架看：' + '、'.join(debater.models) + '）'
