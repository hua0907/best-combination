"""Run the nine-debater consensus roundtable over every full-market day in D:/data, point-in-time.

    python -X utf8 run_backtest.py                       # all full-market days, both modes
    python -X utf8 run_backtest.py --start 2026-06-01 --end 2026-06-05 --out output-test

Order inside one calendar day T (this order is the point-in-time guarantee):
  1. load T's minute bars
  2. settle the eod decisions of the previous day with T's bars (outcomes only)
  3. build each mode's context from panel rows [:t], T's pre-open facts and T's bars before the cutoff,
     run the roundtable and record every debater's choices and the consensus picks
  4. only then compute intraday outcomes from T's bars after the cutoff
  5. append T to the history (whole-day summary, market metrics, stage) for later days
Outcomes include the A-share T+1 sale (the trading day after the buy: its high, low and midpoint), read from the daily
panel after the decision is recorded. Every audit_every-th decision is re-run with all data after the cutoff replaced by
random numbers; the decision must come out identical. A run over all days also writes the market-history caches
(cache/market_1000.json, market_1510.json) that run_day.py reuses. Every day read must be a finished day
(minutes.unfinished): a file still filling up would put a pre-close review into pending.csv and partial prices into
the outcomes, so it stops the run.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine import audit as pit  # noqa: E402
from engine import config, context, evaluate, market_cache, minutes, report, roundtable, store  # noqa: E402
from engine.debaters import all_debaters  # noqa: E402

OUTCOME_COLS = ['filled', 'queued', 'entry', 'buy_chg', 'close_chg', 'ret_close', 'ret_close_s', 'max_up', 'max_up_s',
                'max_dn', 'hit_lu', 'sealed', 'y2', 'win', 'sell_date', 'next_high_chg', 'next_low_chg', 'next_mid_chg',
                't1_best', 't1_mid', 't1_worst', 't1_stuck']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def outcome_dict(out, idx):
    if idx not in out.index:
        return {c: (False if c in ('filled', 'queued', 't1_stuck') else None if c == 'sell_date' else np.nan)
                for c in OUTCOME_COLS}
    o = out.loc[idx]
    return {c: o[c] for c in OUTCOME_COLS}


def settle(rec, db, out, out_dir, panel, next_date=None):
    """Record outcomes (buy day and T+1 sale) and write the record. Called only after the decision was recorded."""
    ctx = db.ctx
    s = ctx.stocks
    names = {d.key: d.name for d in db.debaters}
    t_buy = ctx.t if ctx.mode == 'intraday' else ctx.t + 1
    t1 = evaluate.next_day(panel, t_buy, s.loc[out.index, 'j'].to_numpy(), out['entry'].to_numpy())
    t1.index = out.index
    out = out.join(t1)
    for ch in db.picks:
        rec['picks'].append({
            'date': ctx.date, 'mode': ctx.mode, 'decided_at': f'{ctx.date} {ctx.cutoff}', 'rank': ch.rank,
            'code': ch.code, 'name': ch.name, 'industry': s.at[ch.idx, 'industry'], 'count': ch.count,
            'firsts': ch.firsts, 'choosers': '、'.join(names[k] for k in ch.keys),
            'models': '、'.join(f'{names[k]}:{m}' for k, m, *_ in ch.choosers), 'sealed_at_decision': ch.sealed,
            'chg_at_decision': float(s.at[ch.idx, 'ret']), 'position': db.position, 'flat_votes': db.flat_votes,
            'voters': db.voters, 'stage': ctx.stage, 'kobe_stage': ctx.kobe_stage, 'buy_date': panel.dates[t_buy],
            **outcome_dict(out, ch.idx)})
    for d, noms, _ in db.opening:
        for order, (idx, model, _, score) in enumerate(noms, 1):
            ch = db.choices[idx]
            rec['noms'].append({'date': ctx.date, 'mode': ctx.mode, 'debater': d.name, 'model': model,
                                'debater_model': f'{d.name}:{model}', 'order': order, 'code': ch.code, 'name': ch.name,
                                'fit': score, 'count': ch.count, 'picked': ch.rank > 0, **outcome_dict(out, idx)})
    elig = out.assign(date=ctx.date)
    rec['all'].append(elig[['date'] + OUTCOME_COLS])
    rec['pool'].append(elig.loc[[i for i in db.choices if i in elig.index], ['date'] + OUTCOME_COLS])
    e = s[s['eligible'] & ~s['lu_now']].sort_values(['ret_s', 'amount'], ascending=False).head(3)
    rec['mom'].append(elig.loc[[i for i in e.index if i in elig.index], ['date'] + OUTCOME_COLS])
    base = evaluate.baseline(ctx, db, out)
    m = ctx.market
    rec['days'].append({'date': ctx.date, 'mode': ctx.mode, 'stage': ctx.stage, 'kobe_stage': ctx.kobe_stage,
                        'position': db.position, 'flat_votes': db.flat_votes, 'n_choices': len(db.choices),
                        'n_picks': len(db.picks), 'n_eligible': int(s['eligible'].sum()), 'lu_touch': m['lu_touch'],
                        'lu_now': m['lu_now'], 'zb_rate': m['zb_rate'], 'ld_now': m['ld_now'], 'prem_raw': m['prem_raw'],
                        'height': m['height'], 'breadth': m['breadth'], 'mainline': '、'.join(ctx.mainline),
                        **{f'base_{k}': v for k, v in base.items()}})
    text = report.transcript(db, out, base, outcome_note=next_date or '')
    path = out_dir / ctx.mode / f'{ctx.date}.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def settled(days, dates, mode):
    """Meetings whose T+1 sale day is in the data: 10:00 buys that day, the 15:10 review buys the next day."""
    pos = {d: i for i, d in enumerate(dates)}
    lag = 1 if mode == 'intraday' else 2
    return [d for d in days if pos[d] + lag < len(dates)]


def write_summary(out_dir, recs, modes, meta):
    L = ['# 圆桌回测汇总', '',
         f'- 生成时间：{meta["generated"]}；引擎版本 {config.VERSION}',
         f'- 数据：{config.DATA_DIR} 共 {meta["data_days"]} 个交易日文件；决策日 = 全市场覆盖（≥{config.FULL_MARKET_MIN_CODES} 只）的 '
         f'{meta["window_days"]} 天（{meta["first"]} – {meta["last"]}）；之前的不完整日只作历史',
         f'- 流程：{report.rules_text()}',
         f'- 买入：10:00 场按当天 10:00 价格；收盘后 {config.CLOSE_REVIEW} 复盘按次一交易日开盘价；买入时已在涨停价的只能按涨停价排队，'
         '之后开板才算成交',
         '- 卖出（A股 T+1）：买入后的下一个交易日，按当天最高价（最佳，事后才知道）、高低均价（平均）、最低价（最差）三种算；'
         '另列买入时涨幅和买入日收盘收益。已扣除除权除息影响，未扣交易费用（来回约 0.1%）；卖出日一字跌停（实际卖不掉）的'
         '按跌停价计入并单独计数',
         '- 停牌：没有成交量的分钟线和日线不算成交（TDX 给停牌股写入全天零成交的平线），停牌当天不可见、不进日线、'
         '买不进也卖不出',
         '- 组合：每场一份资金平分给这场的共识票（没成交的那份留作现金），没有共识票的会议算现金；只统计 T+1 卖出日已有数据的会议',
         f'- 防偷看审计：{meta["audit_pass"]}/{meta["audit_total"]} 个抽查决策在“截止时点之后全部换成随机数”时结论完全一致',
         '']
    for mode in modes:
        r = recs[mode]
        label = config.MODES[mode]['label']
        days = [d['date'] for d in r['days']]
        sdays = settled(days, meta['dates'], mode)
        all_ = pd.concat(r['all'], ignore_index=True)
        pool = pd.concat(r['pool'], ignore_index=True)
        mom = pd.concat(r['mom'], ignore_index=True)
        noms = pd.DataFrame(r['noms'])
        picks = pd.DataFrame(r['picks'], columns=['date', 'rank', 'count', 'stage', 'kobe_stage'] + OUTCOME_COLS) \
            if not r['picks'] else pd.DataFrame(r['picks'])
        per = picks.groupby('date').size().reindex(days, fill_value=0)
        dist = '；'.join(f'{k} 只 {int((per == k).sum())} 场' for k in range(int(per.max()) + 1) if (per == k).any())
        columns = [(f'共识票（≥{config.CONSENSUS_MIN} 位选中）', report.stats(picks, sdays)),
                   ('选票池（任何辩手选中的票）', report.stats(pool, sdays)),
                   ('全部可选股', report.stats(all_, sdays)),
                   ('动量前3（未封板涨幅最大）', report.stats(mom, sdays))]
        stuck = int(((picks['filled'] == True) & (picks['t1_stuck'] == True)).sum()) if len(picks) else 0  # noqa: E712
        L += [f'## {label}', '', report.table(columns), '', f'每场共识票数：{dist}（没有共识票的会议空仓）', '',
              f'卖出日一字跌停（实际卖不掉）的共识票：{stuck} 笔，T+1 收益已按跌停价计入', '']
        if len(picks):
            picks['votes_bucket'] = picks['count'].clip(upper=4).map({2: '2 票', 3: '3 票', 4: '4 票及以上'}).fillna('—')
            picks['rank_bucket'] = picks['rank'].clip(upper=4).map({1: '第1名', 2: '第2名', 3: '第3名', 4: '第4名及以后'})
            L += ['### 按票数（共识票）', '', report.group_table(picks, 'votes_bucket', '票数',
                                                         order=['4 票及以上', '3 票', '2 票']), '',
                  '### 按名次（共识票）', '', report.group_table(picks, 'rank_bucket', '名次',
                                                         order=['第1名', '第2名', '第3名', '第4名及以后']), '',
                  '### 按情绪阶段（共识票）', '', report.group_table(picks, 'stage', '阶段'), '',
                  '### 按92科比周期（共识票）', '', report.group_table(picks, 'kobe_stage', '周期'), '']
            buckets = ['≤30%', '30–45%', '45–60%', '>60%']
            picks['position_bucket'] = pd.cut(picks['position'], [-0.01, .3, .45, .6, 1.0], labels=buckets).astype(str)
            L += ['### 按圆桌仓位意见（辩手的平均买入意愿，共识票）', '',
                  report.group_table(picks, 'position_bucket', '仓位意见', order=buckets), '']
        L += ['### 按辩手（各自的全部选票，不论是否成了共识）', '', report.group_table(noms, 'debater', '辩手'), '',
              '### 按选票模式（≥5次）', '', report.group_table(noms, 'debater_model', '辩手:模式', min_n=5), '']
    (out_dir / 'summary.md').write_text('\n'.join(L), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start')
    ap.add_argument('--end')
    ap.add_argument('--modes', default='intraday,eod')
    ap.add_argument('--audit-every', type=int, default=10)
    ap.add_argument('--out', default=str(config.OUTPUT_DIR))
    args = ap.parse_args()
    modes = [m for m in args.modes.split(',') if m in config.MODES]
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = config.ROUNDTABLE_ROOT / out_dir

    t0 = time.time()
    panel = store.build_panel()
    xdxr = store.load_xdxr()
    debaters = all_debaters()
    data_days = sorted(p.stem for p in config.DATA_DIR.glob('*.csv'))
    missing = [d for d in data_days if d not in panel.date_index]
    assert not missing, f'D:/data days missing from the daily panel: {missing}'
    open_ = minutes.unfinished(minutes.load_day(data_days[-1], panel)) if data_days else ''
    if open_:  # the latest file is the one that can still be filling up; fail before any work
        raise SystemExit(f'{open_}。回测只用收完的交易日：{config.CLOSE_REVIEW} 以后换成收盘后的完整文件再跑，'
                         '或先把这个文件移出 D:/data。')
    out_dir.mkdir(parents=True, exist_ok=True)
    first_window = next((i for i, d in enumerate(data_days) if args.start is None or d >= args.start), len(data_days))
    days = data_days[max(first_window - 6, 0):]  # a few days before the window feed the history only
    if args.end:
        days = [d for d in days if d <= args.end]

    hist = context.History()
    recs = {m: {'picks': [], 'noms': [], 'days': [], 'all': [], 'pool': [], 'mom': []} for m in modes}
    keys = {m: config.MODES[m]['key'] for m in modes}
    cuts = tuple(config.MODES[m]['cutoff_index'] for m in modes if config.MODES[m]['mode'] == 'intraday')
    market_days = {keys[m]: {} for m in modes}
    pending = None
    audits, asserts = [], []
    window = []
    rng = np.random.default_rng(20261003)
    for date in days:
        t = panel.index_of(date)
        assert t >= 1 and panel.dates[t - 1] < date
        day = minutes.load_day(date, panel)
        open_ = minutes.unfinished(day)
        if open_:
            raise SystemExit(f'{open_}。回测只用收完的交易日：换成收盘后的完整文件再跑。')
        full = day.size >= config.FULL_MARKET_MIN_CODES
        in_window = full and (args.start is None or date >= args.start)

        if pending is not None:  # previous day's eod decisions, settled with today's bars
            db_prev, ctx_prev = pending
            assert panel.dates[ctx_prev.t + 1] == date
            out = evaluate.outcomes_for(ctx_prev, panel, day, 'eod')
            settle(recs['eod'], db_prev, out, out_dir, panel, next_date=date)
            pending = None

        if in_window:
            window.append(date)
            fs = store.float_shares_asof(xdxr, panel.codes, date)
            for mode in modes:
                assert all(d < date for d in hist.summaries)
                assert all(d < date for d, _ in hist.market[keys[mode]])
                ctx = context.build(panel, t, date, mode, day, hist, fs)
                assert ctx.history_last_date < date and ctx.minute_last_bar == minutes.TIMES[ctx.k - 1]
                asserts.append({'date': date, 'mode': mode, 'history_last_date': ctx.history_last_date,
                                'minute_last_bar': ctx.minute_last_bar,
                                'latest_summary': max(hist.summaries) if hist.summaries else None})
                db = roundtable.run(ctx, debaters)
                if args.audit_every and (len(window) - 1) % args.audit_every == 0:
                    same = pit.check(panel, t, date, config.MODES[mode], day, hist, xdxr, debaters, db, rng)
                    audits.append({'date': date, 'mode': mode, 'identical': same})
                    print(f'  audit {date} {mode}: {"identical" if same else "DIFFERENT"}', flush=True)
                if mode == 'intraday':
                    out = evaluate.outcomes_for(ctx, panel, day, 'intraday')
                    settle(recs[mode], db, out, out_dir, panel)
                else:
                    pending = (db, ctx)
                hist.market[ctx.key].append((date, ctx.market))
                hist.stages[ctx.key].append((date, ctx.stage))
                market_days[ctx.key][date] = market_cache.entry(date, day.size, ctx.market, ctx.stage)
            print(f'{date} done ({time.time() - t0:.0f}s): ' + '; '.join(
                f'{m} {hist.stages[keys[m]][-1][1]}' for m in modes), flush=True)
        else:
            for key in market_days:
                market_days[key][date] = market_cache.entry(date, day.size)

        j = day.panel_j
        hist.summaries[date] = minutes.day_summary(day, panel.ref[t, j], panel.pct[t, j], panel.limit_up[t, j],
                                                   panel.limit_down[t, j], cuts)
        hist.prune()

    if args.start is None:  # a complete history from the first D:/data day: reusable by run_day.py
        for key, entries in market_days.items():
            market_cache.save(key, entries)

    if pending is not None:
        db_prev, ctx_prev = pending
        text = report.transcript(db_prev, None, None, outcome_note=f'待 {ctx_prev.date} 之后的交易日数据（D:/data 截至 {days[-1]}）。')
        path = out_dir / 'eod' / f'{ctx_prev.date}.md'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        recs['eod']['pending'] = db_prev   # the last decision's picks, to buy at the next open

    for mode in modes:
        r = recs[mode]
        mdir = out_dir / mode
        mdir.mkdir(parents=True, exist_ok=True)
        for name in ('picks', 'noms', 'days'):
            if r[name]:
                pd.DataFrame(r[name]).to_csv(mdir / f'{"nominations" if name == "noms" else name}.csv', index=False,
                                             encoding='utf-8-sig')
        if r.get('pending') is not None:  # the last 15:10 review: picks known, not bought yet
            db_prev = r['pending']
            names = {d.key: d.name for d in db_prev.debaters}
            pd.DataFrame([{'date': db_prev.ctx.date, 'mode': 'eod', 'rank': ch.rank, 'code': ch.code, 'name': ch.name,
                           'count': ch.count, 'choosers': '、'.join(names[k] for k in ch.keys)} for ch in db_prev.picks],
                         columns=['date', 'mode', 'rank', 'code', 'name', 'count', 'choosers']) \
                .to_csv(mdir / 'pending.csv', index=False, encoding='utf-8-sig')
    meta = {'generated': datetime.now().strftime('%Y-%m-%d %H:%M'), 'data_days': len(data_days),
            'window_days': len(window), 'first': window[0] if window else '-', 'last': window[-1] if window else '-',
            'audit_total': len(audits), 'audit_pass': sum(a['identical'] for a in audits), 'dates': panel.dates}
    write_summary(out_dir, recs, modes, meta)
    engine_files = sorted((ROOT / 'engine').glob('*.py')) + [ROOT / 'run_backtest.py']
    used = [d.key for d in debaters] + [config.ROUNDTABLE_ROOT.name]   # the debaters' skills and the roundtable's own
    skill_files = [config.SKILLS_ROOT / name / 'SKILL.md' for name in used]
    audit = {
        'version': config.VERSION, 'data_version': config.DATA_VERSION, 'generated': meta['generated'],
        'modes': modes, 'window': window,
        'cutoffs': {m: config.MODES[m]['cutoff'] for m in modes},
        'pit_rules': ['history = panel rows [:t] (previous trading days only)',
                      'today = pre-open reference price, price band and limit prices from panel row t',
                      'today intraday = minute bars [0, k) only; stocks with no bar before the cutoff are invisible',
                      'outcomes computed after the decision is recorded; eod outcomes use the next day and are settled '
                      'when that day is loaded; the T+1 sale reads the daily panel row after the buy day',
                      'names, ST flags and industries are a snapshot fetched 2026-08-22 (no price information)',
                      'universe = stocks still listed when the data was downloaded (minute files 2026-09-20, '
                      '09-24 and 10-03; daily bars 09-24): stocks delisted before that are missing (survivorship)'],
        'pit_asserts': len(asserts), 'scramble_audits': audits,
        'engine_sha256': {p.name: sha256(p) for p in engine_files},
        'skills_sha256': {p.parent.name: sha256(p) for p in skill_files},
        'panel': {'dates': [panel.dates[0], panel.dates[-1]], 'codes': len(panel.codes)},
        'thresholds': {k: getattr(config, k) for k in ('STRONG_SCALED', 'CLUSTER_MIN_STRONG', 'CLUSTER_CORR',
                                                         'CLUSTER_MIN_SIZE', 'STAGE_LOOKBACK', 'STAGE_MIN_HISTORY',
                                                         'STAGE_DEFAULTS', 'NOMINATIONS_PER_DEBATER', 'CONSENSUS_MIN')},
    }
    (out_dir / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print(f'ALL DONE in {time.time() - t0:.0f}s; window {len(window)} days; audits {meta["audit_pass"]}/{meta["audit_total"]}')


if __name__ == '__main__':
    main()
