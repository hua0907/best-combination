"""最优组合（用户 2026-10-17 指定，规则见 最优组合策略.md）：炒股养家、92科比、陈小群、赵老哥、乔帮主五位，只算各自的
第一选择；收盘后 15:10 复盘里同一只票有≥2 位列为第一选择，就在下一个交易日开盘买入，买入后的下一个交易日卖出。

    python -X utf8 optimal_combo.py          # 先跑 run_backtest.py（新交易日先 fetch_xdxr.py --refresh）

读回测的 15:10 复盘记录（output/eod/nominations.csv、days.csv）。最后一场复盘的买入日还不在 D:/data 里，回测不写它的
选票，这一场用引擎重开一次（不看事后结果，并做防偷看审计）取五位的第一选择。写 output/optimal-combo/：votes.csv（每场
复盘五位的第一选择和结论）、trades.csv（每只符合规则的票和结果）、report.md。其他文件只读。

收益口径同回测：开盘买入（开盘一字涨停的排队，开板才成交，全天不开板就买不到）；买入后的下一个交易日按最高/最低的
均价卖出（另列最高价、最低价卖出）；卖出日一字跌停卖不掉，按跌停价计；每场一份资金，平分给当场符合规则的票（没买到的
那份算现金）；未扣交易费用（来回约 0.1%）。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from engine import config, meeting, minutes, profile, store  # noqa: E402

MEMBERS = ['炒股养家', '92科比', '陈小群', '赵老哥', '乔帮主']
MIN_VOTES = 2
SELECTED_THROUGH = '2026-10-13'   # 挑选这条规则时用到的最后一场复盘；之后的场次没有参与挑选
OUT = config.OUTPUT_DIR / 'optimal-combo'


def yes(x):
    return str(x).strip().lower() in ('true', '1', '1.0')


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def pc(x):
    return '—' if not np.isfinite(num(x)) else f'{num(x):+.2f}%'


def buy_detail(panel, buy_date, code, filled, queued):
    if not buy_date:
        return ''
    if filled and queued:   # opened sealed at the limit: bought in the first minute the board opened
        t, j = panel.index_of(buy_date), panel.code_index[code]
        day = minutes.load_day(buy_date, panel)
        row = int(np.searchsorted(day.panel_j, j))
        traded = minutes.traded_bars(day)[row]
        below = ~profile.at_or_above(day.low[row, :], panel.limit_up[t, j])
        idx = [m for m in range(int(np.argmax(traded)), 240) if below[m] and traded[m]]
        return f'{minutes.TIMES[idx[0]]} 开板成交（开盘一字涨停排队）' if idx else '排队成交'
    if filled:
        return '09:30 开盘'
    return '开盘一字涨停，全天没开板，未买到' if queued else '买入日停牌，未买到'


def hold_last(date):
    """Hold the review the backtest could not record yet; return the members' first choices and why others chose nothing."""
    m = meeting.convene(meeting.Data(), date, config.CLOSE_REVIEW, blind=True, audit=True, log=lambda *_: None)
    votes, reasons = {}, {}
    for d, noms, why in m.db.opening:
        if d.name not in MEMBERS:
            continue
        if noms:
            ch = m.db.choices[noms[0][0]]
            votes[d.name] = (ch.code, ch.name, noms[0][1])
        else:
            reasons[d.name] = why
    nine = {ch.code: (ch.count, ch.rank > 0) for ch in m.db.choices.values()}
    return votes, reasons, nine, bool(m.audit_passed)


def main():
    noms = pd.read_csv(config.OUTPUT_DIR / 'eod' / 'nominations.csv', dtype={'code': str})
    settled = pd.read_csv(config.OUTPUT_DIR / 'eod' / 'days.csv')['date'].astype(str).tolist()
    data_days = sorted(p.stem for p in config.DATA_DIR.glob('*.csv'))
    nxt = dict(zip(data_days, data_days[1:]))
    later = [d for d in data_days if d > settled[-1]]
    if len(later) > 1:
        raise SystemExit(f'回测只记录到 {settled[-1]} 的复盘，D:/data 已到 {data_days[-1]}：先运行 run_backtest.py')

    votes = {d: {} for d in settled}
    for r in noms[(noms['order'] == 1) & noms['debater'].isin(MEMBERS)].to_dict('records'):
        votes[r['date']][r['debater']] = (r['code'], r['name'], r['model'])
    outcome = {(r['date'], r['code']): r for r in noms.to_dict('records')}

    pending, reasons, nine, audit_ok = None, {}, {}, None
    if later and not config.too_early(later[0]):
        pending = later[0]
        votes[pending], reasons, nine, audit_ok = hold_last(pending)

    panel = store.build_panel()
    trades, vote_rows, day_rows = [], [], []
    for date in settled + ([pending] if pending else []):
        v = votes.get(date, {})
        count = pd.Series([c for c, _, _ in v.values()], dtype=object).value_counts()
        hits = sorted((c for c, n in count.items() if n >= MIN_VOTES), key=lambda c: (-count[c], c))
        phase = '选定后' if date > SELECTED_THROUGH else '回测'
        day_ret, day_pending = [], date == pending
        for code in hits:
            who = [m for m in MEMBERS if m in v and v[m][0] == code]
            row = {'review': date, 'phase': phase, 'code': code, 'name': v[who[0]][1], 'votes': len(who),
                   'members': '、'.join(who), 'detail': '；'.join(f'{m}「{v[m][2]}」' for m in who),
                   'buy_date': nxt.get(date, '')}
            o = outcome.get((date, code))
            if o is None:   # the review the backtest has not recorded: buy at the next open
                n9, picked = nine.get(code, (len(who), True))
                row.update(all9=n9, roundtable_pick=picked, filled=False, queued=False, buy_when='', entry=np.nan,
                           buy_chg=np.nan, ret_close=np.nan, sell_date='', t1_best=np.nan, t1_mid=np.nan, t1_worst=np.nan,
                           stuck=False, result='待买入（下一个交易日开盘）')
                trades.append(row)
                continue
            filled, queued, t1 = yes(o['filled']), yes(o['queued']), num(o['t1_mid'])
            sale_day = nxt.get(row['buy_date'], '') if row['buy_date'] else ''
            if not filled:
                result = '未买到'
            elif np.isfinite(t1):
                result = '赚' if t1 > 0 else '亏'
            elif sale_day:
                result = '卖出日停牌（按 0 计）'
            else:
                result = '待卖出'
                day_pending = True
            row.update(all9=int(num(o['count'])), roundtable_pick=yes(o['picked']), filled=filled, queued=queued,
                       buy_when=buy_detail(panel, row['buy_date'], code, filled, queued), entry=num(o['entry']),
                       buy_chg=num(o['buy_chg']), ret_close=num(o['ret_close']),
                       sell_date=o['sell_date'] if isinstance(o['sell_date'], str) else sale_day,
                       t1_best=num(o['t1_best']), t1_mid=t1, t1_worst=num(o['t1_worst']), stuck=yes(o['t1_stuck']),
                       result=result)
            day_ret.append(t1 if filled and np.isfinite(t1) else 0.0)
            trades.append(row)
        if not day_pending:
            day_rows.append({'review': date, 'phase': phase, 'signals': len(hits),
                             'ret': float(np.mean(day_ret)) if hits else 0.0})
        cells = {}
        for m in MEMBERS:
            if m in v:
                cells[m] = f'{v[m][1]}({v[m][0]})「{v[m][2]}」'
            else:
                cells[m] = f'不选（{reasons[m]}）' if reasons.get(m) else '不选'
        names = {c: v[[m for m in MEMBERS if m in v and v[m][0] == c][0]][1] for c in hits}
        verdict = ('买入：' + '、'.join(f'{names[c]}（{count[c]} 票）' for c in hits)) if hits else '空仓（没有≥2 位的第一选择相同）'
        vote_rows.append({'review': date, **cells, 'verdict': verdict})

    t = pd.DataFrame(trades)
    d = pd.DataFrame(day_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(vote_rows).to_csv(OUT / 'votes.csv', index=False, encoding='utf-8-sig')
    t.to_csv(OUT / 'trades.csv', index=False, encoding='utf-8-sig')
    (OUT / 'report.md').write_text(report(t, d, vote_rows, settled, pending, audit_ok), encoding='utf-8')
    done = t[t['result'].isin(['赚', '亏'])]
    print(f'reviews {len(d)} settled (+{1 if pending else 0} pending); signal days {int((d["signals"] > 0).sum())}; '
          f'trades {len(done)}; per trade {done["t1_mid"].mean():+.2f}%; win {(done["t1_mid"] > 0).mean():.1%}; '
          f'total {d["ret"].sum():+.2f}%; written {OUT}')


def stats(t, d):
    done = t[t['result'].isin(['赚', '亏'])]
    cum = d['ret'].cumsum()
    best = done.loc[done['t1_mid'].idxmax()] if len(done) else None
    worst = done.loc[done['t1_mid'].idxmin()] if len(done) else None
    return {'reviews': len(d), 'signal_days': int((d['signals'] > 0).sum()) if len(d) else 0,
            'picks': int(t['review'].isin(d['review']).sum()) if len(t) else 0,
            'trades': len(done), 'per_trade': done['t1_mid'].mean() if len(done) else np.nan,
            'win': (done['t1_mid'] > 0).mean() if len(done) else np.nan, 'total': d['ret'].sum() if len(d) else np.nan,
            'max_dd': float((cum - cum.cummax()).min()) if len(d) else np.nan,
            'best': f'{best["name"]}（{best["review"]}）{pc(best["t1_mid"])}' if best is not None else '—',
            'worst': f'{worst["name"]}（{worst["review"]}）{pc(worst["t1_mid"])}' if worst is not None else '—'}


def report(t, d, vote_rows, settled, pending, audit_ok):
    audit = json.loads((config.OUTPUT_DIR / 'audit.json').read_text(encoding='utf-8'))
    s_all = stats(t, d)
    s_after = stats(t[t['phase'] == '选定后'], d[d['phase'] == '选定后'])
    L = ['# 最优组合：五位辩手第一选择共识 —— 逐笔投票与收益', '',
         '规则：炒股养家、92科比、陈小群、赵老哥、乔帮主五位，只算各自的第一选择；收盘后 15:10 复盘里同一只票有≥2 位列为'
         '第一选择，下一个交易日开盘买入（几只同时符合就平分资金，没有就空仓）；买入后的下一个交易日卖出。详见 最优组合策略.md。', '',
         f'- 数据：圆桌引擎 {audit["version"]} 回测（{audit["generated"]}），已结算的复盘 {settled[0]} 至 {settled[-1]}，共 {len(settled)} 场'
         + (f'；最新一场 {pending} 的复盘已开（不看事后结果），防偷看审计{"通过" if audit_ok else "未通过！"}' if pending else ''),
         '- 收益口径：开盘买入（开盘一字涨停的排队，开板才成交，全天不开板就买不到）；买入后的下一个交易日按最高/最低的均价卖出'
         '（另列最高价、最低价卖出）；卖出日一字跌停按跌停价计；每场一份资金平分给当场符合规则的票，没买到的那份算现金；'
         '“合计”= 各场收益相加（单利）；未扣交易费用（来回约 0.1%）',
         f'- 挑选这条规则时用的是截至 {SELECTED_THROUGH} 的复盘；之后的场次标“选定后”，没有参与挑选', '',
         '## 一、汇总', '', '| 项目 | 全部已结算 | 其中选定后 |', '|---|---|---|']
    for label, key in (('已结算的复盘场次', 'reviews'), ('有买入信号的场次', 'signal_days'),
                       ('符合规则的票（含没买到的）', 'picks'), ('买到并已卖出的笔数', 'trades')):
        L.append(f'| {label} | {s_all[key]} | {s_after[key]} |')
    win_after = f'{s_after["win"]:.0%}' if np.isfinite(s_after['win']) else '—'
    L += [f'| 每笔平均（均价卖出） | {pc(s_all["per_trade"])} | {pc(s_after["per_trade"])} |',
          f'| 胜率 | {s_all["win"]:.0%} | {win_after} |',
          f'| 合计（各场收益相加） | {pc(s_all["total"])} | {pc(s_after["total"])} |',
          f'| 最大回撤（合计曲线从高点回落） | {pc(s_all["max_dd"])} | {pc(s_after["max_dd"])} |',
          f'| 最好一笔 | {s_all["best"]} | {s_after["best"]} |', f'| 最差一笔 | {s_all["worst"]} | {s_after["worst"]} |', '']

    L += ['## 二、逐笔明细', '',
          '| # | 复盘日 | 股票 | 票数 | 投第一票的成员（各自的模式） | 9 人圆桌票数 | 买入 | 买入价（相对昨收） | 卖出日 | '
          '均价卖出 | 最高价卖出 | 最低价卖出 | 结果 |', '|' + '---|' * 13]
    for i, r in enumerate(t.to_dict('records'), 1):
        buy = f'{r["buy_date"]} {r["buy_when"]}'.strip() if r['buy_date'] else '下一个交易日开盘'
        price = f'{r["entry"]:.2f}（{pc(r["buy_chg"])}）' if r['filled'] else '—'
        sell = (r['sell_date'] or '—') + ('（一字跌停，实际卖不掉）' if r['stuck'] else '')
        tag = '（选定后）' if r['phase'] == '选定后' else ''
        L.append(f'| {i} | {r["review"]}{tag} | {r["name"]}({r["code"]}) | {r["votes"]} | {r["detail"]} | {r["all9"]} | {buy} | '
                 f'{price} | {sell if r["filled"] else "—"} | {pc(r["t1_mid"])} | {pc(r["t1_best"])} | {pc(r["t1_worst"])} | '
                 f'{r["result"]} |')

    d = d.assign(month=d['review'].str[:7])
    done = t[t['result'].isin(['赚', '亏'])].assign(month=lambda x: x['review'].str[:7])
    L += ['', '## 三、每月', '', '| 月份 | 复盘场次 | 有信号的场次 | 买到并卖出的笔数 | 胜率 | 合计 |', '|---|---|---|---|---|---|']
    for mth, g in d.groupby('month'):
        x = done[done['month'] == mth]
        win = f'{(x["t1_mid"] > 0).mean():.0%}' if len(x) else '—'
        L.append(f'| {mth} | {len(g)} | {int((g["signals"] > 0).sum())} | {len(x)} | {win} | {pc(g["ret"].sum())} |')

    L += ['', f'## 四、每场复盘五位的第一选择（共 {len(vote_rows)} 场）', '',
          '| 复盘日 | ' + ' | '.join(MEMBERS) + ' | 结论 |', '|' + '---|' * (len(MEMBERS) + 2)]
    for r in vote_rows:
        L.append(f'| {r["review"]} | ' + ' | '.join(r[m] for m in MEMBERS) + f' | {r["verdict"]} |')
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()
