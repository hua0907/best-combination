"""盘前计划：按 timing/rules.json 的买卖规则，打印今天要买的票和要卖的票的价格台阶（不选股、不改引擎）。

    python -X utf8 timing\\plan.py                  # 今天
    python -X utf8 timing\\plan.py --date 2026-10-26

要买的票：最近一期 status=ok 的复盘（reports\\<复盘日>.json）的 T+1 清单，且它的 next_day 就是当天。
要卖的票：冻结引擎 trades.csv 里“待卖出”的票（买入日的下一个交易日卖）。
昨收 = 引擎日线面板里该股最后一个有成交的收盘价；涨跌停按引擎的规则算。
同时写 timing\\output\\plan-<日期>.md。
"""
import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.pop('STOCK_ROUNDTABLE_ROOT', None)
ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / 'skills' / 'stock-roundtable'
REPORTS = ROOT / 'reports'
OUT = ROOT / 'timing' / 'output'
RULES = json.loads((ROOT / 'timing' / 'rules.json').read_text(encoding='utf-8'))
sys.path.insert(0, str(ENGINE_DIR))
from engine import profile, store  # noqa: E402

SH_TZ = dt.timezone(dt.timedelta(hours=8))


def tick(x):
    return float(profile.round_tick(x))


def symbol(code):
    """TDX symbol with exchange prefix."""
    if code.startswith(('6', '9')):
        return 'sh' + code
    if code.startswith(('4', '8', '92')):
        return 'bj' + code
    return 'sz' + code


def latest_review():
    for p in sorted(REPORTS.glob('????-??-??.json'), reverse=True):
        rec = json.loads(p.read_text(encoding='utf-8'))
        if rec.get('status') == 'ok':
            return rec
    return None


def levels(panel, code, name, day):
    """昨收、涨停、跌停、卖出挂单价 for `code` on `day` (str)."""
    j = panel.code_index.get(code)
    if j is None:
        return None
    closes = panel.close[:, j]
    idx = np.nonzero(np.isfinite(closes))[0]
    if not len(idx):
        return None
    last_i = int(idx[-1])
    ref = float(closes[last_i])
    pct = profile.limit_pct(code, name, day)
    lu = tick(ref * (1 + pct / 100)) if pct else np.nan
    ld = tick(ref * (1 - pct / 100)) if pct else np.nan
    limit_px = tick(ref * (1 + RULES['sell']['limit_pct'] / 100))
    if np.isfinite(lu):
        limit_px = min(limit_px, lu)
    stop = RULES['sell'].get('stop_loss_pct')
    stop_px = tick(ref * (1 - stop / 100)) if stop else None
    return {'code': code, 'name': name, 'symbol': symbol(code), 'ref': ref, 'ref_date': panel.dates[last_i], 'pct': pct,
            'lu': lu, 'ld': ld, 'sell_limit': limit_px, 'stop': stop_px}


def today_lists(day):
    """(to buy today, to sell today) as lists of (code, name, note)."""
    rec = latest_review()
    buys, note_b = [], ''
    if rec is None:
        note_b = '没有 status=ok 的复盘'
    elif rec.get('next_day') != day:
        note_b = f'最近一期复盘是 {rec["date"]}，它的 T+1 是 {rec["next_day"]}，不是 {day}'
    else:
        buys = [(x['code'], x['name'], f'{x["votes"]} 票：{"、".join(x["members"])}') for x in rec['review']['t1']]
        note_b = f'来自 {rec["date"]} 的复盘' + ('' if buys else '：空仓')
    t = pd.read_csv(ENGINE_DIR / 'output' / 'optimal-combo' / 'trades.csv', dtype={'code': str})
    held = t[t['result'] == '待卖出']
    sells = [(r['code'], r['name'], f'{r["buy_date"]} {r["buy_when"]} 买入 {r["entry"]:.2f}') for r in held.to_dict('records')]
    return buys, note_b, sells


def build(day):
    panel = store.build_panel()
    buys, note_b, sells = today_lists(day)
    s = RULES['sell']
    L = [f'# {day} 买卖计划（规则见 timing\\买卖点方案.md）', '']
    L += ['## 今天要卖（昨天买入的）', '']
    if not sells:
        L.append('没有持仓。')
    for code, name, note in sells:
        v = levels(panel, code, name, day)
        if v is None:
            L.append(f'- {name}({code})：日线面板里没有这只票，手动按昨收 × {1 + s["limit_pct"] / 100:.2f} 挂单')
            continue
        L += [f'- **{name}({code})** {note}',
              f'  - 昨收 {v["ref"]:.2f}（{v["ref_date"]}），涨停 {v["lu"]:.2f}，跌停 {v["ld"]:.2f}',
              f'  - 09:15 前挂好卖单：**{v["sell_limit"]:.2f}**（昨收 +{s["limit_pct"]:g}%，不超过涨停价），不撤单',
              f'  - 开盘价 ≥ {v["sell_limit"]:.2f} 时挂单开盘就成交；开盘一字涨停也是在涨停价成交',
              f'  - 到 **{s["fallback_time"]}** 还没成交：撤单，按当时价格市价卖出，不留到第二天']
        if v['stop']:
            L.append(f'  - 止损：跌破 {v["stop"]:.2f}（昨收 -{s["stop_loss_pct"]:g}%）就市价卖')
        else:
            L.append('  - 不设盘中止损（研究里止损没有提高收益；最差一笔 -19.8% 这种属于规则内的回撤）')
    L += ['', '## 今天要买（T+1 清单）', '', f'{note_b}', '']
    for code, name, note in buys:
        v = levels(panel, code, name, day)
        if v is None:
            L.append(f'- {name}({code})：日线面板里没有这只票')
            continue
        L += [f'- **{name}({code})** {note}',
              f'  - 昨收 {v["ref"]:.2f}（{v["ref_date"]}），涨停 {v["lu"]:.2f}，跌停 {v["ld"]:.2f}',
              f'  - 09:25 看集合竞价：开盘价 < {v["lu"]:.2f} → 09:30 按开盘价市价买入（全部资金，几只平分）',
              f'  - 开盘价 = {v["lu"]:.2f}（一字涨停）→ 挂涨停价排队，开板自动成交；全天不开板就不买',
              '  - 买入后不低吸、不加仓、不因为回落止损；明天按卖出规则卖']
    if buys:
        L += ['', '研究结论（69 笔）：开盘买 +2.41%/笔（均价卖口径），所有等回调、定时、挂低价的买法都更差；'
              '买入日有 74% 的最低点在 10:00 前出现，但那些回调主要发生在后来走弱的票上，低吸等于专挑弱票。']
    text = '\n'.join(L)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'plan-{day}.md').write_text(text, encoding='utf-8')
    return text, buys, sells, panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', help='YYYY-MM-DD，默认今天')
    args = ap.parse_args()
    day = args.date or dt.datetime.now(SH_TZ).strftime('%Y-%m-%d')
    text, _, _, _ = build(day)
    print(text)


if __name__ == '__main__':
    main()
