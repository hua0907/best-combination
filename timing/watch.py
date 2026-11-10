"""盘中盯盘：按 timing/rules.json 的规则，轮询 TDX 行情，今天要买 / 要卖的票到价、到时就提示（响铃 + 高亮）。
只提示，不下单；不选股、不改引擎。

    python -X utf8 timing\\watch.py                      # 09:15–15:01 自动盯今天的买入清单和持仓
    python -X utf8 timing\\watch.py --sell 002724        # 手动指定要卖的票（例如系统外买的）
    python -X utf8 timing\\watch.py --buy 603089         # 手动指定要买的票
    python -X utf8 timing\\watch.py --once               # 只取一次行情打印后退出（测试连接）

卖出提示：
  1) 到价：最新价或最高价 ≥ 挂单价（昨收 +limit_pct%，不超过涨停价）→ 核对委托应已成交
  2) 到时：fallback_time（默认 14:50）还没成交 → 撤单，市价卖出
  3) 涨停：封在涨停价时挂单已在涨停价成交；若没挂单，按规则在涨停价卖
买入提示：
  1) 09:30 开盘：报开盘价和涨幅 → 按开盘价市价买入
  2) 开盘一字涨停：挂涨停价排队；之后开板会提示
  3) 盘中触及跌停只记录，不建议加仓（研究里 10 笔触及跌停的买入日平均没有好处）
日志写 timing\\output\\watch-<日期>.log。
"""
import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan import OUT, RULES, SH_TZ, build, levels, symbol  # noqa: E402
from engine import profile, store  # noqa: E402  (engine path was added by plan.py)

W = RULES['watch']
S = RULES['sell']


def beep():
    if not W.get('beep', True):
        return
    try:
        import winsound
        winsound.Beep(1200, 350)
        winsound.Beep(1600, 350)
    except Exception:  # noqa: BLE001
        print('\a', end='')


class Log:
    def __init__(self, day):
        OUT.mkdir(parents=True, exist_ok=True)
        self.path = OUT / f'watch-{day}.log'

    def __call__(self, msg, alert=False):
        line = f'[{dt.datetime.now(SH_TZ):%H:%M:%S}] {msg}'
        print(('\n>>> ' + line + '\n') if alert else line, flush=True)
        with open(self.path, 'a', encoding='utf-8') as fh:
            fh.write(line + '\n')
        if alert:
            beep()


def snapshot_map(client, symbols):
    rows = client.quotes.get_snapshots(symbols)
    out = {}
    for r in rows:
        d = r if isinstance(r, dict) else getattr(r, '__dict__', {}) or {}
        if not d:
            try:
                import eltdx
                d = eltdx.to_jsonable(r)
            except Exception:  # noqa: BLE001
                continue
        code = str(d.get('code', ''))
        last = float(d.get('last_price') or 0)
        pre = float(d.get('pre_close_price') or d.get('last_close_price') or 0)
        hand = float(d.get('total_hand') or 0)
        amt = float(d.get('amount') or 0)
        out[code] = {'last': last, 'pre': pre, 'open': float(d.get('open_price') or 0), 'high': float(d.get('high_price') or 0),
                     'low': float(d.get('low_price') or 0), 'vwap': amt / (hand * 100) if hand > 0 else np.nan,
                     'bid1': (d.get('buy_levels') or [{}])[0].get('price', 0), 'ask1': (d.get('sell_levels') or [{}])[0].get('price', 0)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date')
    ap.add_argument('--sell', nargs='*', default=[], help='手动指定要卖的代码')
    ap.add_argument('--buy', nargs='*', default=[], help='手动指定要买的代码')
    ap.add_argument('--once', action='store_true')
    args = ap.parse_args()
    day = args.date or dt.datetime.now(SH_TZ).strftime('%Y-%m-%d')
    log = Log(day)

    text, buys, sells, panel = build(day)
    print(text)
    names = dict(zip(panel.codes, panel.names))
    sell_codes = [c for c, _, _ in sells] + [c for c in args.sell if c not in [x[0] for x in sells]]
    buy_codes = [c for c, _, _ in buys] + [c for c in args.buy if c not in [x[0] for x in buys]]
    watch = {}
    for c in sell_codes:
        watch[c] = {'side': 'sell', **(levels(panel, c, names.get(c, ''), day) or {'code': c, 'name': names.get(c, ''), 'symbol': symbol(c),
                                                                                  'ref': np.nan, 'lu': np.nan, 'ld': np.nan, 'sell_limit': np.nan, 'stop': None})}
    for c in buy_codes:
        if c in watch:
            continue
        watch[c] = {'side': 'buy', **(levels(panel, c, names.get(c, ''), day) or {'code': c, 'name': names.get(c, ''), 'symbol': symbol(c),
                                                                                 'ref': np.nan, 'lu': np.nan, 'ld': np.nan, 'sell_limit': np.nan, 'stop': None})}
    if not watch:
        log('今天没有要买或要卖的票，退出')
        return
    for v in watch.values():
        log(f'盯盘 {v["name"]}({v["code"]}) {"卖出" if v["side"] == "sell" else "买入"}：昨收 {v["ref"]:.2f} 涨停 {v["lu"]:.2f} 跌停 {v["ld"]:.2f}'
            + (f' 挂单卖出价 {v["sell_limit"]:.2f}，{S["fallback_time"]} 兜底' if v['side'] == 'sell' else ''))

    import eltdx
    client = eltdx.Client()
    client.connect()
    fired = {c: set() for c in watch}
    fb_h, fb_m = (int(x) for x in S['fallback_time'].split(':'))
    last_status = 0.0
    symbols = [v['symbol'] for v in watch.values()]
    try:
        while True:
            now = dt.datetime.now(SH_TZ)
            hm = now.hour * 60 + now.minute
            if not args.once and hm < 9 * 60 + 15:
                time.sleep(min(60, (9 * 60 + 15 - hm) * 60))
                continue
            if not args.once and hm > 15 * 60 + 1:
                log('15:01 收盘，退出')
                break
            try:
                q = snapshot_map(client, symbols)
            except Exception as exc:  # noqa: BLE001
                log(f'行情读取失败：{type(exc).__name__}: {exc}；重连')
                try:
                    client.close()
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(3)
                client = eltdx.Client()
                client.connect()
                continue
            in_session = (9 * 60 + 30 <= hm <= 11 * 60 + 30) or (13 * 60 <= hm <= 15 * 60)
            lines = []
            for c, v in watch.items():
                s = q.get(c)
                if not s or s['last'] <= 0:
                    lines.append(f'{v["name"]}({c}) 无行情')
                    continue
                ref = v['ref'] if np.isfinite(v['ref']) else s['pre']
                chg = (s['last'] / ref - 1) * 100 if ref else np.nan
                opn = (s['open'] / ref - 1) * 100 if ref and s['open'] else np.nan
                lines.append(f'{v["name"]}({c}) {s["last"]:.2f} {chg:+.2f}% 开{s["open"]:.2f}({opn:+.1f}%) 高{s["high"]:.2f} 低{s["low"]:.2f} '
                             f'均{s["vwap"]:.2f}' + (f' | 挂{v["sell_limit"]:.2f}' if v['side'] == 'sell' else f' | 涨停{v["lu"]:.2f}'))
                f = fired[c]
                at_lu = np.isfinite(v['lu']) and bool(profile.at_or_above(s['last'], v['lu']))
                if v['side'] == 'sell':
                    if s['open'] > 0 and 'open' not in f and hm >= 9 * 60 + 30:
                        f.add('open')
                        if s['open'] >= v['sell_limit'] - 1e-9:
                            log(f'{v["name"]} 开盘 {s["open"]:.2f}（{opn:+.2f}%）≥ 挂单价 {v["sell_limit"]:.2f}：挂单开盘已成交，核对委托', alert=True)
                        else:
                            log(f'{v["name"]} 开盘 {s["open"]:.2f}（{opn:+.2f}%），挂单 {v["sell_limit"]:.2f} 继续等；{S["fallback_time"]} 没成交就市价卖')
                    if 'hit' not in f and in_session and (s['high'] >= v['sell_limit'] - 1e-9 or s['last'] >= v['sell_limit'] - 1e-9):
                        f.add('hit')
                        log(f'{v["name"]} 到价：最新 {s["last"]:.2f} 最高 {s["high"]:.2f} ≥ 挂单价 {v["sell_limit"]:.2f}，挂单应已成交，核对委托；'
                            '没挂单的现在卖', alert=True)
                    if at_lu and 'lu' not in f:
                        f.add('lu')
                        log(f'{v["name"]} 涨停 {s["last"]:.2f}：按规则在涨停价卖出（挂单已成交的不用管）', alert=True)
                    if v['stop'] and 'stop' not in f and in_session and s['last'] <= v['stop'] + 1e-9:
                        f.add('stop')
                        log(f'{v["name"]} 跌破止损价 {v["stop"]:.2f}（最新 {s["last"]:.2f}）：撤卖单，市价卖出', alert=True)
                    if 'fallback' not in f and hm >= fb_h * 60 + fb_m and 'hit' not in f:
                        f.add('fallback')
                        log(f'{v["name"]} 到 {S["fallback_time"]} 挂单没成交：撤单，按 {s["last"]:.2f} 左右市价卖出，不留到明天', alert=True)
                else:
                    if s['open'] > 0 and 'open' not in f and hm >= 9 * 60 + 30:
                        f.add('open')
                        if np.isfinite(v['lu']) and profile.at_or_above(s['open'], v['lu']):
                            log(f'{v["name"]} 开盘一字涨停 {s["open"]:.2f}：挂涨停价 {v["lu"]:.2f} 排队，开板自动成交；全天不开板就不买', alert=True)
                            f.add('queued')
                        else:
                            log(f'{v["name"]} 开盘 {s["open"]:.2f}（{opn:+.2f}%）：现在按开盘价市价买入；买入后不低吸、不加仓', alert=True)
                    if 'queued' in f and 'opened' not in f and not at_lu and in_session:
                        f.add('opened')
                        log(f'{v["name"]} 开板了（最新 {s["last"]:.2f}）：排队的涨停价买单应已成交，核对委托', alert=True)
                    if np.isfinite(v['ld']) and 'ld' not in f and in_session and profile.at_or_below(s['low'], v['ld']):
                        f.add('ld')
                        log(f'{v["name"]} 盘中触及跌停 {v["ld"]:.2f}（只记录：研究里买入日触及跌停的 10 笔，加仓或等跌停买平均没有好处）')
            if args.once:
                print('\n'.join(lines))
                break
            if time.time() - last_status >= W.get('status_every_seconds', 30):
                last_status = time.time()
                log(' | '.join(lines))
            time.sleep(W.get('poll_seconds', 3))
    except KeyboardInterrupt:
        log('手动退出')
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == '__main__':
    main()
