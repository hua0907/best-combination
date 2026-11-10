"""Daily pipeline of the optimal-combination review (README.md). Scheduled on weekdays at 15:10 and 16:40.

    python pipeline/run.py                           # today: data, dividends, backtest, optimal combination, review, page
    python pipeline/run.py --date 2026-10-14         # replay a past review from the current engine outputs (no download)
    python pipeline/run.py --date 2026-10-16 --full  # every step for a past day (its data must already be in D:/data)

A second run on a day whose review already finished (the 16:40 trigger) does not repeat it: it compares the day's
data file with a fresh TDX fetch and adds the result to the page. Exit codes: 0 done or nothing to do, 1 failed,
2 the day's data was not ready.
"""
import argparse
import datetime as dt
import json
import time
import traceback

import pandas as pd

from common import (CLOSE_BAR_SHARE, ENGINE_DIR, LOGS, REPORTS, REVIEW_TIME, Log, calendar_info, close_bar_share,
                    data_file, frozen_status, now, run_script, sha256)
from pull_today import pull


def save(rec):
    (REPORTS / f'{rec["date"]}.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str),
                                                 encoding='utf-8')


def marker(day, status, new_report):
    (LOGS / 'last-run.json').write_text(json.dumps({'date': str(day), 'status': status, 'new_report': new_report,
                                                    'at': now().isoformat(timespec='seconds')}, ensure_ascii=False),
                                        encoding='utf-8')


def existing_data(day):
    path = data_file(day)
    if not path.exists():
        return 'failed', f'D:\\data 里没有 {day}.csv', {}
    stocks, share, rows = close_bar_share(path)
    info = {'file': str(path), 'stocks': stocks, 'rows': rows, 'close_bar_share': round(share, 4), 'sha256': sha256(path)}
    if share < CLOSE_BAR_SHARE:
        return 'failed', f'只有 {share:.1%} 的股票有 15:00 那根（要求 ≥99%）', info
    return 'ok', f'{stocks} 只股票，{share:.1%} 有 15:00 收盘那根（已在 D:\\data）', info


def engine_step(day, script, args, timeout, attempts=1, need=None):
    def fn():
        rc, tail = None, ''
        for _ in range(attempts):
            rc, tail, _secs = run_script([script] + args, ENGINE_DIR, LOGS / f'{day}-{script[:-3]}.out', timeout)
            if rc == 0 and (need is None or need in tail):
                return 'ok', tail, {}
        return 'failed', f'exit {rc}: {tail}', {}
    return fn


def data_check(day, old, log):
    from verify_data import verify
    try:
        old['data_check'] = verify(day, [x['code'] for x in old['review']['t1']], log)
    except Exception as exc:  # noqa: BLE001 - the check is informative; the review stands
        old['data_check'] = {'error': f'{type(exc).__name__}: {exc}'}
        log(traceback.format_exc())
    save(old)
    from render import render
    render(old)


def main():
    ap = argparse.ArgumentParser(description='最优组合每日复盘')
    ap.add_argument('--date', help='复盘日，默认今天')
    ap.add_argument('--full', action='store_true', help='过去的日子也跑全部步骤（数据要已在 D:/data）')
    args = ap.parse_args()
    today = now().date()
    day = dt.date.fromisoformat(args.date) if args.date else today
    log = Log(day)
    full = args.full or day == today
    is_trading, prev_day, next_day, calendar = calendar_info(day)
    if not is_trading:
        log(f'{day} 不是交易日，不复盘')
        marker(day, 'not-a-trading-day', False)
        return 0
    if day == today and now().time() < REVIEW_TIME:
        log('15:10 以前不开收盘后复盘')
        marker(day, 'too-early', False)
        return 0
    path = REPORTS / f'{day}.json'
    old = json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
    if day == today and not args.full and old and old.get('status') == 'ok':
        if not old.get('data_check'):
            data_check(day, old, log)
            log('16:40 data check added to the page')
        marker(day, 'checked', False)
        return 0

    manifest, changed = frozen_status()
    rec = {'date': day.isoformat(), 'prev_day': prev_day.isoformat(), 'next_day': next_day.isoformat(),
           'calendar': calendar, 'replay': day != today, 'status': 'running',
           'started': now().isoformat(timespec='seconds'), 'steps': [],
           'engine': {'version': manifest['version'], 'frozen_at': manifest['frozen_at'], 'changed': changed}}
    if changed:
        log(f'WARNING: the frozen engine changed: {changed}')

    def step(name, label, fn):
        t0 = time.time()
        try:
            status, note, extra = fn()
        except Exception as exc:  # noqa: BLE001 - every failure ends up on the page
            status, note, extra = 'failed', f'{type(exc).__name__}: {exc}', {}
            log(traceback.format_exc())
        rec['steps'].append({'name': name, 'label': label, 'status': status, 'seconds': round(time.time() - t0, 1),
                             'note': note})
        log(f'{label}: {status} ({note})')
        return status, extra

    if full:
        st, info = step('data', '拉取当天分钟数据', lambda: pull(day, today, log))
    else:
        st, info = step('data', '检查当天分钟数据', lambda: existing_data(day))
    rec['data'] = info
    if st == 'ok' and full:
        st, _ = step('xdxr', '刷新除权除息', engine_step(day, 'fetch_xdxr.py', ['--refresh'], 1800, attempts=3))
    if st == 'ok' and full:
        st, _ = step('backtest', '回测（全部历史 + 当天复盘）',
                     engine_step(day, 'run_backtest.py', [], 7200, need='ALL DONE'))
    if st == 'ok' and full:
        st, _ = step('optimal', '最优组合逐日记录', engine_step(day, 'optimal_combo.py', [], 1800))

    def do_review():
        from review_day import review
        rv = rec['review'] = review(day)
        if not rv['audit_passed']:
            return 'failed', '防偷看审计未通过：今天的信号作废', {}
        return 'ok', f'T+1 {len(rv["t1"])} 只' + ('：' + '、'.join(x['label'] for x in rv['t1']) if rv['t1'] else '（空仓）'), {}

    def do_cross():
        t = pd.read_csv(ENGINE_DIR / 'output' / 'optimal-combo' / 'trades.csv', dtype={'code': str})
        a = sorted(t.loc[t['review'] == day.isoformat(), 'code'])
        b = sorted(x['code'] for x in rec['review']['t1'])
        rec['cross_check'] = a == b
        return ('ok' if a == b else 'failed'), f'回测记录：{"、".join(a) or "空仓"}；当场复盘：{"、".join(b) or "空仓"}', {}

    if st == 'ok':
        st, _ = step('review', '当天 15:10 复盘（盲测 + 防偷看审计）', do_review)
    if st == 'ok':
        st, _ = step('cross', 'T+1 清单两套计算核对', do_cross)

    statuses = [s['status'] for s in rec['steps']]
    rec['status'] = 'ok' if all(s == 'ok' for s in statuses) else 'not-ready' if 'not-ready' in statuses else 'failed'
    rec['finished'] = now().isoformat(timespec='seconds')
    save(rec)
    try:
        from render import render
        log(f'page: {render(rec)}')
    except Exception:  # noqa: BLE001
        log(traceback.format_exc())
    marker(day, rec['status'], True)
    return 0 if rec['status'] == 'ok' else 2 if rec['status'] == 'not-ready' else 1


if __name__ == '__main__':
    raise SystemExit(main())
