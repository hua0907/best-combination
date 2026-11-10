"""The nine debaters meeting early in the session of the trading day itself, instead of after the previous close.

    python -X utf8 combo_study/early_meetings.py 09:35 09:45 10:00

The frozen engine is imported and used as it is (nothing under skills/ is written): the same minute loader, context,
debaters and roundtable as run_backtest.py, held at each of the times given on every full-market day of D:/data. A
named stock is bought at the meeting minute's close (sealed at the limit-up price then: queued at the limit, filled
only if a later bar trades below it) and sold on the next trading day at its high, midpoint or low: the engine's own
outcome rules (evaluate.same_day, evaluate.next_day). The 10:00 meeting repeats the engine's own 10:00 backtest and
is compared with its nominations.csv, as a check of this runner.

Every day is read once by a pool of loaders; each meeting time keeps its own history in its own process.
Writes combo_study/output/early/nominations-<HHMM>.csv and days-<HHMM>.csv.
"""
import queue
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent / 'skills' / 'stock-roundtable'
OUT = HERE / 'output' / 'early'
sys.path.insert(0, str(ENGINE))

from engine import config, context, evaluate, minutes, profile, roundtable, store  # noqa: E402
from engine.debaters import all_debaters  # noqa: E402

LOADERS = 4
RESULT = ['filled', 'queued', 'entry', 'buy_chg', 'ret_close', 'sell_date', 't1_best', 't1_mid', 't1_worst', 't1_stuck']
_panel = None


def _init():
    global _panel
    _panel = store.build_panel()


def load(date, cuts):
    """One day for the meetings: the bars up to the last meeting time, the whole-day summary that later days read,
    and what the outcome rules need from the rest of the day."""
    day = minutes.load_day(date, _panel)
    t = _panel.index_of(date)
    j = day.panel_j
    summary = minutes.day_summary(day, _panel.ref[t, j], _panel.pct[t, j], _panel.limit_up[t, j],
                                  _panel.limit_down[t, j], cuts)
    n = max(cuts)
    head = minutes.Day(date, j, *(np.ascontiguousarray(x[:, :n]) for x in (
        day.open, day.high, day.low, day.close, day.vol, day.amt, day.real)))
    low_after = {k: day.low[:, k:].min(axis=1) for k in cuts}
    return date, head, summary, low_after, day.close[:, -1].copy()


def meet(time_str, inbox):
    """All days at one meeting time, in date order."""
    spec = config.meeting(time_str)
    k, key = spec['cutoff_index'], spec['key']
    tag = key.replace(':', '')
    panel, xdxr, debaters = store.build_panel(), store.load_xdxr(), all_debaters()
    hist = context.History()
    noms, days = [], []
    while True:
        item = inbox.get()
        if item is None:
            break
        date, head, summary, low_after, close = item
        t = panel.index_of(date)
        if head.size >= config.FULL_MARKET_MIN_CODES:
            ctx = context.build(panel, t, date, spec, head, hist, store.float_shares_asof(xdxr, panel.codes, date))
            db = roundtable.run(ctx, debaters)
            s = ctx.stocks[ctx.stocks['eligible']]
            row, lu = s['row'].to_numpy(), s['limit_up'].to_numpy(float)
            px = head.close[row, k - 1]
            queued = profile.at_or_above(px, lu)
            filled = ~queued | ~profile.at_or_above(low_after[row], lu)
            entry = np.where(filled, np.where(queued, lu, px), np.nan)
            out = evaluate.next_day(panel, t, s['j'].to_numpy(), entry)
            out.index = s.index
            out['filled'], out['queued'], out['entry'] = filled, queued, entry
            out['ret_close'] = (close[row] / entry - 1.0) * 100.0
            for d, picks, _ in db.opening:
                for order, (idx, model, _, score) in enumerate(picks, 1):
                    ch = db.choices[idx]
                    noms.append({'date': date, 'time': key, 'debater': d.name, 'model': model, 'order': order,
                                 'code': ch.code, 'name': ch.name, 'count': ch.count, 'sealed': ch.sealed,
                                 **({c: out.at[idx, c] for c in RESULT} if idx in out.index else {})})
            sold = out[out['filled'] & out['t1_mid'].notna()]
            pool = sold.loc[[i for i in db.choices if i in sold.index]]
            days.append({'date': date, 'time': key, 'stage': ctx.stage, 'n_eligible': len(s),
                         'n_choices': len(db.choices), 'n_picks': len(db.picks), 'base_all_n': len(sold),
                         **{f'base_all_{x}': sold[f't1_{x}'].mean() for x in ('best', 'mid', 'worst')},
                         'base_pool_t1': pool['t1_mid'].mean()})
            hist.market[ctx.key].append((date, ctx.market))
            hist.stages[ctx.key].append((date, ctx.stage))
        hist.summaries[date] = summary
        hist.prune()
    noms, days = pd.DataFrame(noms), pd.DataFrame(days)
    noms.to_csv(OUT / f'nominations-{tag}.csv', index=False, encoding='utf-8-sig')
    days.to_csv(OUT / f'days-{tag}.csv', index=False, encoding='utf-8-sig')
    note = ''
    if key == config.MODES['intraday']['key']:   # the engine's own 10:00 backtest: the choices must be the same
        cols = ['date', 'debater', 'order', 'code']
        theirs = pd.read_csv(config.OUTPUT_DIR / 'intraday' / 'nominations.csv', dtype={'code': str})[cols]
        both = noms[cols].astype(str).merge(theirs.astype(str), how='outer', indicator=True)
        note = f'; against the engine\'s 10:00 backtest: {dict(both["_merge"].value_counts())}'
    print(f'{key}: {len(days)} meetings, {len(noms)} choices, {noms.groupby(["date", "code"]).ngroups} named stocks'
          + note, flush=True)


def run(time_str, inbox):
    try:
        meet(time_str, inbox)
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        raise


def main():
    times = sys.argv[1:] or ['09:35', '09:45', '10:00']
    cuts = tuple(config.meeting(x)['cutoff_index'] for x in times)
    OUT.mkdir(parents=True, exist_ok=True)
    dates = sorted(p.stem for p in config.DATA_DIR.glob('*.csv'))
    boxes = [Queue(maxsize=3) for _ in times]
    procs = [Process(target=run, args=(x, q)) for x, q in zip(times, boxes)]
    for p in procs:
        p.start()
    t0 = time.time()
    with ProcessPoolExecutor(LOADERS, initializer=_init) as pool:
        ahead = {}
        for i, date in enumerate(dates):
            for d in dates[i:i + LOADERS + 2]:
                if d not in ahead:
                    ahead[d] = pool.submit(load, d, cuts)
            _, head, summary, low_after, close = ahead.pop(date).result()
            for k, q, p in zip(cuts, boxes, procs):
                while True:
                    if not p.is_alive():
                        raise SystemExit('a meeting process stopped; see the traceback above')
                    try:
                        q.put((date, head, summary, low_after[k], close), timeout=5)
                        break
                    except queue.Full:
                        pass
            if i % 10 == 0:
                print(f'{date} read ({time.time() - t0:.0f}s)', flush=True)
    for q in boxes:
        q.put(None)
    for p in procs:
        p.join()
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
