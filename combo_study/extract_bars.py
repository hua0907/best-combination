"""Pull what the 511-combination study needs out of D:/data, which is only read.

    python -X utf8 combo_study/extract_bars.py

Reads every D:/data/<date>.csv once and writes combo_study/cache/:
  daily.parquet   one row per stock per day, derived from the minute bars alone: the first traded bar and its open,
                  the high and low from the first trade on, the last close, volume and amount
  bars.parquet    the 1-minute bars of every stock a debater named (10:00 or 15:10 review) on the trading day before
                  the review, the review day and the two trading days after it
  extract.json    what was read: file sizes and times, row counts and anything odd in the files

The debaters' choices come from the frozen engine's backtest (skills/stock-roundtable/output/<mode>/nominations.csv,
rewritten from D:/data every day); only the two columns date and code are used here.
"""
import hashlib
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = HERE / 'cache'
ENGINE_OUT = ROOT / 'skills' / 'stock-roundtable' / 'output'
DATA_DIR = Path(json.loads((ROOT / 'config.json').read_text(encoding='utf-8-sig'))['data_dir'])
N_BARS = 240
WORKERS = 4


def bar_index(minute_of_day):
    """Position 0..239 of a bar stamped HH:MM (the minute it ends); -1 for a stamp outside the session.
    09:31..11:30 are 0..119 and 13:01..15:00 are 120..239. The files stamp the last morning bar 13:00 instead of
    11:30: both mean position 119."""
    m = np.asarray(minute_of_day, int)
    idx = np.full(m.shape, -1)
    am = (m >= 571) & (m <= 690)
    pm = (m >= 781) & (m <= 900)
    idx[am] = m[am] - 571
    idx[m == 780] = 119
    idx[pm] = m[pm] - 781 + 120
    return idx


def read_day(job):
    date, wanted = job
    path = DATA_DIR / f'{date}.csv'
    stat = path.stat()
    table = pacsv.read_csv(path, convert_options=pacsv.ConvertOptions(
        include_columns=['symbol', 'code', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'amount'],
        column_types={'symbol': pa.string(), 'code': pa.string(), 'datetime': pa.timestamp('s'),
                      'open': pa.float64(), 'high': pa.float64(), 'low': pa.float64(), 'close': pa.float64(),
                      'volume': pa.float64(), 'amount': pa.float64()}))
    df = table.to_pandas(strings_to_categorical=True)
    info = {'date': date, 'bytes': stat.st_size, 'rows': int(len(df)),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(timespec='seconds')}

    stamp = df['datetime']
    minute = (stamp.dt.hour * 60 + stamp.dt.minute).to_numpy()
    df['m'] = bar_index(minute)
    on_day = (stamp.dt.normalize() == pd.Timestamp(date)).to_numpy()
    priced = (df['close'] > 0).to_numpy()
    info.update(other_day=int((~on_day).sum()), outside_session=int((df['m'] < 0).sum()),
                no_close=int((~priced).sum()), stamped_1130=int((minute == 690).sum()),
                stamped_1300=int((minute == 780).sum()))
    df = df[on_day & priced & (df['m'] >= 0).to_numpy()].copy()
    for col in ('open', 'high', 'low'):   # a bar with a close but without this price: use its close
        bad = ~(df[col] > 0)
        info[f'no_{col}'] = int(bad.sum())
        if bad.any():
            df.loc[bad, col] = df.loc[bad, 'close']
    info['high_below_low'] = int((df['high'] < df['low']).sum())
    info['zero_volume'] = int((df['volume'] <= 0).sum())

    df['code'] = df['code'].cat.remove_unused_categories()
    df['symbol'] = df['symbol'].cat.remove_unused_categories()
    per_code = df.groupby('code', observed=True)['symbol'].nunique()
    if (per_code > 1).any():   # e.g. an index sharing a stock's six digits
        raise SystemExit(f'{date}: codes with more than one symbol: {list(per_code[per_code > 1].index[:10])}')
    twice = df.duplicated(['code', 'm'], keep='last')
    info['duplicate_bars'] = int(twice.sum())
    df = df[~twice.to_numpy()]

    row, codes = pd.factorize(df['code'], sort=True)
    codes = np.asarray(codes.astype(str))
    m = df['m'].to_numpy()
    n = len(codes)
    shape = (n, N_BARS)
    c = np.full(shape, np.nan)
    o, h, lo = c.copy(), c.copy(), c.copy()
    v, a = np.zeros(shape), np.zeros(shape)
    c[row, m] = df['close'].to_numpy(float)
    o[row, m] = df['open'].to_numpy(float)
    h[row, m] = df['high'].to_numpy(float)
    lo[row, m] = df['low'].to_numpy(float)
    v[row, m] = df['volume'].to_numpy(float)
    a[row, m] = df['amount'].to_numpy(float)

    real = np.isfinite(c)
    traded = real & (v > 0)   # a bar without volume is not a trade: it only repeats the last price
    trading = traded.any(axis=1)
    first = np.where(trading, traded.argmax(axis=1), 0)
    started = real & (np.arange(N_BARS)[None, :] >= first[:, None])
    last = N_BARS - 1 - real[:, ::-1].argmax(axis=1)
    each = np.arange(n)
    daily = pd.DataFrame({
        'date': date, 'code': codes, 'bars': real.sum(axis=1), 'traded_bars': traded.sum(axis=1),
        'first_bar': np.where(trading, first, -1), 'last_bar': last,
        'open': np.where(trading, o[each, first], np.nan),
        'high': np.where(trading, np.where(started, h, -np.inf).max(axis=1), np.nan),
        'low': np.where(trading, np.where(started, lo, np.inf).min(axis=1), np.nan),
        'close': c[each, last], 'volume': v.sum(axis=1), 'amount': a.sum(axis=1)})
    info.update(stocks=int(n), trading=int(trading.sum()), full_days=int((real.sum(axis=1) == N_BARS).sum()),
                with_close_bar=int(real[:, -1].sum()))

    bars = df[df['code'].astype(str).isin(wanted)]
    bars = pd.DataFrame({'date': date, 'code': bars['code'].astype(str).to_numpy(), 'm': bars['m'].to_numpy('int16'),
                         'open': bars['open'].to_numpy(), 'high': bars['high'].to_numpy(),
                         'low': bars['low'].to_numpy(), 'close': bars['close'].to_numpy(),
                         'volume': bars['volume'].to_numpy(), 'amount': bars['amount'].to_numpy()})
    info['wanted'] = len(wanted)
    info['wanted_found'] = int(bars['code'].nunique())
    return daily, bars, info


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    t0 = time.time()
    cal = sorted(p.stem for p in DATA_DIR.glob('*.csv') if re.fullmatch(r'\d{4}-\d{2}-\d{2}', p.stem))
    pos = {d: i for i, d in enumerate(cal)}
    wanted = {d: set() for d in cal}
    sources = {}
    for mode in ('eod', 'intraday'):
        path = ENGINE_OUT / mode / 'nominations.csv'
        noms = pd.read_csv(path, dtype={'code': str}, usecols=['date', 'code']).drop_duplicates()
        sources[mode] = {'file': str(path), 'sha256': sha256(path), 'rows_distinct': int(len(noms)),
                         'modified': datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec='seconds')}
        for date, code in noms.itertuples(index=False):
            i = pos[date]
            for k in range(max(i - 1, 0), min(i + 3, len(cal))):
                wanted[cal[k]].add(code)
    print(f'{len(cal)} trading days in {DATA_DIR}: {cal[0]} .. {cal[-1]}; '
          f'{sum(len(w) for w in wanted.values())} stock-days wanted', flush=True)

    daily, bars, infos = [], [], []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for d, b, info in pool.map(read_day, [(day, sorted(wanted[day])) for day in cal]):
            daily.append(d)
            bars.append(b)
            infos.append(info)
            if len(infos) % 20 == 0 or len(infos) == len(cal):
                print(f'  {len(infos)}/{len(cal)} files, through {info["date"]}, {time.time() - t0:.0f}s', flush=True)

    daily = pd.concat(daily, ignore_index=True)
    bars = pd.concat(bars, ignore_index=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(CACHE / 'daily.parquet', index=False)
    bars.to_parquet(CACHE / 'bars.parquet', index=False)
    odd_keys = ('other_day', 'outside_session', 'no_close', 'no_open', 'no_high', 'no_low', 'high_below_low',
                'duplicate_bars', 'stamped_1130', 'stamped_1300')
    totals = {k: int(sum(i[k] for i in infos)) for k in odd_keys + ('rows', 'bytes', 'zero_volume')}
    missing = int(sum(i['wanted'] - i['wanted_found'] for i in infos))
    (CACHE / 'extract.json').write_text(json.dumps({
        'generated': datetime.now().isoformat(timespec='seconds'), 'data_dir': str(DATA_DIR), 'calendar': cal,
        'nominations': sources, 'totals': totals, 'wanted_stock_days_without_bars': missing, 'files': infos},
        ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'read {totals["rows"]:,} bars, {totals["bytes"] / 2**30:.2f} GiB; daily rows {len(daily):,}; '
          f'kept {len(bars):,} bars of {bars.groupby(["date", "code"]).ngroups:,} stock-days '
          f'({missing} wanted stock-days have no bars)')
    print('odd: ' + ', '.join(f'{k} {totals[k]:,}' for k in odd_keys))
    print(f'last files written: ' + ', '.join(f'{i["date"]} {i["modified"]}' for i in infos[-3:]))
    print(f'nominations: ' + '; '.join(f'{m} {s["modified"]}' for m, s in sources.items()))
    print(f'done in {time.time() - t0:.0f}s; written {CACHE}')


if __name__ == '__main__':
    main()
