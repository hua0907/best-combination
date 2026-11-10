"""Point-in-time daily panel: official TDX daily bars x codes, with exchange reference prices.

Everything a decision on day T may see from history is panel rows [0, t) where dates[t] == T.
Ex-rights events are applied from their ex-date on (reference price on the ex-date, forward
adjustment of earlier bars); an event dated after T never changes anything visible at T.
"""
import json
import math

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from . import config, profile


def data_dates():
    """Trading days that have a 1-minute file in D:/data."""
    return sorted(p.stem for p in config.DATA_DIR.glob('*.csv'))


def _file_size(date):
    return (config.DATA_DIR / f'{date}.csv').stat().st_size


def minute_daily(dates):
    """Daily bars built from D:/data 1-minute bars, for days after the end of the official daily history."""
    frames = []
    for date in dates:
        table = pacsv.read_csv(config.DATA_DIR / f'{date}.csv', convert_options=pacsv.ConvertOptions(
            include_columns=['code', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'amount'],
            column_types={'code': pa.string(), 'datetime': pa.string()}))
        df = table.to_pandas()
        df = df[(df['datetime'].str.slice(0, 10) == date) & (df['close'] > 0)].sort_values(['code', 'datetime'])
        # Bars before a stock's first trade carry no traded price (TDX writes 240 flat zero-volume bars for a
        # suspended stock): the day starts at the first trade, and a stock without a trade gets no daily bar.
        traded = (df['volume'] > 0).astype(np.int8)
        df = df[traded.groupby(df['code']).cummax() > 0]
        g = df.groupby('code')
        day = pd.DataFrame({'open': g['open'].first(), 'high': g['high'].max(), 'low': g['low'].min(),
                            'close': g['close'].last(), 'volume': g['volume'].sum(), 'amount': g['amount'].sum()})
        frames.append(day.reset_index().assign(date=date))
    columns = ['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']
    return pd.concat(frames, ignore_index=True)[columns] if frames else pd.DataFrame(columns=columns)


def load_universe():
    rows = json.loads(config.UNIVERSE_FILE.read_text(encoding='utf-8'))['rows']
    df = pd.DataFrame(rows)[['code', 'name', 'industry']].drop_duplicates('code')
    df['name'] = df['name'].fillna('').str.replace(r'\s+', '', regex=True)
    df['industry'] = df['industry'].fillna('').replace({'-': ''})
    return df.set_index('code')


def load_daily_raw(force=False):
    cache = config.CACHE_DIR / 'daily_raw.parquet'
    if cache.exists() and not force:
        return pd.read_parquet(cache)
    records = []
    for path in sorted(config.DAILY_RAW_DIR.glob('*.json')):
        payload = json.loads(path.read_text(encoding='utf-8'))
        code = str(payload.get('code') or path.stem)
        for row in payload.get('rows') or []:
            records.append((code, row['date'], row.get('open') or 0.0, row.get('high') or 0.0, row.get('low') or 0.0,
                            row.get('close') or 0.0, row.get('volume') or 0.0, row.get('amount') or 0.0))
    df = pd.DataFrame(records, columns=['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount'])
    df = df[df['close'] > 0].drop_duplicates(['code', 'date']).sort_values(['code', 'date']).reset_index(drop=True)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def load_xdxr(force=False):
    cache = config.CACHE_DIR / 'xdxr.parquet'
    if cache.exists() and not force:
        return pd.read_parquet(cache)
    seen, records = set(), []
    for folder in config.XDXR_DIRS:
        if not folder.exists():
            continue
        for path in sorted(folder.glob('*.json')):
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
            except Exception:
                continue
            code = str(payload.get('code') or path.stem)
            if payload.get('status') != 'ok' or code in seen:
                continue
            seen.add(code)
            for row in payload.get('rows') or []:
                try:
                    date = f"{int(row['year']):04d}-{int(row['month']):02d}-{int(row['day']):02d}"
                except Exception:
                    continue
                records.append((code, date, int(row.get('category') or 0), row.get('fenhong'), row.get('songzhuangu'),
                                row.get('peigu'), row.get('peigujia'), row.get('panhouliutong')))
    df = pd.DataFrame(records, columns=['code', 'date', 'category', 'fenhong', 'songzhuangu', 'peigu', 'peigujia',
                                        'panhouliutong'])
    df.to_parquet(cache, index=False)
    (config.CACHE_DIR / 'xdxr_codes.json').write_text(json.dumps(sorted(seen)), encoding='utf-8')
    return df


class Panel:
    """Dense date x code arrays. Index with [t, j]; history for decision index t is [:t]."""

    def __init__(self, dates, codes, names, industries, arrays, xdxr_codes):
        self.dates = list(dates)
        self.codes = list(codes)
        self.names = np.asarray(names, dtype=object)
        self.industries = np.asarray(industries, dtype=object)
        self.code_index = {c: j for j, c in enumerate(self.codes)}
        self.date_index = {d: i for i, d in enumerate(self.dates)}
        self.xdxr_codes = set(xdxr_codes)
        for key, value in arrays.items():
            setattr(self, key, value)

    def index_of(self, date):
        return self.date_index[date]

    def adjusted(self, field, t):
        """History rows [:t] of open/high/low/close forward-adjusted to the price basis of dates[t]."""
        raw = getattr(self, field)[:t]
        return raw * np.exp(self.logcum[t][None, :] - self.logcum[:t])

    def limit_pct_on(self, date):
        """Band for every code on `date` (None -> nan): board, ST name rule and date rule."""
        return np.array([np.nan if (p := profile.limit_pct(code, name, date)) is None else p
                         for code, name in zip(self.codes, self.names)], dtype=float)


def _event_reference(prev, fenhong, song, peigu, peigujia):
    fh = float(fenhong or 0.0)
    sz = float(song or 0.0)
    pg = float(peigu or 0.0)
    pgj = float(peigujia or 0.0)
    denom = 1.0 + sz / 10.0 + pg / 10.0
    ref = (prev - fh / 10.0 + pg / 10.0 * pgj) / denom
    return ref


def build_panel(force=False):
    npz = config.CACHE_DIR / 'panel.npz'
    meta_file = config.CACHE_DIR / 'panel_meta.json'
    wanted = data_dates()
    if npz.exists() and meta_file.exists() and not force:
        meta = json.loads(meta_file.read_text(encoding='utf-8'))
        # A day whose daily bar came from its minute file is stale once that file changes (e.g. a file fetched during
        # the session replaced by the finished day).
        sizes = meta.get('minute_file_sizes', {})
        changed = [d for d in meta.get('minute_derived_days', []) if d in wanted and _file_size(d) != sizes.get(d)]
        stale = meta.get('version') != config.DATA_VERSION
        if set(wanted) <= set(meta['dates']) and not changed and not stale:
            z = np.load(npz, allow_pickle=False)
            arrays = {k: z[k] for k in z.files}
            return Panel(meta['dates'], meta['codes'], meta['names'], meta['industries'], arrays, meta['xdxr_codes'])
        why = (f'built by an older engine ({meta.get("version")})' if stale
               else f'minute files changed since it was built ({", ".join(changed)})' if changed
               else 'D:/data has trading days the cached panel lacks')
        print(f'panel: {why}; rebuilding', flush=True)

    universe = load_universe()
    raw = load_daily_raw()
    xdxr = load_xdxr()
    xdxr_codes = json.loads((config.CACHE_DIR / 'xdxr_codes.json').read_text(encoding='utf-8'))
    raw = raw[raw['code'].str.match(r'^\d{6}$')]
    # Days in D:/data after the official daily history: derive their daily bars from the minute bars.
    minute_days = [d for d in wanted if d > raw['date'].max()]
    if minute_days:
        raw = pd.concat([raw, minute_daily(minute_days)], ignore_index=True)
        raw = raw[raw['code'].str.match(r'^\d{6}$')]
    # A daily bar exists only when the stock traded: TDX's daily history also has flat zero-volume rows for a
    # suspended stock. Without them a suspension is a gap, as the history features expect (it does not break a
    # limit-up run, averages skip it, nothing can be bought or sold on it).
    raw = raw[raw['volume'] > 0]
    codes = sorted(raw['code'].unique())
    dates = sorted(raw['date'].unique())
    ci = {c: j for j, c in enumerate(codes)}
    di = {d: i for i, d in enumerate(dates)}
    n_d, n_c = len(dates), len(codes)

    def grid(column):
        out = np.full((n_d, n_c), np.nan)
        out[raw['date'].map(di).to_numpy(), raw['code'].map(ci).to_numpy()] = raw[column].to_numpy(dtype=float)
        return out

    o, h, l, c, v, a = (grid(k) for k in ('open', 'high', 'low', 'close', 'volume', 'amount'))
    has = np.isfinite(c)
    # Previous available raw close for each cell (suspensions carry the last close forward).
    prev_close = pd.DataFrame(c).ffill().shift(1).to_numpy()

    # Ex-rights: reference price on the ex-date and the log factor used for forward adjustment.
    ref = prev_close.copy()
    logf = np.zeros((n_d, n_c))
    events = xdxr[(xdxr['category'] == 1) & xdxr['code'].isin(ci)]
    applied = []
    for row in events.itertuples(index=False):
        j = ci[row.code]
        # Apply on the first trading row of this code on or after the ex-date.
        start = int(np.searchsorted(np.asarray(dates), row.date))
        rows_on_after = np.nonzero(has[start:, j])[0]
        if not len(rows_on_after):
            continue
        i = start + int(rows_on_after[0])
        p = prev_close[i, j]
        if not (p > 0):
            continue
        r = _event_reference(p, row.fenhong, row.songzhuangu, row.peigu, row.peigujia)
        if not (r > 0) or abs(r / p - 1.0) < 1e-6:
            continue
        ref[i, j] = r
        logf[i, j] += math.log(r / p)
        applied.append((row.code, dates[i]))
    logcum = np.cumsum(logf, axis=0)

    names = universe['name'].reindex(codes).fillna('').to_numpy(dtype=object)
    industries = universe['industry'].reindex(codes).fillna('').to_numpy(dtype=object)

    # Price band per cell: board, ST-by-name date rule, first five trading rows of a new listing.
    pct = np.full((n_d, n_c), np.nan)
    before = np.array([np.nan if (p := profile.limit_pct(code, name, '2026-07-05')) is None else p
                       for code, name in zip(codes, names)])
    after = np.array([np.nan if (p := profile.limit_pct(code, name, '2026-07-06')) is None else p
                      for code, name in zip(codes, names)])
    switch = int(np.searchsorted(np.asarray(dates), config.MAIN_ST_TEN_PERCENT_DATE))
    pct[:switch] = before[None, :]
    pct[switch:] = after[None, :]
    first_row = np.argmax(has, axis=0)
    for j in range(n_c):
        if has[:, j].any() and first_row[j] > 0:  # listed inside the window: first 5 rows have no limit
            rows = np.nonzero(has[:, j])[0][:5]
            pct[rows, j] = np.nan
    limit_up = profile.round_tick(ref * (1 + pct / 100.0))
    limit_down = profile.round_tick(ref * (1 - pct / 100.0))
    lu_seal = has & profile.at_or_above(c, limit_up)
    lu_touch = has & profile.at_or_above(h, limit_up)
    ld_seal = has & profile.at_or_below(c, limit_down)
    ret_scaled = (c / ref - 1.0) * 100.0 * 10.0 / pct

    arrays = dict(open=o, high=h, low=l, close=c, volume=v, amount=a, has=has, ref=ref, logcum=logcum, pct=pct,
                  limit_up=limit_up, limit_down=limit_down, lu_seal=lu_seal, lu_touch=lu_touch, ld_seal=ld_seal,
                  ret_scaled=ret_scaled)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, **arrays)
    meta_file.write_text(json.dumps({'dates': dates, 'codes': codes, 'names': list(names),
                                     'industries': list(industries), 'xdxr_codes': xdxr_codes,
                                     'events_applied': len(applied), 'minute_derived_days': minute_days,
                                     'minute_file_sizes': {d: _file_size(d) for d in minute_days},
                                     'version': config.DATA_VERSION},
                                    ensure_ascii=False), encoding='utf-8')
    return Panel(dates, codes, names, industries, arrays, xdxr_codes)


def float_shares_asof(xdxr, codes, date):
    """Latest float share count (shares) on or before `date`, per code; nan when unknown."""
    rows = xdxr[(xdxr['date'] <= date) & (xdxr['panhouliutong'].fillna(0) > 0)]
    latest = rows.sort_values('date').groupby('code')['panhouliutong'].last() * 1e4
    return latest.reindex(codes).to_numpy(dtype=float)
