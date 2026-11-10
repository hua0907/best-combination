"""16:40 check: compare the review day's D:/data file with a fresh TDX fetch, in memory. Nothing in D:/data changes.

TDX 1-minute bars can differ a little between fetches; the page reports how many stocks changed and whether any stock
on the T+1 list is among them, so a review built on data that later changed is visible.
"""
import importlib.util

import pandas as pd

from common import CONFIG, data_file, now

COLS = ['symbol', 'exchange', 'code', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'amount']


def _downloader(log):
    spec = importlib.util.spec_from_file_location('tdx_daily_download', CONFIG['downloader'])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.log = lambda msg: log('  tdx: ' + msg)      # keep the shared download log for real downloads only
    return mod


def _per_stock(df):
    g = df.sort_values('datetime').groupby('symbol')
    return pd.DataFrame({'close': g['close'].last(), 'volume': g['volume'].sum()})


def verify(day, t1_codes, log):
    dl = _downloader(log)
    import eltdx
    client = eltdx.Client()
    client.connect()
    try:
        workdays = eltdx.WorkdayService(client)
        codes = client.codes.all_a_shares()
        depth = dl.bars_needed(workdays, [day], dl.dt.datetime.now(dl.SH_TZ))
        lines, _ = dl.download(client, [day], codes, depth)
    finally:
        client.close()
    fresh = pd.DataFrame([ln.split(',') for ln in lines[day]], columns=COLS)
    fresh[['close', 'volume']] = fresh[['close', 'volume']].astype(float)
    stored = pd.read_csv(data_file(day), usecols=['symbol', 'code', 'datetime', 'close', 'volume'], dtype={'code': str})
    bars = stored.merge(fresh[['symbol', 'datetime', 'close']], on=['symbol', 'datetime'], how='inner',
                        suffixes=('', '_new'))
    a, b = _per_stock(stored), _per_stock(fresh)
    j = a.join(b, how='outer', rsuffix='_new')
    close_changed = set(j.index[(j['close'] - j['close_new']).abs() > 1e-6])
    vol_changed = set(j.index[(j['volume'] - j['volume_new']).abs() > 0.03 * j['volume'].clip(lower=1)])
    sym_of = dict(zip(stored['code'], stored['symbol']))
    picks = {c: ('收盘价或成交量变了' if sym_of.get(c) in close_changed | vol_changed else '一致') for c in t1_codes}
    return {'checked_at': now().isoformat(timespec='seconds'), 'stocks_file': int(len(a)), 'stocks_fresh': int(len(b)),
            'close_changed': len(close_changed), 'volume_changed': len(vol_changed),
            'bar_close_diff_share': round(float((bars['close'] - bars['close_new']).abs().gt(1e-6).mean()), 4),
            'only_in_file': int(j['close_new'].isna().sum()), 'only_in_fresh': int(j['close'].isna().sum()),
            'picks': picks}
