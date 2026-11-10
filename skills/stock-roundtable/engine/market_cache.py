"""Market readings of earlier days, one file per meeting time (cache/market_<HHMM>.json).

The emotion-stage rules compare a day with every earlier full-market day at the same time of day, and a
stage also depends on the previous day's stage. A single-day meeting therefore needs the same chain of
readings the backtest builds day by day. Entries are kept as an unbroken prefix of D:/data days; a day
whose CSV changed size invalidates itself and everything after it.
"""
import json

from . import config


def _path(key):
    return config.CACHE_DIR / f'market_{key.replace(":", "")}.json'


def file_size(date):
    return (config.DATA_DIR / f'{date}.csv').stat().st_size


def load(key):
    path = _path(key)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('version') != config.DATA_VERSION or data.get('time') != key:
        return {}
    return data['days']


def save(key, days):
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {'version': config.DATA_VERSION, 'time': key, 'days': dict(sorted(days.items()))}
    _path(key).write_text(json.dumps(payload, ensure_ascii=False, default=float), encoding='utf-8')


def entry(date, codes, market=None, stage=None):
    """One day's record; market and stage only for full-market days."""
    record = {'size': file_size(date), 'codes': int(codes)}
    if market is not None:
        record['metrics'], record['stage'] = market, stage
    return record
