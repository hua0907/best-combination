"""Fetch TDX corporate-action (xdxr) records.

    python -X utf8 fetch_xdxr.py            # only codes that have no record yet
    python -X utf8 fetch_xdxr.py --refresh  # every code again (do this before analysing new trading days)

Records go to <roundtable>/cache/xdxr/<code>.json; youlong's folder is only read. The record publication
time is unknown (historicalPublicationTimeKnown=False); the engine only applies an event from its ex-date on.
After a refresh the derived caches (xdxr.parquet, panel.npz) are deleted so the next run rebuilds them.
"""
import argparse
import datetime
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from pytdx.hq import TdxHq_API

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import config  # noqa: E402

HOSTS = [('59.36.5.11', 7709), ('220.178.55.71', 7709), ('60.191.117.167', 7709), ('115.238.90.165', 7709),
         ('220.178.55.86', 7709), ('180.153.18.172', 80), ('180.153.18.170', 7709), ('218.75.126.9', 7709),
         ('115.238.56.198', 7709), ('117.34.114.15', 7709)]
_tls = threading.local()


def market_of(code):
    if code.startswith(('4', '8', '92')):
        return 2
    if code.startswith(('5', '6', '9')):
        return 1
    return 0


def api():
    if not hasattr(_tls, 'api'):
        seed = getattr(_tls, 'seed', threading.get_ident() % len(HOSTS))
        last = None
        for off in range(len(HOSTS)):
            ip, port = HOSTS[(seed + off) % len(HOSTS)]
            client = TdxHq_API(heartbeat=False, auto_retry=False, raise_exception=True)
            try:
                if client.connect(ip, port, time_out=3):
                    _tls.api = client
                    break
            except Exception as exc:  # try the next host
                last = exc
        if not hasattr(_tls, 'api'):
            raise RuntimeError(f'no tdx host: {last}')
    return _tls.api


def reset():
    try:
        if hasattr(_tls, 'api'):
            _tls.api.disconnect()
    except Exception:
        pass
    if hasattr(_tls, 'api'):
        delattr(_tls, 'api')
    _tls.seed = (getattr(_tls, 'seed', 0) + 1) % len(HOSTS)


def write_json(path, obj):
    tmp = f'{path}.{os.getpid()}.{threading.get_ident()}.tmp'
    with open(tmp, 'w', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, separators=(',', ':'))
    os.replace(tmp, path)


def fetch(code, dest):
    errors = []
    for _ in range(4):
        try:
            rows = api().get_xdxr_info(market_of(code), code)
            if not isinstance(rows, list):
                raise ValueError('INVALID_XDXR_RESPONSE')
            clean = [{k: v for k, v in row.items() if k != 'name'} for row in rows]
            write_json(os.path.join(dest, code + '.json'), {
                'code': code, 'source': 'tdx-get_xdxr_info', 'status': 'ok', 'rows': clean,
                'fetchedAt': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'historicalPublicationTimeKnown': False})
            return 'ok'
        except Exception as exc:
            errors.append(type(exc).__name__ + ':' + str(exc))
            reset()
            time.sleep(0.1)
    write_json(os.path.join(dest, code + '.json'), {'code': code, 'status': 'error', 'errors': errors})
    return 'error'


def all_codes():
    """Universe snapshot plus every code in the latest D:/data file (catches later listings)."""
    codes = {row['code'] for row in json.loads(config.UNIVERSE_FILE.read_text(encoding='utf-8'))['rows']}
    latest = max(config.DATA_DIR.glob('*.csv'), default=None)
    if latest is not None:
        import pyarrow as pa
        import pyarrow.csv as pacsv
        table = pacsv.read_csv(latest, convert_options=pacsv.ConvertOptions(
            include_columns=['code'], column_types={'code': pa.string()}))
        codes |= {c for c in set(table.column('code').to_pylist()) if c and len(c) == 6 and c.isdigit()}
    return codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true', help='fetch every code again')
    args = ap.parse_args()
    dest = str(config.CACHE_DIR / 'xdxr')
    os.makedirs(dest, exist_ok=True)
    have = set()
    for folder in ([] if args.refresh else config.XDXR_DIRS):
        if folder.exists():
            for name in os.listdir(folder):
                if name.endswith('.json'):
                    try:
                        if json.loads((folder / name).read_text(encoding='utf-8')).get('status') == 'ok':
                            have.add(name[:-5])
                    except Exception:
                        pass
    todo = sorted(all_codes() - have)
    print('xdxr: have', len(have), 'to fetch', len(todo), flush=True)
    counts = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = [pool.submit(fetch, code, dest) for code in todo]
        for i, job in enumerate(as_completed(jobs), 1):
            status = job.result()
            counts[status] = counts.get(status, 0) + 1
            if i % 250 == 0 or i == len(jobs):
                print('xdxr progress', i, '/', len(jobs), counts, flush=True)
    print('xdxr DONE', counts, flush=True)
    if args.refresh or counts.get('ok'):
        stale = [config.CACHE_DIR / n for n in ('xdxr.parquet', 'xdxr_codes.json', 'panel.npz', 'panel_meta.json')]
        stale += sorted(config.CACHE_DIR.glob('market_*.json'))
        for path in stale:
            if path.exists():
                path.unlink()
                print('removed stale', path, flush=True)


if __name__ == '__main__':
    main()
