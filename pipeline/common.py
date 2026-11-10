"""Shared paths, settings and helpers of the D:/review pipeline (see README.md)."""
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                      # D:/review
CONFIG = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
SKILLS = ROOT / 'skills'                                        # frozen snapshot: the engine and the nine SKILL.md
ENGINE_DIR = SKILLS / 'stock-roundtable'                        # frozen copy of the roundtable engine
DATA_DIR = Path(CONFIG['data_dir'])
REPORTS = ROOT / 'reports'
LEDGER = ROOT / 'ledger'
LOGS = ROOT / 'logs'
PYTHON = CONFIG['python'] if Path(CONFIG['python']).exists() else sys.executable
SH_TZ = dt.timezone(dt.timedelta(hours=8))
REVIEW_TIME = dt.time(15, 10)
CLOSE_BAR_SHARE = 0.99          # a day's file is complete when this share of its stocks has the 15:00 bar (engine rule)
for folder in (REPORTS, LEDGER, LOGS):
    folder.mkdir(parents=True, exist_ok=True)


def now():
    return dt.datetime.now(SH_TZ)


class Log:
    """Appends to logs/<day>.log and prints."""

    def __init__(self, day):
        self.path = LOGS / f'{day}.log'

    def __call__(self, msg):
        line = f'[{now():%Y-%m-%d %H:%M:%S}] {msg}'
        print(line, flush=True)
        with open(self.path, 'a', encoding='utf-8') as fh:
            fh.write(line + '\n')


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def run_script(args, cwd, out_file, timeout):
    """Run a Python script (of the frozen engine or the downloader) to completion; its output goes to out_file."""
    env = {k: v for k, v in os.environ.items() if k != 'STOCK_ROUNDTABLE_ROOT'}
    env['PYTHONIOENCODING'] = 'utf-8'
    t0 = time.time()
    with open(out_file, 'w', encoding='utf-8') as fh:
        p = subprocess.run([PYTHON, '-X', 'utf8'] + [str(a) for a in args], cwd=str(cwd), env=env, stdout=fh,
                           stderr=subprocess.STDOUT, timeout=timeout)
    tail = Path(out_file).read_text(encoding='utf-8', errors='replace').strip().splitlines()[-3:]
    return p.returncode, ' | '.join(tail), round(time.time() - t0, 1)


def frozen_status():
    """Compare the frozen engine with the fingerprints taken when it was frozen (skills/FROZEN.json)."""
    manifest = json.loads((SKILLS / 'FROZEN.json').read_text(encoding='utf-8'))
    changed = [name for name, digest in manifest['files'].items()
               if not (SKILLS / name).exists() or sha256(SKILLS / name) != digest]
    return manifest, changed


def _step_weekday(day, step, holidays):
    d = day + dt.timedelta(days=step)
    while d.weekday() >= 5 or d.isoformat() in holidays:
        d += dt.timedelta(days=step)
    return d


def calendar_info(day):
    """(is a trading day, previous trading day, next trading day, source).

    The TDX workday service knows only days that have already traded (it reads the index's daily bars), so a future
    T+1 date is the next weekday that is not in config.json's holidays (add the exchange's closures when they are
    published). When TDX cannot be reached, weekdays are used throughout."""
    holidays = set(CONFIG.get('holidays', []))
    try:
        import eltdx
        client = eltdx.Client()
        client.connect()
        try:
            w = eltdx.WorkdayService(client)
            nxt, source = w.next_workday(day), 'TDX 交易日历'
            if nxt is None:
                nxt = _step_weekday(day, 1, holidays)
                source += '；T+1 日期按工作日推算（config.json 的 holidays 顺延）'
            return bool(w.is_workday(day)), w.previous_workday(day), nxt, source
        finally:
            client.close()
    except Exception as exc:  # noqa: BLE001 - any connection problem falls back to weekdays
        return (day.weekday() < 5 and day.isoformat() not in holidays, _step_weekday(day, -1, holidays),
                _step_weekday(day, 1, holidays), f'工作日推算（TDX 日历不可用：{type(exc).__name__}）')


def data_file(day):
    return DATA_DIR / f'{day}.csv'


def close_bar_share(path):
    """(stocks, share of them with the 15:00 bar, rows) of a D:/data day file."""
    import pandas as pd
    df = pd.read_csv(path, usecols=['symbol', 'datetime'], dtype=str)
    stocks = df['symbol'].nunique()
    closed = df.loc[df['datetime'].str.endswith('15:00:00'), 'symbol'].nunique()
    return stocks, (closed / stocks if stocks else 0.0), len(df)
