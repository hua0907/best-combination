"""Step 1: make sure D:/data has the review day's complete 1-minute file.

A file counts as complete when at least 99% of its stocks have the 15:00 bar (the engine's own rule). For today the
shared downloader (D:/youlong/scripts/tdx-daily-download.py) runs in its --early mode, which writes the day only once
that holds; this step retries every few minutes until the deadline. An existing complete file is never replaced.
"""
import time

from common import CLOSE_BAR_SHARE, CONFIG, LOGS, close_bar_share, data_file, now, run_script, sha256


def pull(day, today, log):
    path = data_file(day)
    deadline = now().replace(hour=int(CONFIG['data_deadline'][:2]), minute=int(CONFIG['data_deadline'][3:]), second=0)
    retry = 60 * float(CONFIG['retry_minutes'])
    attempts = 0
    while True:
        if path.exists():
            stocks, share, rows = close_bar_share(path)
            info = {'file': str(path), 'stocks': stocks, 'rows': rows, 'close_bar_share': round(share, 4),
                    'sha256': sha256(path), 'downloads': attempts}
            if share >= CLOSE_BAR_SHARE:
                return 'ok', f'{stocks} 只股票，{share:.1%} 有 15:00 收盘那根', info
            return 'failed', f'文件存在但只有 {share:.1%} 的股票有 15:00 那根（要求 ≥99%），没有覆盖它', info
        if day != today:
            return 'failed', f'D:\\data 里没有 {day}.csv', {'file': str(path)}
        attempts += 1
        rc, tail, secs = run_script([CONFIG['downloader'], '--day', day.isoformat(), '--early'],
                                    CONFIG['downloader_cwd'], LOGS / f'{day}-download-{attempts}.out', timeout=900)
        log(f'download attempt {attempts}: exit {rc} in {secs}s; {tail}')
        if path.exists():
            continue
        if now() >= deadline:
            return 'not-ready', f'到 {CONFIG["data_deadline"]} 当天数据仍未收完（下载 {attempts} 次）', {'downloads': attempts}
        time.sleep(retry)
