"""Paths, meeting times and frozen thresholds. Nothing here is tuned on backtest outcomes."""
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

VERSION = 'stock-roundtable-v2.2'
# Format of the cached market readings (cache/market_<HHMM>.json) and of the daily panel (cache/panel.npz). Bump it
# only when a change alters them; v1.2 (verdict), v1.3 and v1.4 (debater set), v2.0 (consensus process) and v2.1 (the
# after-close review moved to 15:10, so its readings are cached as market_1510.json) left them alone. v2.2: a bar
# without volume is not a trade (TDX writes flat zero-volume bars for suspended stocks), which changes the panel and
# the market readings of days with suspended stocks.
DATA_VERSION = 'stock-roundtable-v2.2'

# Inputs (all read-only).
DATA_DIR = Path(r'D:\data')
YOULONG = Path(r'D:\youlong')
DAILY_RAW_DIR = YOULONG / '.catdesk-output' / 'pit-cache-data24-t0ev035' / '_daily-raw'
UNIVERSE_FILE = YOULONG / '.catdesk-output' / 'pit-cache-data20' / 'universe.json'

# Outputs and caches live with the skill (this file is <roundtable>/engine/config.py).
# STOCK_ROUNDTABLE_ROOT overrides the location, e.g. to test a working copy against the installed cache.
ROUNDTABLE_ROOT = Path(os.environ.get('STOCK_ROUNDTABLE_ROOT') or Path(__file__).resolve().parents[1])
SKILLS_ROOT = ROUNDTABLE_ROOT.parent
CACHE_DIR = ROUNDTABLE_ROOT / 'cache'
OUTPUT_DIR = ROUNDTABLE_ROOT / 'output'
# Ex-rights records; for a code present in both folders the first one wins (fetch_xdxr.py --refresh
# rewrites the roundtable's own folder with current records).
XDXR_DIRS = [
    CACHE_DIR / 'xdxr',
    YOULONG / '.catdesk-tmp' / 'v8-trend-comparison-20260927' / 'corporate-actions',
]

# Exchange rule: main-board risk-warning (ST) stocks move from a 5% to a 10% band on this date.
MAIN_ST_TEN_PERCENT_DATE = '2026-07-06'

# A CSV day counts as a full-market day when it carries at least this many codes.
FULL_MARKET_MIN_CODES = 5400

# The after-close review. The day's last minute bar (15:00) already holds the closing call auction (14:57-15:00), so
# the review reads the same 240 bars whenever it is held; it is held at 15:10 so that the closing prices are final
# before anything reads them. On the day itself, nothing treats the day as finished before this Beijing time.
CLOSE_REVIEW = '15:10'
BEIJING = timezone(timedelta(hours=8))  # exchange time; China has no daylight saving time
# A day's file counts as finished when at least this share of its stocks has the 15:00 bar. Every D:/data day through
# 2026-09-30 has it for 100% of its stocks on full-market days (lowest of all days 99.2%, 2026-05-11); a file
# fetched before the close has none.
CLOSE_BAR_SHARE = 0.99

# A stock needs this many prior daily bars before it can be a candidate.
MIN_HISTORY_DAYS = 20

# 新逻辑 F3：位置体检闸门（pick_qc.inspect）。默认关闭 —— 旧引擎的规则里本来没有"位置"这一维。
# 打开后：ctx.pool 会多一列 qc_pass，debaters.build() 只放行 True 的票，
# 也就是"辩手只能在过闸的池子里选"。这正是新逻辑与旧引擎的差别，所以打开后选票会变。
POSITION_GATE = False
# 闸门要求哪几项通过。默认"低吸 + 可买"：低吸挡追高，可买挡一字。
POSITION_GATE_KEYS = ('低吸', '可买')

def _clock(time_str):
    """'HH:MM' -> minutes after midnight."""
    try:
        hour, minute = (int(x) for x in time_str.split(':'))
    except ValueError:
        raise ValueError(f'时间格式应为 HH:MM，收到 {time_str!r}') from None
    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise ValueError(f'时间格式应为 HH:MM，收到 {time_str!r}')
    return hour * 60 + minute


def bars_until(time_str):
    """Minute bars visible at cutoff 'HH:MM' (09:31 -> 1, 10:00 -> 30, 11:30 -> 120, 13:01 -> 121, 15:00 -> 240)."""
    m = _clock(time_str)
    if 9 * 60 + 31 <= m <= 11 * 60 + 30:
        return m - (9 * 60 + 30)
    if 13 * 60 + 1 <= m <= 15 * 60:
        return 120 + m - 13 * 60
    raise ValueError(f'{time_str} 不是连续竞价时间：应在 09:31–11:30 或 13:01–15:00 之间')


def scaled(k, at_1000, at_close):
    """An amount threshold written for 10:00 (30 bars) and for the close (240 bars), at k visible bars.

    Linear in elapsed bars between the two anchors, proportional to elapsed bars before 10:00. At 10:00 and
    at the close it returns the anchor values exactly, so the frozen backtest thresholds are unchanged.
    """
    if k >= 240:
        return at_close
    if k <= 30:
        return at_1000 * k / 30
    return at_1000 + (at_close - at_1000) * (k - 30) / 210


def meeting(time_str):
    """Meeting held at 'HH:MM'. 09:31-11:30 and 13:01-14:59: intraday, buy at that minute's price.
    15:00 or later: the after-close review, held at CLOSE_REVIEW on the whole day's bars, buy at the next trading
    day's open."""
    if _clock(time_str) >= 15 * 60:
        return {'key': CLOSE_REVIEW, 'mode': 'eod', 'label': f'收盘后 {CLOSE_REVIEW}', 'cutoff': CLOSE_REVIEW,
                'cutoff_index': 240, 'entry': '次日开盘价买入（开盘即在涨停价的按涨停价排队，之后开板才算成交）',
                'min_amount': 8e7}
    try:
        k = bars_until(time_str)
    except ValueError:
        raise ValueError(f'{time_str} 不能开会：盘中会议在 09:31–11:30、13:01–14:59，收盘后复盘在 {CLOSE_REVIEW}'
                         '（15:00 以后的时间都按收盘后复盘）') from None
    return {'key': time_str, 'mode': 'intraday', 'label': f'盘中 {time_str}', 'cutoff': time_str, 'cutoff_index': k,
            'entry': f'当日 {time_str} 的价格买入；{time_str} 已封涨停的只能按涨停价排队，之后开板才算成交',
            'min_amount': scaled(k, 2e7, 8e7)}


def too_early(date, now=None):
    """Why `date` cannot be read as a finished day yet by the clock alone ('' when it can): on the day itself, before
    CLOSE_REVIEW Beijing time."""
    now = now or datetime.now(BEIJING)
    if date == now.strftime('%Y-%m-%d') and now.strftime('%H:%M') < CLOSE_REVIEW:
        return f'北京时间现在 {now:%H:%M}，{date} 要到收盘后 {CLOSE_REVIEW} 才算收完'
    return ''


# The two meetings of the backtest.
MODES = {'intraday': meeting('10:00'), 'eod': meeting(CLOSE_REVIEW)}

# Board-scaled strength: a move is multiplied by 10 / limit_pct so 10cm, 20cm and 30cm boards compare.
STRONG_SCALED = 5.0          # "strong" stock threshold, board-scaled percent
CLUSTER_MIN_STRONG = 6       # need this many strong stocks before looking for co-moving clusters
CLUSTER_CORR = {'intraday': 0.6, 'eod': 0.5}
CLUSTER_MIN_SIZE = 3

# Emotion-stage baselines: rolling percentiles over this many prior full-market days of the same mode;
# until enough days exist, the fixed defaults below are used (market-folklore levels, not fitted).
STAGE_LOOKBACK = 20
STAGE_MIN_HISTORY = 10
STAGE_DEFAULTS = {
    'intraday': {'lu_hot': 60, 'lu_cold': 15, 'ld_cold': 15, 'zb_high': 0.35, 'prem_hot': 2.5, 'prem_cold': -1.5},
    'eod': {'lu_hot': 90, 'lu_cold': 30, 'ld_cold': 20, 'zb_high': 0.35, 'prem_hot': 3.0, 'prem_cold': -1.5},
}

# Meeting protocol (v2.0): each debater chooses up to NOMINATIONS_PER_DEBATER stocks from its own framework, or none;
# nobody votes on anyone else's choice; the moderator only counts. A stock chosen by at least CONSENSUS_MIN debaters
# is a consensus pick (ranked by that count); a meeting without one stays in cash.
NOMINATIONS_PER_DEBATER = 2
CONSENSUS_MIN = 2
