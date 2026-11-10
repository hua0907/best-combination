"""Price-limit rules, ported from D:/youlong/public/security-profile.js (rule version 2026-08-19 v3)."""
import re

import numpy as np

from . import config

_ST = re.compile(r'(?:^|\*)ST', re.I)
_DELIST = re.compile(r'退市|退$')
_NO_LIMIT_NAME = re.compile(r'^[NC](?!ST)', re.I)


def board_of(code):
    if re.match(r'^(92|4|8)', code):
        return 'BSE_30'
    if re.match(r'^(300|301)', code):
        return 'CHINEXT_20'
    if re.match(r'^(688|689)', code):
        return 'STAR_20'
    if re.match(r'^(200|900)', code):
        return 'UNSUPPORTED'
    return 'MAIN_10'


def is_risk_warning(name):
    return bool(_ST.search(re.sub(r'\s', '', name or '')))


def is_delisting(name):
    return bool(_DELIST.search(re.sub(r'\s', '', name or '')))


def is_no_limit_name(name):
    return bool(_NO_LIMIT_NAME.match((name or '').strip()))


def limit_pct(code, name, date):
    """Percent band for this code on this date, or None when the stock has no price limit."""
    if is_no_limit_name(name):
        return None
    board = board_of(code)
    if board in ('CHINEXT_20', 'STAR_20'):
        return 20.0
    if board == 'BSE_30':
        return 30.0
    if board == 'MAIN_10':
        if is_risk_warning(name) and date < config.MAIN_ST_TEN_PERCENT_DATE:
            return 5.0
        return 10.0
    return None


def round_tick(values):
    """Exchange rounding to 0.01 (half up), matching Math.round(v / tick + EPSILON) * tick."""
    return np.floor(np.asarray(values, dtype=float) * 100.0 + 0.5 + 1e-9) / 100.0


def at_limit(price, limit):
    """Same tolerance as security-profile isAtLimit: max(0.51 tick, 0.005% of the limit)."""
    price = np.asarray(price, dtype=float)
    limit = np.asarray(limit, dtype=float)
    tol = np.maximum(0.0051, limit * 0.00005)
    return np.isfinite(price) & np.isfinite(limit) & (np.abs(price - limit) <= tol)


def at_or_above(price, limit):
    price = np.asarray(price, dtype=float)
    limit = np.asarray(limit, dtype=float)
    tol = np.maximum(0.0051, limit * 0.00005)
    return np.isfinite(price) & np.isfinite(limit) & (price >= limit - tol)


def at_or_below(price, limit):
    price = np.asarray(price, dtype=float)
    limit = np.asarray(limit, dtype=float)
    tol = np.maximum(0.0051, limit * 0.00005)
    return np.isfinite(price) & np.isfinite(limit) & (price <= limit + tol)
