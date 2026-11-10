"""Point-in-time audit: re-run a decision with every value after its cutoff replaced by random noise.

If the meeting depends only on data visible at the cutoff, the re-run must produce the same market reads, the
same choices from every debater and the same consensus picks.
"""
import copy

import numpy as np

from . import context, minutes, roundtable, store


def signature(db):
    """Everything the decision part of a record depends on: each debater's read and choices, the consensus picks."""
    opening = tuple((d.key, round(db.views[d.key].exposure, 4), tuple((int(i), m) for i, m, _, _ in noms), abst)
                    for d, noms, abst in db.opening)
    picks = tuple((ch.code, ch.count, ch.rank, ch.sealed) for ch in db.picks)
    return opening, picks


def scrambled(panel, t, day, k, xdxr, date, rng):
    """Copies of the inputs in which every value after the decision cutoff is random noise."""
    p2 = copy.copy(panel)
    for name in ('open', 'high', 'low', 'close', 'volume', 'amount', 'ret_scaled'):
        arr = getattr(panel, name).copy()
        arr[t:] = rng.uniform(0.5, 200.0, size=arr[t:].shape)
        setattr(p2, name, arr)
    for name in ('has', 'lu_seal', 'lu_touch', 'ld_seal'):
        arr = getattr(panel, name).copy()
        arr[t:] = rng.random(arr[t:].shape) < 0.5
        setattr(p2, name, arr)
    for name in ('ref', 'pct', 'limit_up', 'limit_down', 'logcum'):  # row t holds pre-open facts and stays
        arr = getattr(panel, name).copy()
        arr[t + 1:] = rng.uniform(0.5, 200.0, size=arr[t + 1:].shape)
        setattr(p2, name, arr)
    d2 = copy.copy(day)
    if k < len(minutes.TIMES):
        for name in ('open', 'high', 'low', 'close', 'vol', 'amt'):
            arr = getattr(day, name).copy()
            arr[:, k:] = rng.uniform(0.5, 200.0, size=arr[:, k:].shape)
            setattr(d2, name, arr)
        real = day.real.copy()
        real[:, k:] = rng.random(real[:, k:].shape) < 0.5
        d2.real = real
    x2 = xdxr.copy()
    later = x2['date'] > date
    x2.loc[later, 'panhouliutong'] = rng.uniform(1.0, 1e6, size=int(later.sum()))
    return p2, d2, x2


def check(panel, t, date, spec, day, hist, xdxr, debaters, db, rng=None):
    """True when the decision is unchanged after randomising everything past the cutoff."""
    rng = rng if rng is not None else np.random.default_rng(20261003)
    p2, d2, x2 = scrambled(panel, t, day, spec['cutoff_index'], xdxr, date, rng)
    fs2 = store.float_shares_asof(x2, panel.codes, date)
    db2 = roundtable.run(context.build(p2, t, date, spec, d2, hist, fs2), debaters)
    return signature(db) == signature(db2)
