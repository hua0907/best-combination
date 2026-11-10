"""Compare this project's frozen-engine outputs with the roundtable's current outputs (the results the optimal
combination was confirmed on). Prints a report and saves it as logs/consistency-<time>.md.

    python tests/check_consistency.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pipeline'))
from common import CONFIG, ENGINE_DIR, LOGS, now  # noqa: E402

REF = Path(CONFIG['engine_source']) / 'output'
NEW = ENGINE_DIR / 'output'
NOM_KEY = ['date', 'debater', 'order', 'code', 'model']
NOM_OUT = ['count', 'picked', 'filled', 'queued', 'entry', 'buy_chg', 'ret_close', 'sell_date', 't1_best', 't1_mid',
           't1_worst', 't1_stuck']


def compare(a, b, key, cols):
    m = a.merge(b, on=key, how='outer', suffixes=('_ref', '_new'), indicator=True)
    both = m[m['_merge'] == 'both']
    diffs = {}
    for c in cols:
        x, y = both[c + '_ref'], both[c + '_new']
        if x.dtype.kind in 'fi' and y.dtype.kind in 'fi':
            bad = ~np.isclose(x.astype(float), y.astype(float), equal_nan=True, rtol=0, atol=1e-9)
        else:
            bad = x.astype(str) != y.astype(str)
        if bad.any():
            diffs[c] = int(bad.sum())
    return {'rows': len(both), 'only_ref': int((m['_merge'] == 'left_only').sum()),
            'only_new': int((m['_merge'] == 'right_only').sum()), 'diffs': diffs}


def record(out):
    """The optimal combination's totals from one output folder (same arithmetic as optimal_combo.py)."""
    t = pd.read_csv(out / 'optimal-combo' / 'trades.csv', dtype={'code': str})
    settled = pd.read_csv(out / 'eod' / 'days.csv')['date'].astype(str)
    pending = set(t.loc[t['result'].astype(str).str.startswith('待'), 'review'])
    day = []
    for d in settled:
        if d in pending:
            continue
        x = t[t['review'] == d]
        v = np.where(x['result'].isin(['赚', '亏']), x['t1_mid'].astype(float), 0.0) if len(x) else np.array([0.0])
        day.append(float(v.mean()))
    done = t[t['result'].isin(['赚', '亏'])]
    cum = np.cumsum(day)
    return {'reviews': len(day), 'picks': int(t['review'].isin(settled).sum()), 'trades': len(done),
            'per_trade': round(float(done['t1_mid'].mean()), 4), 'win': round(float((done['t1_mid'] > 0).mean()), 4),
            'total': round(float(np.sum(day)), 4), 'max_dd': round(float((cum - np.maximum.accumulate(cum)).min()), 4)}


def main():
    L = [f'# 一致性核对 · {now():%Y-%m-%d %H:%M}', '', f'- 对照（圆桌当前结果）：{REF}', f'- 本项目（冻结引擎）：{NEW}', '']
    a_ref = json.loads((REF / 'audit.json').read_text(encoding='utf-8'))
    a_new = json.loads((NEW / 'audit.json').read_text(encoding='utf-8'))
    L += ['| 项目 | 对照 | 本项目 |', '|---|---|---|',
          f'| 引擎版本 | {a_ref["version"]} | {a_new["version"]} |',
          f'| 回测生成时间 | {a_ref["generated"]} | {a_new["generated"]} |',
          f'| 回测窗口 | {a_ref["window"][0]} – {a_ref["window"][-1]}（{len(a_ref["window"])} 天） | '
          f'{a_new["window"][0]} – {a_new["window"][-1]}（{len(a_new["window"])} 天） |',
          f'| 防偷看抽查 | {sum(x["identical"] for x in a_ref["scramble_audits"])}/{len(a_ref["scramble_audits"])} | '
          f'{sum(x["identical"] for x in a_new["scramble_audits"])}/{len(a_new["scramble_audits"])} |', '']
    ok = a_ref['window'] == a_new['window']
    L += ['## 每位辩手的每张选票（含买入和 T+1 结果）', '', '| 会议 | 共同的选票 | 只在对照 | 只在本项目 | 有差异的列 |', '|---|---|---|---|---|']
    for mode in ('eod', 'intraday'):
        r = compare(pd.read_csv(REF / mode / 'nominations.csv', dtype={'code': str}),
                    pd.read_csv(NEW / mode / 'nominations.csv', dtype={'code': str}), NOM_KEY, NOM_OUT)
        ok &= r['only_ref'] == 0 and r['only_new'] == 0 and not r['diffs']
        L.append(f'| {mode} | {r["rows"]} | {r["only_ref"]} | {r["only_new"]} | {r["diffs"] or "无"} |')
    vr, vn = (pd.read_csv(p / 'optimal-combo' / 'votes.csv', dtype=str) for p in (REF, NEW))
    v = compare(vr, vn, ['review'], [c for c in vr.columns if c != 'review'])
    tr, tn = (pd.read_csv(p / 'optimal-combo' / 'trades.csv', dtype={'code': str}) for p in (REF, NEW))
    t = compare(tr, tn, ['review', 'code'], [c for c in tr.columns if c not in ('review', 'code')])
    ok &= all(x['only_ref'] == 0 and x['only_new'] == 0 and not x['diffs'] for x in (v, t))
    L += ['', '## 最优组合', '', '| 文件 | 共同的行 | 只在对照 | 只在本项目 | 有差异的列 |', '|---|---|---|---|---|',
          f'| votes.csv（每场五位的第一选择） | {v["rows"]} | {v["only_ref"]} | {v["only_new"]} | {v["diffs"] or "无"} |',
          f'| trades.csv（每只符合规则的票） | {t["rows"]} | {t["only_ref"]} | {t["only_new"]} | {t["diffs"] or "无"} |', '']
    rr, rn = record(REF), record(NEW)
    L += ['| 汇总 | 对照 | 本项目 |', '|---|---|---|'] + [f'| {k} | {rr[k]} | {rn[k]} |' for k in rr]
    ok &= rr == rn
    L += ['', f'**结论：{"完全一致" if ok else "有差异，见上表"}**', '']
    text = '\n'.join(L)
    out = LOGS / f'consistency-{now():%Y%m%d-%H%M}.md'
    out.write_text(text, encoding='utf-8')
    print(text)
    print(f'saved {out}')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
