"""Step 6: the fixed review page reports/<date>.html, plus latest.html, index.html and ledger/signals.csv.

Every number comes from the frozen engine: the day's review (reports/<date>.json, built by review_day.py) and the
optimal-combination record (output/optimal-combo/trades.csv, votes.csv). Every result is given in the engine's three
cases of the sale on the trading day after the buy: best (at that day's high), worst (at its low) and average (at the
midpoint of the two, the backtest's basis). The ledger also lists next-day open and close returns, priced from the
engine's daily panel exactly as the engine prices its midpoint sale.

    python pipeline/render.py      # re-render every page, latest.html, index.html and the ledger from reports/*.json
"""
import html
import json
import re
import shutil
import sys

import numpy as np
import pandas as pd

from common import CONFIG, ENGINE_DIR, LEDGER, REPORTS

sys.path.insert(0, str(ENGINE_DIR))
from engine import store  # noqa: E402

OUT = ENGINE_DIR / 'output' / 'optimal-combo'
SELL = CONFIG.get('sell_rule', 'open')
RULE_NAME = {'open': '次日开盘价卖出', 'close': '次日收盘价卖出', 'mid': '次日最高/最低均价卖出（回测口径）'}
CASES = (('t1_best', '最好', '最高价卖出'), ('t1_worst', '最差', '最低价卖出'), ('t1_mid', '平均', '均价卖出'))
CASE_NOTE = ('结果给三种情况，都是买入后的下一个交易日卖出：最好 = 当天最高价卖出，最差 = 当天最低价卖出，'
             '平均 = 最高价和最低价的均价卖出（回测口径）；那天不管什么时候卖，收益都在最好和最差之间。')
DEFAULT_RULE = {'members': ['炒股养家', '92科比', '陈小群', '赵老哥', '乔帮主'], 'min_votes': 2,
                'selected_through': '2026-10-13'}
WEEK = '一二三四五六日'
CSS = """
:root{--ink:#1f2933;--muted:#52606d;--line:#cbd2d9;--bg:#f7f8fa;--green:#0b6e4f;--green-bg:#e3f4ec;--red-bg:#fef3f2;
--amber-bg:#fff7e6;--amber:#f5c46b;--blue:#1d4ed8}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 "Microsoft YaHei","PingFang SC","Noto Sans CJK SC",sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 18px 60px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:19px;margin:30px 0 8px;padding-top:10px;border-top:1px solid var(--line)}
h3{font-size:16px;margin:16px 0 6px}.sub{color:var(--muted);margin:0 0 12px}
.card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:12px 0}
.big{font-size:17px}.ok{background:var(--green-bg);border-color:#9fd3bb}.warn{background:var(--amber-bg);border-color:var(--amber)}
.bad{background:var(--red-bg);border-color:#fda29b}.flat{font-size:22px;font-weight:700;color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:14px;background:#fff}
th,td{border:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}th{background:#eef1f4;white-space:nowrap}
td.hit{background:var(--green-bg);font-weight:700;color:var(--green)}td.off{color:#7b8794}
.tag{display:inline-block;padding:0 6px;border-radius:4px;font-size:12px;background:#eef1f4;color:var(--muted);margin-left:4px}
.pos{color:#b42318;font-weight:700}.neg{color:#027a48;font-weight:700}.small{font-size:13px;color:var(--muted)}
ul{margin:6px 0;padding-left:22px}li{margin:3px 0}details{margin:8px 0}summary{cursor:pointer;color:var(--muted)}a{color:var(--blue)}
"""


def esc(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ''
    return html.escape(str(x))


def md(x):
    return re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', esc(x))


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def pc(x):
    v = num(x)
    if not np.isfinite(v):
        return '—'
    cls = 'pos' if v > 0 else 'neg' if v < 0 else ''
    return f'<span class="{cls}">{v:+.2f}%</span>'


def rate(x):
    v = num(x)
    return '—' if not np.isfinite(v) else f'{v:.0%}'


def price(x):
    v = num(x)
    return '—' if not np.isfinite(v) else f'{v:.2f}'


def yes(x):
    return str(x).strip().lower() in ('true', '1', '1.0')


def trades_with_rules():
    """trades.csv of the frozen optimal_combo.py plus next-day open and close returns."""
    t = pd.read_csv(OUT / 'trades.csv', dtype={'code': str})
    t['filled'] = t['filled'].map(yes)
    t['stuck'] = t['stuck'].map(yes)
    t['ret_open'] = np.nan
    t['ret_close_t1'] = np.nan
    panel = store.build_panel()
    for i, r in t.iterrows():
        if r['result'] not in ('赚', '亏'):
            continue
        tb, j = panel.index_of(r['buy_date']), panel.code_index[r['code']]
        s = tb + 1
        if s >= len(panel.dates) or panel.dates[s] != r['sell_date'] or not panel.has[s, j]:
            continue
        base = r['entry'] * panel.ref[s, j] / panel.close[tb, j]     # the engine's basis for the midpoint sale
        t.at[i, 'ret_open'] = (panel.open[s, j] / base - 1) * 100
        t.at[i, 'ret_close_t1'] = (panel.close[s, j] / base - 1) * 100
    return t


def settled_reviews():
    return set(pd.read_csv(ENGINE_DIR / 'output' / 'eod' / 'days.csv')['date'].astype(str))


def daily(t, votes, col):
    """Per settled review day: one stake split equally among the day's picks (not bought or not sold = cash)."""
    settled = settled_reviews()
    pending = set(t.loc[t['result'].astype(str).str.startswith('待'), 'review'])
    rows = []
    for d in votes['review'].astype(str):
        if d not in settled or d in pending:
            continue
        x = t[t['review'] == d]
        if x.empty:
            rows.append((d, 0, 0.0))
            continue
        v = np.where(x['result'].isin(['赚', '亏']), x[col].astype(float).fillna(0.0), 0.0)
        rows.append((d, len(x), float(v.mean())))
    return pd.DataFrame(rows, columns=['review', 'signals', 'ret'])


def summary(t, days, col, since=None):
    if since:
        t, days = t[t['review'] > since], days[days['review'] > since]
    r = t.loc[t['result'].isin(['赚', '亏']), col].astype(float).dropna()
    cum = days['ret'].cumsum()
    return {'days': len(days), 'signal_days': int((days['signals'] > 0).sum()), 'trades': len(r),
            'win': (r > 0).mean() if len(r) else np.nan, 'avg': r.mean() if len(r) else np.nan,
            'total': days['ret'].sum() if len(days) else np.nan,
            'dd': float((cum - cum.cummax()).min()) if len(days) else np.nan}


def rule_of(rec):
    return (rec.get('review') or {}).get('rule') or DEFAULT_RULE


def case_heads():
    return ''.join(f'<th>{name}（{how}）</th>' for _, name, how in CASES)


def case_cells(r, why=True):
    """A trade's three cases as table cells; without a sale, one cell saying why (or a dash)."""
    if r['result'] not in ('赚', '亏'):
        text = (esc(r['buy_when']) if r['result'] == '未买到' else '') or esc(r['result'])
        return f'<td colspan="{len(CASES)}" class="off">{text if why else "—"}</td>'
    cells = [pc(r[col]) for col, _, _ in CASES]
    if r['stuck']:
        cells[-1] += '<span class="tag">卖出日一字跌停，实际卖不掉</span>'
    return ''.join(f'<td>{c}</td>' for c in cells)


def section_t1(rec, rv):
    nxt = esc(rec.get('next_day', ''))
    if rec['status'] != 'ok':
        bad = next((s for s in rec['steps'] if s['status'] != 'ok'), None)
        why = f'{bad["label"]}：{bad["note"]}' if bad else '流程没有完成'
        return (f'<div class="card bad big"><b>今天没有复盘结果。</b>{esc(why)}<br><span class="small">当天数据没收完时，'
                '16:40 会自动再试一次；也可以手动运行 run_review.ps1。</span></div>')
    if not rv['t1']:
        return (f'<div class="card big"><div class="flat">空仓</div>五位成员的第一选择没有 ≥{rv["rule"]["min_votes"]} 位相同，'
                f'{nxt} 不买。</div>')
    rows = ''
    for x in rv['t1']:
        who = '；'.join(f'{esc(m)}「{esc(mo)}」' for m, mo in zip(x['members'], x['models']))
        sealed = '<span class="tag">收盘封涨停：次日可能一字开盘，挂涨停价排队</span>' if x['sealed'] else ''
        rows += (f'<tr><td><b>{esc(x["label"])}</b></td><td>{x["votes"]} 票</td><td>{who}</td>'
                 f'<td>{esc(x["status"])}{sealed}</td></tr>')
    n = len(rv['t1'])
    money = '资金平分' if n > 1 else '一份资金'
    return (f'<div class="card ok big"><b>{nxt} 开盘买入 {n} 只</b>（集合竞价买入；{money}；再下一个交易日按“{RULE_NAME[SELL]}”）'
            '</div><table><tr><th>股票</th><th>票数</th><th>投第一票的成员（模式）</th><th>复盘时（收盘）</th></tr>'
            f'{rows}</table>')


def section_after(rec, rv, t):
    """Replay pages only: what the T+1 list later did (not visible at the review)."""
    if not rec.get('replay') or not rv or not rv['t1']:
        return ''
    x = t[t['review'] == rec['date']]
    if x.empty:
        return ''
    rows = ''
    for r in x.to_dict('records'):
        entry = price(r['entry']) if r['filled'] else '—'
        rows += (f'<tr><td>{esc(r["name"])}({esc(r["code"])})</td><td>{esc(r["buy_date"])} {esc(r["buy_when"])}</td>'
                 f'<td>{entry}</td><td>{esc(r["sell_date"])}</td>{case_cells(r, why=False)}'
                 f'<td>{esc(r["result"])}</td></tr>')
    return ('<h2>事后结果（复盘当时不可见，回放时附上）</h2><table><tr><th>股票</th><th>买入</th><th>买入价</th><th>卖出日</th>'
            f'{case_heads()}<th>结果（按平均）</th></tr>{rows}</table>')


def section_votes(rv):
    hits = {x['code'] for x in rv['t1']}
    rows, others = '', ''
    for m in rv['members']:
        p1 = m['picks'][0] if m['picks'] else None
        p2 = m['picks'][1] if len(m['picks']) > 1 else None
        will = '本场不参加' if m['out'] else f'买入意愿 {m["exposure"]:.0%}'
        if m['member']:
            if p1:
                cls = ' class="hit"' if p1['code'] in hits else ''
                first = (f'<td{cls}>{esc(p1["label"])}「{esc(p1["model"])}」<br><span class="small">{esc(p1["why"])}</span>'
                         f'<br><span class="small">复盘时：{esc(p1["status"])}</span></td>')
            else:
                first = f'<td class="off">不选：{esc(m["abstain"])}</td>'
            second = f'<td class="off">{esc(p2["label"])}「{esc(p2["model"])}」</td>' if p2 else '<td class="off">—</td>'
            rows += (f'<tr><td><b>{esc(m["name"])}</b><br><span class="small">{esc(m["school"])} · {will}</span></td>'
                     f'<td class="small">{esc(m["stance"])}</td>{first}{second}</tr>')
        else:
            picks = '；'.join(f'{"①②"[i]} {esc(p["label"])}' for i, p in enumerate(m['picks']))
            others += f'<tr><td>{esc(m["name"])}</td><td>{picks or "不选（" + esc(m["abstain"]) + "）"}</td></tr>'
    return (f'<table><tr><th>成员</th><th>判断</th><th>① 第一选择（算）</th><th>② 第二选择（不算）</th></tr>{rows}</table>'
            '<details><summary>另外四位辩手的选票（不参与最优组合）</summary><table><tr><th>辩手</th><th>选票</th></tr>'
            f'{others}</table></details>')


def section_positions(rec, t):
    d = rec['date']
    bought = t[(t['buy_date'] == d) & t['filled']]
    sold = t[(t['sell_date'] == d) & t['result'].isin(['赚', '亏'])]
    items = []
    for r in bought.to_dict('records'):
        items.append(f'<li>今天开盘买入 <b>{esc(r["name"])}({esc(r["code"])})</b>（{esc(r["review"])} 复盘的信号）：'
                     f'{esc(r["buy_when"])}，买入价 {price(r["entry"])}（相对昨收 {pc(r["buy_chg"])}），今天收盘 {pc(r["ret_close"])}。'
                     f'<b>下一个交易日（{esc(rec.get("next_day", ""))}）按“{RULE_NAME[SELL]}”卖出。</b></li>')
    for r in sold.to_dict('records'):
        stuck = '（一字跌停，实际卖不掉）' if r['stuck'] else ''
        three = '，'.join(f'{name}（{how}）{pc(r[col])}' for col, name, how in CASES)
        items.append(f'<li>今天卖出 <b>{esc(r["name"])}({esc(r["code"])})</b>（{esc(r["buy_date"])} 买入）{stuck}：{three}</li>')
    return '<ul>' + ''.join(items) + '</ul>' if items else '<p>今天没有买入，也没有卖出。</p>'


def record_table(t, votes, rule):
    since = rule['selected_through']
    rows = ''
    for label, cut in ((f'选定后（{since} 之后的复盘）', since), ('回测全期', None)):
        for col, name, how in CASES:
            s = summary(t, daily(t, votes, col), col, cut)
            rows += (f'<tr><td>{esc(label)}</td><td>{name}（{how}）</td><td>{s["days"]}</td><td>{s["signal_days"]}</td>'
                     f'<td>{s["trades"]}</td><td>{rate(s["win"])}</td><td>{pc(s["avg"])}</td><td>{pc(s["total"])}</td>'
                     f'<td>{pc(s["dd"])}</td></tr>')
    return ('<table><tr><th>范围</th><th>情况</th><th>复盘场次</th><th>有信号</th><th>成交笔数</th><th>胜率</th><th>每笔平均</th>'
            f'<th>合计（各场相加）</th><th>最大回撤</th></tr>{rows}</table><p class="small">每场复盘一份资金，几只信号平分，'
            '没买到或卖出日停牌的那份算现金；未扣交易费用（来回约 0.1%，另有滑点）。</p>')


def section_record(t, votes, rule):
    done = t[t['result'].isin(['赚', '亏'])].tail(10).iloc[::-1]
    recent = ''.join(
        f'<tr><td>{esc(r["review"])}</td><td>{esc(r["name"])}({esc(r["code"])})</td><td>{r["votes"]}</td>'
        f'<td>{esc(r["members"])}</td><td>{esc(r["buy_date"])}</td><td>{esc(r["sell_date"])}</td>{case_cells(r)}</tr>'
        for r in done.to_dict('records'))
    return (record_table(t, votes, rule) + '<h3>最近 10 笔</h3><table><tr><th>复盘日</th><th>股票</th><th>票数</th>'
            f'<th>成员</th><th>买入日</th><th>卖出日</th>{case_heads()}</tr>{recent}</table>')


def section_check(rec):
    c = rec.get('data_check')
    if not c:
        return ''
    if 'error' in c:
        return f'<div class="card warn">16:40 数据核对没有完成：{esc(c["error"])}</div>'
    picks = '；'.join(f'{esc(code)}：{esc(v)}' for code, v in c['picks'].items()) or '（没有 T+1 股票）'
    cls = 'ok' if all(v == '一致' for v in c['picks'].values()) else 'warn'
    return (f'<div class="card {cls}">16:40 数据核对（{esc(c["checked_at"])}，重新下载当天分钟线对比，D:\\data 不改动）：'
            f'{c["close_changed"]} 只股票的收盘价不同，{c["volume_changed"]} 只成交量相差 >3%，'
            f'{c["bar_close_diff_share"]:.2%} 的分钟线收盘价不同；T+1 清单：{picks}</div>')


def page(rec, t, votes):
    rv = rec.get('review') or {}
    d = rec['date']
    wk = WEEK[pd.Timestamp(d).weekday()]
    data = rec.get('data') or {}
    eng = rec.get('engine') or {}
    rule = rule_of(rec)
    status = []
    if data.get('stocks'):
        status.append(f'数据：D:\\data\\{d}.csv，{data["stocks"]} 只股票，{data["close_bar_share"]:.1%} 有 15:00 收盘那根；'
                      f'文件指纹 {esc(data.get("sha256", "")[:12])}')
    if eng:
        frozen = '冻结副本未改动' if not eng.get('changed') else f'<b>冻结副本有 {len(eng["changed"])} 个文件被改动，策略需要重新验证</b>'
        status.append(f'引擎：{esc(eng.get("version", ""))}（{frozen}，冻结于 {esc(eng.get("frozen_at", ""))}）')
    if rv:
        audit = '通过（15:10 之后的数据换成随机数重开，选票不变）' if rv['audit_passed'] else '<b>未通过！今天的信号作废</b>'
        status.append(f'防偷看审计：{audit}')
        status.append(f'复盘可见数据：{esc(rv["visible"])}')
    if 'cross_check' in rec:
        same = '一致' if rec['cross_check'] else '<b>不一致</b>'
        status.append(f'两套计算（回测记录 / 当场复盘）的 T+1 清单：{same}')
    status.append(f'交易日历：{esc(rec.get("calendar", ""))}；生成于 {esc(rec.get("finished") or rec.get("started", ""))}')
    replay = ''
    if rec.get('replay'):
        replay = (f'<div class="card warn">回放：这一页是 {esc(rec.get("finished", ""))} 重新生成的 {esc(d)} 复盘。'
                  '选票只用当时可见的数据（同样做了防偷看审计），页尾另附事后结果。</div>')
    steps = ''.join(f'<tr><td>{esc(s["label"])}</td><td>{"✔" if s["status"] == "ok" else esc(s["status"])}</td>'
                    f'<td>{s["seconds"]:.0f} 秒</td><td class="small">{esc(s["note"])}</td></tr>' for s in rec.get('steps', []))
    body = [f'<h1>最优组合复盘 · {esc(d)}（星期{wk}）</h1>',
            f'<p class="sub">{esc("、".join(rule["members"]))} 五位，只算各自的第一选择；同一只票 ≥{rule["min_votes"]} 位相同，'
            '下一个交易日开盘买入，再下一个交易日卖出。没有就空仓，不凑数。</p>', replay,
            '<div class="card small">' + '<br>'.join(status) + '</div>',
            f'<h2>T+1（{esc(rec.get("next_day", ""))}）买入清单</h2>', section_t1(rec, rv)]
    if rv:
        if rv['roundtable']:
            rt = ('<table><tr><th>名次</th><th>股票</th><th>票数</th><th>选它的辩手（★=第一选择）</th></tr>'
                  + ''.join(f'<tr><td>{x["rank"]}</td><td>{esc(x["label"])}</td><td>{x["count"]}</td><td>{esc(x["who"])}</td></tr>'
                            for x in rv['roundtable']) + '</table>')
        else:
            rt = f'<p>{esc(rv["roundtable_note"])}</p>'
        body += ['<h2>五位成员的选票</h2>', section_votes(rv), '<h2>持仓与卖出提醒</h2>', section_positions(rec, t),
                 '<h2>盘面简报</h2><ul>' + ''.join(f'<li>{md(x)}</li>' for x in rv['briefing']) + '</ul>',
                 '<h2>圆桌九位的共识票（只作参考，不是买入信号）</h2>', rt, section_check(rec)]
    body += ['<h2>战绩</h2>', section_record(t, votes, rule), section_after(rec, rv, t),
             '<h2>规则与风险</h2><ul>'
             '<li>只用收盘后 15:10 复盘；买入：下一个交易日开盘集合竞价，开盘一字涨停的挂涨停价排队，全天不开板就买不到；'
             f'卖出：再下一个交易日，按“{RULE_NAME[SELL]}”（config.json 的 sell_rule 可改）。</li>'
             f'<li>{CASE_NOTE}卖出日一字跌停卖不掉，三种情况都按跌停价计。</li>'
             '<li>这个组合是从 511 个组合里事后挑出的第一名；纯运气挑出的第一名也常有很高的回测收益。'
             '向前记录还很少，请用小额、固定金额，先试一个月。</li>'
             '<li>回测收益靠少数大涨的交易；亏起来很快（合计曲线最大回落约 32 个百分点）。本页不自动下单。</li></ul>',
             '<h2>运行步骤</h2><table><tr><th>步骤</th><th>结果</th><th>耗时</th><th>说明</th></tr>' + steps + '</table>',
             '<p class="small"><a href="index.html">全部复盘</a> · 台账 D:\\review\\ledger\\signals.csv · 日志 D:\\review\\logs</p>']
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,'
            f' initial-scale=1"><title>最优组合复盘 {esc(d)}</title><style>{CSS}</style></head><body><main>'
            + ''.join(body) + '</main></body></html>')


def index_rows(rec, t):
    """A review's rows on the index: one per T+1 stock (its sale day and three cases), one for a day without any."""
    rv = rec.get('review') or {}
    rest = 1 + len(CASES)
    if rec['status'] != 'ok':
        return [f'<td colspan="{rest + 1}"><span class="small">没有结果（{esc(rec["status"])}）</span></td>']
    if not rv.get('t1'):
        return [f'<td>空仓</td><td colspan="{rest}"></td>']
    rows = []
    for x in rv['t1']:
        r = t[(t['review'] == rec['date']) & (t['code'] == x['code'])]
        if r.empty:
            rows.append(f'<td>{esc(x["label"])}</td><td colspan="{rest}">—</td>')
            continue
        r = r.iloc[0]
        sold = esc(r['sell_date']) if r['filled'] else ''
        rows.append(f'<td>{esc(x["label"])}</td><td>{sold or "—"}</td>{case_cells(r)}')
    return rows


def index_page(t, votes, rule):
    rows = ''
    for p in sorted(REPORTS.glob('20??-??-??.json'), reverse=True):
        rec = json.loads(p.read_text(encoding='utf-8'))
        lines = index_rows(rec, t)
        span = f' rowspan="{len(lines)}"' if len(lines) > 1 else ''
        tag = '<span class="tag">回放</span>' if rec.get('replay') else ''
        rows += (f'<tr><td{span}><a href="{rec["date"]}.html">{rec["date"]}</a>{tag}</td>'
                 f'<td{span}>{esc(rec.get("next_day", ""))}</td>{lines[0]}</tr>'
                 + ''.join(f'<tr>{x}</tr>' for x in lines[1:]))
    return (f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>最优组合复盘 · 全部</title><style>{CSS}</style>'
            f'</head><body><main><h1>最优组合复盘 · 全部</h1><p class="sub">{CASE_NOTE}收益相对买入价（除权已调整），'
            '未扣交易费用。<a href="latest.html">最新一期</a></p><h2>战绩</h2>' + record_table(t, votes, rule)
            + f'<h2>每场复盘</h2><table><tr><th>复盘日</th><th>买入日（T+1）</th><th>信号</th><th>卖出日</th>{case_heads()}</tr>'
            f'{rows}</table></main></body></html>')


def ledger(t):
    cols = {'review': '复盘日', 'phase': '阶段', 'code': '代码', 'name': '股票', 'votes': '票数', 'members': '投第一票的成员',
            'buy_date': '买入日', 'buy_when': '买入方式', 'entry': '买入价', 'sell_date': '卖出日',
            't1_best': '最好收益%（最高价卖出）', 't1_worst': '最差收益%（最低价卖出）', 't1_mid': '平均收益%（均价卖出）',
            'ret_open': '开盘卖出收益%', 'ret_close_t1': '收盘卖出收益%', 'result': '结果'}
    t[list(cols)].rename(columns=cols).to_csv(LEDGER / 'signals.csv', index=False, encoding='utf-8-sig')


def render(rec):
    t = trades_with_rules()
    votes = pd.read_csv(OUT / 'votes.csv')
    path = REPORTS / f'{rec["date"]}.html'
    path.write_text(page(rec, t, votes), encoding='utf-8')
    dates = sorted(p.stem for p in REPORTS.glob('20??-??-??.json'))
    if not dates or rec['date'] >= dates[-1]:
        shutil.copyfile(path, REPORTS / 'latest.html')
    (REPORTS / 'index.html').write_text(index_page(t, votes, rule_of(rec)), encoding='utf-8')
    ledger(t)
    return path


def render_all():
    """Re-render every page from its reports/<date>.json, e.g. after a format change; no review is held again."""
    t = trades_with_rules()
    votes = pd.read_csv(OUT / 'votes.csv')
    recs = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(REPORTS.glob('20??-??-??.json'))]
    for rec in recs:
        (REPORTS / f'{rec["date"]}.html').write_text(page(rec, t, votes), encoding='utf-8')
    if recs:
        shutil.copyfile(REPORTS / f'{recs[-1]["date"]}.html', REPORTS / 'latest.html')
        (REPORTS / 'index.html').write_text(index_page(t, votes, rule_of(recs[-1])), encoding='utf-8')
    ledger(t)
    return len(recs)


if __name__ == '__main__':
    print(f'{render_all()} pages, latest.html, index.html and {LEDGER / "signals.csv"} rendered')
