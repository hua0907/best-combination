"""Step 5: hold one day's 15:10 review with the frozen engine and extract what the page shows.

The meeting is held blind (nothing after the decision is read) and with the engine's anti-peeking audit (every value
after 15:10 replaced by random numbers; the decision must come out identical). The optimal-combination rule itself
(members, first choices only, at least 2 votes) is imported from the frozen optimal_combo.py, so the page and the
backtest use one definition.
"""
import sys

from common import ENGINE_DIR

sys.path.insert(0, str(ENGINE_DIR))
from engine import config as ecfg, meeting, report  # noqa: E402
from engine.debaters import now as status_at_cutoff  # noqa: E402
import optimal_combo as oc  # noqa: E402


def review(day):
    m = meeting.convene(meeting.Data(), day.isoformat(), ecfg.CLOSE_REVIEW, blind=True, audit=True,
                        log=lambda *_: None)
    db, ctx = m.db, m.ctx
    names = {d.key: d.name for d in db.debaters}
    s = ctx.stocks
    members = []
    for d, noms, abstain in db.opening:
        v = db.views[d.key]
        picks = []
        for order, (idx, model, why, _score) in enumerate(noms, 1):
            row = next(s.loc[[idx]].itertuples())
            picks.append({'order': order, 'code': str(s.at[idx, 'code']), 'name': str(s.at[idx, 'name']),
                          'label': report._lab(ctx, idx), 'model': model, 'why': why,
                          'status': status_at_cutoff(row), 'sealed': bool(s.at[idx, 'lu_now'])})
        members.append({'name': d.name, 'school': d.school, 'member': d.name in oc.MEMBERS, 'out': bool(v.mode_out),
                        'exposure': float(v.exposure), 'stance': v.stance, 'picks': picks, 'abstain': abstain})
    firsts = {x['name']: x['picks'][0] for x in members if x['member'] and x['picks']}
    by_code = {}
    for name in oc.MEMBERS:
        if name in firsts:
            by_code.setdefault(firsts[name]['code'], []).append(name)
    t1 = []
    for code, who in sorted(by_code.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(who) >= oc.MIN_VOTES:
            p = firsts[who[0]]
            t1.append({'code': code, 'name': p['name'], 'label': p['label'], 'votes': len(who), 'members': who,
                       'models': [firsts[n]['model'] for n in who], 'status': p['status'], 'sealed': p['sealed']})
    roundtable = [{'rank': ch.rank, 'label': report._lab(ctx, ch.idx), 'count': ch.count,
                   'who': '、'.join(names[k] + ('★' if order == 1 else '') for k, _, _, _, order in ch.choosers)}
                  for ch in db.picks]
    return {'audit_passed': bool(m.audit_passed), 'cutoff': ctx.cutoff,
            'visible': (f'当日 09:31–{ctx.minute_last_bar} 的 1 分钟线（收盘价已定）、截至 {ctx.history_last_date} 的日线、'
                        '开盘前已知的昨收与涨跌停价'),
            'briefing': report.briefing_lines(ctx), 'members': members, 't1': t1, 'roundtable': roundtable,
            'roundtable_note': db.note, 'engine_version': ecfg.VERSION,
            'rule': {'members': list(oc.MEMBERS), 'min_votes': oc.MIN_VOTES, 'selected_through': oc.SELECTED_THROUGH}}
