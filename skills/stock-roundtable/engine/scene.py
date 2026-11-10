"""A roundtable meeting told as a scene: the moderator runs the agenda and each debater speaks in role.

What is decided (each debater's read and choices, the count, the consensus picks) comes from the meeting's record,
i.e. the same rule engine as the backtest; this module only turns it into speeches in meeting order. Since v2.0 the
debaters do not vote on or argue about each other's picks: each one gives its own choices and the moderator counts.
The debaters are style models of the skills, speaking as roles at this meeting, not the real traders.
"""
from . import config
from .debaters import label, now, yi

CALL = {'chaogu-yangjia': '养家', '92-kobe-a-share': '科比', 'chen-xiaoqun-a-share': '小群', 'zhao-laoge': '老哥',
        'zhang-mengzhu': '盟主', 'fang-xinxia': '方新侠', 'zuoshou-xinyi': '新一',
        'qiao-bangzhu': '乔帮主', 'ruihexian': '瑞鹤仙'}
LEAD = {
    'chaogu-yangjia': '我从情绪周期说起。',
    '92-kobe-a-share': '我先看周期，再看龙头、补涨、切换三种机会。',
    'chen-xiaoqun-a-share': '我还是先找主流，再找核心。',
    'zhao-laoge': '我只看龙头和接力的节点。',
    'zhang-mengzhu': '我看大资金能不能进出、趋势还在不在。',
    'fang-xinxia': '我看容量票这一两天的节点。',
    'zuoshou-xinyi': '我先看热门强势股有没有低吸位置，其次才是板上确认。',
    'qiao-bangzhu': '我只做强势股的恐慌低吸。',
    'ruihexian': '我先看大盘定仓位，再看逻辑和资金能不能共振。',
}
ORDINAL = '一二三四五六七八九十'
TALLY_STAGE = '第三项 · 唱票'
VERDICT_STAGE = '主持人宣布结果'


def _trim(text, parts=2):
    return '；'.join([s for s in str(text).split('；') if s][:parts])


def _core(stance):
    return '，'.join([s for s in stance.split('；') if s and not s.startswith('依据')][:2])


def _nth(i):
    return f'第{ORDINAL[i]}' if i < len(ORDINAL) else f'第{i + 1}'


def line(who, text, kind=None, stage=None, focus=None, blocks=None):
    return {'who': who, 'text': text, 'kind': kind, 'stage': stage, 'focus': focus, 'blocks': blocks}


def briefing(ctx):
    m = ctx.market
    s = [f'截至 {ctx.cutoff}，全市场 {m["n_trading"]} 只股票，上涨 {m["up"]} 只、下跌 {m["down"]} 只，'
         f'上涨的占 {m["breadth"]:.0%}，等权平均 {m["ew_ret"]:+.2f}%。']
    av = m['amount_vs5']
    s.append(f'成交 {yi(m["amount_total"])}' + (f'，是前五天同一时间的 {av:.2f} 倍' if av else '') + '。')
    s.append(f'涨停这边，触板 {m["lu_touch"]} 家、现在封住 {m["lu_now"]} 家，炸板率 {m["zb_rate"]:.0%}；跌停 {m["ld_now"]} 家。')
    if m['prem_raw'] is not None:
        s.append(f'昨天的涨停股今天平均 {m["prem_raw"]:+.1f}%'
                 + (f'，昨天的连板股平均 {m["prem_lb_raw"]:+.1f}%' if m['prem_lb_raw'] is not None else '') + '。')
    leaders = '、'.join(n for _, n in m['leaders'])
    if m['height'] >= 2 and leaders:
        s.append(f'最高 {m["height"]} 板是{leaders}，今天 {m["leader_ret"]:+.1f}%。')
    s.append(f'按养家用的情绪分段，今天算“{ctx.stage}”；按科比的四阶段，是“{ctx.kobe_stage}”。')
    s.append(f'主线行业是{"、".join(ctx.mainline)}。' if ctx.mainline else '今天看不出明确的主线。')
    if ctx.clusters:
        c = ctx.clusters[0]
        s.append(f'盘中联动最强的一组有 {c["size"]} 只、{c["lu"]} 家触板，以{"/".join(c["industries"][:2])}为主。')
    return ''.join(s)


def build(m):
    """The meeting as a list of lines: who speaks, what is said, with which stance, under which agenda item."""
    ctx, db = m.ctx, m.db
    views = db.views
    lines = []

    def row(idx):
        return next(ctx.stocks.loc[[idx]].itertuples())

    def lab(idx):
        return label(row(idx))

    def mod(text, **kw):
        lines.append(line('moderator', text, **kw))

    def say(key, text, kind=None, **kw):
        lines.append(line(key, text, kind=kind, **kw))

    def call(key, stage):
        lines.append(line('call', f'主持人请{CALL[key]}发言', stage=stage))

    def divider(title):
        lines.append(line('divider', title, stage=title))

    # ---- opening
    stage = '开会'
    divider(f'{m.date} {m.spec["label"]} 圆桌')
    mod(f'各位好，现在开会。今天是 {m.date}，{m.spec["label"]}。只能用 {ctx.cutoff} 之前的分时和 {ctx.history_last_date} '
        f'及以前的日线。今天的规矩：每位只按自己的框架选票，最多 {config.NOMINATIONS_PER_DEBATER} 只，也可以不选；不互相投票，'
        f'我只统计票数——{config.CONSENSUS_MIN} 位及以上选中的就是共识票，按票数从多到少排，没有共识票就空仓。'
        f'成交规则：{m.spec["entry"]}。议程三项：盘面通报、各位选票、唱票。我先通报盘面。', stage=stage)

    stage = '第一项 · 盘面通报'
    divider(stage)
    mod(briefing(ctx), stage=stage)

    # ---- each debater's own read and choices
    stage = '第二项 · 各位选票'
    divider(stage)
    mod('下面请各位依次说今天的判断和选票，只说自己的，不用评价别人。从养家开始。', stage=stage)
    for d, noms, abstain in db.opening:
        v = views[d.key]
        call(d.key, stage)
        if v.mode_out:
            say(d.key, LEAD[d.key] + f'{_core(v.stance)}。这一场不是我的打法，我不选。', 'neutral', stage=stage)
            continue
        if v.flat_reason:
            say(d.key, LEAD[d.key] + f'今天我空仓，不选：{v.flat_reason}。', 'neutral', stage=stage)
            continue
        core = _core(v.stance)
        s = [LEAD[d.key], f'{core}。']
        if '意愿' not in core:
            s.append(f'今天我的买入意愿是{v.exposure:.0%}。')
        for i, (idx, model, why, _) in enumerate(noms):
            s.append(f'{"我选" if i == 0 else "再选一只，"}**{lab(idx)}**，按我的「{model}」：{_trim(why)}。')
        if not noms:
            s.append(f'今天我不选：{abstain}。')
        say(d.key, ''.join(s), 'nominate' if noms else 'neutral', stage=stage)

    # ---- the count
    stage = TALLY_STAGE
    divider(stage)
    names = {d.key: CALL[d.key] for d in m.debaters}
    ranked = sorted(db.choices.values(), key=lambda ch: (-ch.count, -ch.firsts, ch.rank or 999, ch.code))
    if not ranked:
        mod('今天没有人选票。', stage=stage)
    else:
        multi = [ch for ch in ranked if ch.count >= 2]
        single = [ch for ch in ranked if ch.count < 2]
        s = [f'选票都在这里了，一共 {len(ranked)} 只。我来唱票：']
        for ch in multi:
            s.append(f'**{lab(ch.idx)}**，{"、".join(names[k] for k in ch.keys)}，{ch.count} 票；')
        if single:
            s.append(f'其余 {len(single)} 只各 1 票：' + '、'.join(f'{ch.name}（{names[ch.keys[0]]}）' for ch in single) + '。')
        rows = [[lab(ch.idx), ch.count, '、'.join(names[k] + ('★' if order == 1 else '') for k, _, _, _, order in ch.choosers),
                 now(row(ch.idx))] for ch in ranked]
        lines.append(line('moderator', ''.join(s), stage=stage,
                          blocks=[{'type': 'text', 'text': ''.join(s)},
                                  {'type': 'table', 'headers': ['股票', '票数', '选它的辩手（★第一选择）', '开会时'], 'rows': rows}]))

    # ---- verdict
    stage = VERDICT_STAGE
    divider(stage)
    s = []
    if db.picks:
        s.append(f'按规矩，{config.CONSENSUS_MIN} 位及以上选中的是共识票，今天一共 {len(db.picks)} 只，按票数排：')
        for i, ch in enumerate(db.picks):
            note = f'；它 {ctx.cutoff} 已经封住涨停，只能按涨停价排队，开板才买得到' if ch.sealed else ''
            s.append(f'{_nth(i)}，**{lab(ch.idx)}**，{ch.count} 票（{"、".join(names[k] for k in ch.keys)}）{note}。')
    else:
        s.append(f'{db.note}。')
    s.append(f'仓位意见：参会的 {db.voters} 位平均买入意愿 {db.position:.0%}，{db.flat_votes} 位今天不选。散会。')
    mod(''.join(s), stage=stage)
    return lines
