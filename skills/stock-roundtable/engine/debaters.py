"""Nine rule-based debaters, one per skill in D:/skills/stock. After the T+1 study 小鳄鱼 (xiao-eyu) left the roundtable
in v1.3 and 北京炒家 (beijing-chaogu) and 佛山无影脚 (foshan-wuyingjiao) in v1.4; their skill folders are not used here.

Each debater is a frozen, explicit translation of its SKILL.md into rules over the Context (data
visible at the cutoff). Thresholds were written from the skill text before any backtest result
existed and are not tuned on outcomes. A debater models a published style; it is not the person,
and its wording is not a quote unless the skill marks it as one.

Interface used by the roundtable:
  view(ctx)                 -> View: today's market read in this framework and willingness to buy
  candidates(ctx, view)     -> DataFrame indexed like ctx.stocks: score, model, why (best model per stock)
  objection(ctx, row, view) -> Review or None: this framework's reservation about a stock; shown in the dialog
                               window when someone asks, never part of the verdict (v2.0: the host only counts picks)
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import context, minutes


@dataclass
class View:
    exposure: float          # willingness to open new positions today, 0..1
    stance: str              # the market read in this framework, one line
    max_nominations: int = 2
    flat_reason: str = ''    # non-empty: 空仓 today (chooses nothing)
    mode_out: bool = False   # this decision time is outside the framework: chooses nothing, not counted in 仓位意见


@dataclass
class Review:
    kind: str                # support / soft / hard / neutral
    tag: str
    text: str


EMPTY = pd.DataFrame({'score': pd.Series(dtype=float), 'model': pd.Series(dtype=object),
                      'why': pd.Series(dtype=object)})


# ---------------------------------------------------------------- wording helpers

def ok(x):
    try:
        return x is not None and bool(np.isfinite(x))
    except TypeError:
        return False


def num(x, fmt='{:+.1f}'):
    return fmt.format(x) if ok(x) else '—'


def yi(a):
    return f'{a / 1e8:.2f}亿' if ok(a) else '—'


def tm(i):
    return minutes.TIMES[int(i)] if ok(i) and i >= 0 else '—'


def label(r):
    band = '' if r.pct == 10 else f'，{r.pct:g}cm'
    return f'{r.name or r.code}({r.code}{band})'


def now(r):
    return f'{"涨停" if r.lu_now else "现"}{num(r.ret)}%'


def vw(r):
    side = '上方' if r.vs_vwap >= 0 else '下方'
    return f'在均价线{side}{abs(r.vs_vwap):.1f}%（{r.above_vwap:.0%}时间在线上）'


def amt(r):
    return f'成交{yi(r.amount)}（第{r.amount_rank}名）'


def ratio(r):
    return f'量比{num(r.amt_ratio, "{:.1f}")}'


def seal(r):
    if not r.lu_touched:
        return '未触板'
    s = f'{tm(r.lu_first)}触板'
    if r.lu_opens:
        s += f'、开板{int(r.lu_opens)}次'
    return s + ('、现封住' if r.lu_now else '、现未封')


def streak(r):
    b = int(r.boards_y)
    if b >= 2:
        return f'昨日{b}连板'
    if b == 1:
        return '昨日首板'
    if r.lu_touch_y:
        return '昨日炸板'
    return f'昨日{num(r.ret_y)}%'


def ind(r):
    return f'{r.industry}（行业第{r.ind_rank}，触板{r.ind_lu}家）'


def pos(r):
    p = r.pos20 * 100 if ok(r.pos20) else None
    return f'20日涨{num(r.ret20, "{:+.0f}")}%、位于20日区间{num(p, "{:.0f}")}%处'


def role(r):
    tags = []
    if r.is_top_leader:
        tags.append('全市场最高板')
    if r.ind_leader:
        tags.append('行业龙头')
    if r.cluster_leader:
        tags.append(f'联动簇#{int(r.cluster)}领头')
    if r.cap_core:
        tags.append('行业容量核心')
    return '、'.join(tags) or '非龙头'


def build(e, parts):
    """parts: (mask, score, model, why). Keeps each stock's best-scoring model."""
    frames = []
    for mask, score, model, why in parts:
        mask = mask.fillna(False).astype(bool)
        rows = e[mask]
        # F3 闸门：位置不合格的票不进投票环节。池子没这一列时行为不变（旧引擎）。
        if 'qc_pass' in e.columns:
            rows = rows[rows['qc_pass'].fillna(False).astype(bool)]
        if rows.empty:
            continue
        sc = score[mask] if isinstance(score, pd.Series) else pd.Series(float(score), index=rows.index)
        frames.append(pd.DataFrame({'score': sc.astype(float), 'model': model,
                                    'why': [why(r) for r in rows.itertuples()]}, index=rows.index))
    if not frames:
        return EMPTY.copy()
    out = pd.concat(frames)
    out['amount'] = e.loc[out.index, 'amount'].to_numpy()
    out['code'] = e.loc[out.index, 'code'].to_numpy()
    out = out.sort_values(['score', 'amount', 'code'], ascending=[False, False, True], kind='mergesort')
    return out[~out.index.duplicated(keep='first')]


class Debater:
    key = name = school = ''
    models = ()       # the buy models this framework chooses with (for "not in my framework" answers)
    aliases = ()      # other names the dialog window accepts
    short = ''        # avatar label in the dialog window

    def view(self, ctx):
        raise NotImplementedError

    def candidates(self, ctx, view):
        raise NotImplementedError

    def objection(self, ctx, r, view):
        return None


# ---------------------------------------------------------------- 1 炒股养家

class ChaoguYangjia(Debater):
    key, name, school = 'chaogu-yangjia', '炒股养家', '情绪拐点'
    models = ('模式A 冰点后的主动弱转强', '模式B 龙头第一次强分歧', '模式C 新周期首个高辨识度核心', '主线核心')
    aliases, short = ('养家',), '养家'
    EXPOSURE = {'冰点': .3, '修复': .6, '发酵': 1.0, '加速': .8, '高潮': .4, '分歧': .5, '退潮': .2, '混合': .5}
    NOTE = {'冰点': '冰点不等于立即买入，只看新题材核心', '修复': '修复是试错阶段', '发酵': '发酵确认，可提高参与度',
            '加速': '加速段不是新的低风险买点，只等龙头分歧', '高潮': '高潮更接近兑现窗口，不新增',
            '分歧': '分歧后可能修复也可能退潮，用承接验证', '退潮': '退潮优先保护资本', '混合': '指标互相矛盾，按混合处理'}

    def view(self, ctx):
        st = ctx.stage
        big = '上方' if ctx.market['idx_above_ma5'] else '下方'
        return View(self.EXPOSURE[st], f'情绪阶段{st}（{self.NOTE[st]}）；依据：{"，".join(ctx.stage_evidence)}；'
                                       f'大盘代理在5日线{big}', max_nominations=0 if st == '高潮' else 2)

    def candidates(self, ctx, v):
        e, st, prev = ctx.pool, ctx.stage, ctx.prev_stage
        parts = []
        if prev in ('冰点', '退潮') and st not in ('冰点', '退潮', '高潮'):
            mask = (e.theme & (e.weak_y | (e.ret_s_y <= 0)) & (e.open_s >= -2) & (e.ret_s >= 3) & (e.vs_vwap > 0)
                    & (e.above_vwap >= .6) & (e.amt_ratio >= 1.5) & (e.ind_mean_s > 0))
            parts.append((mask, 8 + e.ret_s.clip(upper=10) / 2 + 2 * e.ind_leader, '模式A 冰点后的主动弱转强',
                          lambda r: f'前一日{prev}、今日{st}；{streak(r)}，今日开{num(r.open_ret)}%后承接走强，'
                                    f'{now(r)}，{vw(r)}，{ratio(r)}；{ind(r)}同步走强'))
        if st != '高潮':
            mask = ((e.boards_y >= 3) & (e.is_top_leader | e.ind_leader) & (e.ind_lu >= 2)
                    & ((e.low_s <= -3) | (e.lu_opens >= 1)) & (e.lu_reseal | ((e.vs_vwap > 0) & (e.rebound_s >= 4)))
                    & ~(e.opens_y >= 2))
            parts.append((mask, 10 + e.boards_y.clip(upper=8), '模式B 龙头第一次强分歧',
                          lambda r: f'{streak(r)}的{role(r)}，今日最低{num(r.low_ret)}%后回流（{seal(r)}），{vw(r)}；'
                                    f'{ind(r)}，主线未结束'))
        if st in ('冰点', '修复', '发酵', '分歧', '混合'):
            mask = (e.lu_now & e.ind_leader & e.theme & (e.ind_lu >= 3) & (e.boards_y <= 1)
                    & ((e.cluster_new >= .5) | (e.ind_yday_lu <= 1)))
            parts.append((mask, 8 + e.ind_lu.clip(upper=8) / 2 + e.lu_first.between(0, 14), '模式C 新周期首个高辨识度核心',
                          lambda r: f'{ind(r)}昨日仅{r.ind_yday_lu}家涨停、今日成批启动，它是{role(r)}，{seal(r)}；'
                                    f'新题材+主动性+板块效应'))
        if st in ('修复', '发酵'):
            mask = e.theme & (e.is_top_leader | e.ind_leader) & (e.lu_now | (e.ret_s >= 7)) & (e.above_vwap >= .7)
            parts.append((mask, 7 + e.board_now.clip(upper=6) / 2 + 2 * e.is_top_leader, '主线核心',
                          lambda r: f'{role(r)}，{streak(r)}，{now(r)}（{seal(r)}），{vw(r)}；{ind(r)}；'
                                    f'优先级：总龙头>主线核心>容量中军>前排补涨>后排跟风'))
        return build(e, parts)

    def objection(self, ctx, r, v):
        st = ctx.stage
        if st == '退潮' and r.boards_y >= 3:
            return Review('hard', 'retreat_high', f'退潮阶段接{streak(r)}的高位股，框架要求优先保护资本')
        if st == '高潮' and r.back_row:
            return Review('hard', 'climax_back_row', f'高潮阶段追后排（{seal(r)}）：MUST NOT 因害怕踏空无条件追后排')
        if not r.theme and not r.ind_leader and r.amount_rank > 1500:
            return Review('hard', 'junk', f'不在主线、不是板块龙头且{amt(r)}，属杂毛（MUST NOT 因价格低选杂毛股）')
        if r.back_row:
            return Review('soft', 'back_row', f'在{r.industry}里是后排跟风（{seal(r)}），优先级排在末位')
        if st == '加速' and ok(r.ret5) and r.ret5 >= 30:
            return Review('soft', 'accelerating', f'5日已涨{r.ret5:.0f}%，加速段离失效位置越来越远')
        return None


# ---------------------------------------------------------------- 2 92科比

class Kobe92(Debater):
    key, name, school = '92-kobe-a-share', '92科比', '周期·龙头/补涨/切换'
    models = ('龙头', '补涨', '切换', '弱转强', '低位新方向试错')
    aliases, short = ('科比',), '科比'
    EXPOSURE = {'主升': 1.0, '低位震荡': .5, '高位震荡': .3, '主跌': .1}
    NOTE = {'低位震荡': '题材试错期，看低位新方向', '主升': '主升阶段敢于在分歧时介入',
            '高位震荡': '高位震荡轻仓应对', '主跌': '主跌阶段只看新方向的修复时点'}

    def view(self, ctx):
        st, m = ctx.kobe_stage, ctx.market
        lead = '、'.join(n for _, n in m['leaders']) or '无'
        return View(self.EXPOSURE[st], f'周期{st}（{self.NOTE[st]}）；最高{m["height"]}板{lead}，'
                                       f'今日{num(m["leader_ret"])}%；5日前最高{m["height_5d_ago"]}板')

    def candidates(self, ctx, v):
        e, st, m = ctx.pool, ctx.kobe_stage, ctx.market
        parts = []
        if st in ('主升', '低位震荡', '高位震荡'):
            lead = e.is_top_leader | (e.ind_leader & e.theme & (e.boards_y >= 2))
            mask = lead & (e.lu_now | ((e.ret_s >= 5) & (e.vs_vwap > 0)))
            if st == '主升':
                mask = mask | (e.is_top_leader & (e.low_s <= -3) & (e.vs_vwap > 0))
            parts.append((mask, 10 + e.boards_y.clip(upper=8), '龙头',
                          lambda r: f'{role(r)}，{streak(r)}，{now(r)}（{seal(r)}），{vw(r)}；{ind(r)}'))
            mask = (e.weak_y & (e.open_s <= 2) & (e.ret_s >= 5) & (e.vs_vwap > 0) & (e.amt_ratio >= 1.2)
                    & (e.theme | (e.ind_lu >= 2)))
            parts.append((mask, 8 + e.ret_s.clip(upper=10) / 2, '弱转强',
                          lambda r: f'预期：{streak(r)}，次日竞价本应偏弱；实际：开{num(r.open_ret)}%后买盘很凶，'
                                    f'{now(r)}，{vw(r)}，{ratio(r)}'))
        if st in ('主升', '低位震荡'):
            mask = ((e.ind_max_boards >= 3) & (e.boards_y == 0) & (e.pos20 <= .5) & (e.ret20 < 15)
                    & (e.lu_now | (e.ret_s >= 7)) & (e.ind_lu >= 2) & ~e.is_top_leader)
            parts.append((mask, 7 + e.ind_lu.clip(upper=6) / 2, '补涨',
                          lambda r: f'{r.industry}的龙头已打开{r.ind_max_boards}板高度，它是低位（{pos(r)}）首次启动，'
                                    f'{now(r)}（{seal(r)}），板块触板{r.ind_lu}家'))
        old_cycle = m['height'] >= 3
        old_weak = old_cycle and not m['leader_sealed'] and (m['leader_ret_s'] is None or m['leader_ret_s'] < 3)
        new_dir = ((e.ind_leader | e.cluster_leader) & e.lu_now & (e.ind_lu >= 3) & (e.ind_yday_lu <= 1)
                   & (e.boards_y <= 1) & (e.industry != (m['leader_industry'] or '')))
        if old_weak or (old_cycle and st in ('主跌', '高位震荡')):
            parts.append((new_dir, 9 + e.ind_lu.clip(upper=8) / 2, '切换',
                          lambda r: f'老龙头今日{num(m["leader_ret"])}%{"未封板" if not m["leader_sealed"] else ""}，'
                                    f'新方向{ind(r)}昨日仅{r.ind_yday_lu}家涨停；它是{role(r)}，{seal(r)}；题材出现的时机正好'))
        elif st == '低位震荡':
            parts.append((new_dir & (e.pos20 <= .6), 7 + e.ind_lu.clip(upper=8) / 2, '低位新方向试错（非三模式）',
                          lambda r: f'没有老周期可切换；{ind(r)}昨日仅{r.ind_yday_lu}家涨停，它是{role(r)}，{seal(r)}，'
                                    f'{pos(r)}：最低位的新方向试错'))
        return build(e, parts)

    def objection(self, ctx, r, v):
        if 3 <= r.boards_y <= 5 and not r.is_top_leader and not r.ind_leader:
            return Review('hard', 'mid_board', f'{streak(r)}的中位股，不是总龙头也不是板块龙头（直播笔记：“中位股的风险较大”）')
        if r.weak_y and r.open_s >= 7:
            return Review('soft', 'auction_gap', f'{streak(r)}，今日竞价高开{num(r.open_ret)}%：“弱转强不喜欢竞价高开，'
                                                 f'竞价高开属于做盘的迹象太明显了”')
        if ctx.kobe_stage == '主跌' and r.boards_y >= 1:
            return Review('soft', 'main_down_relay', f'主跌阶段接{streak(r)}的股票，周期不支持接力')
        return None


# ---------------------------------------------------------------- 3 陈小群

class ChenXiaoqun(Debater):
    key, name, school = 'chen-xiaoqun-a-share', '陈小群', '主流·总核心'
    models = ('市场总核心', '主线龙头', '容量核心')
    aliases, short = ('小群',), '小群'

    def nature(self, ctx):
        m = ctx.market
        lead = m['leader_ret_s']
        if lead is not None and lead <= -3 and (m['prem_lb'] or 0) <= -2 and m['zb_rate'] >= .35:
            return '退潮分歧'
        if (m['leader_sealed'] or (lead is not None and lead >= 0)) and (m['prem'] or 0) >= 0:
            return '良性'
        return '无法确认'

    def view(self, ctx):
        if not ctx.mainline:
            return View(0.0, '看不到主流（没有行业触板≥2家或强势≥4家）', 0, flat_reason='主流优先：今天没有可确认的主流，不做个股')
        nat = self.nature(ctx)
        m = ctx.market
        return View({'退潮分歧': .2, '良性': .8, '无法确认': .5}[nat],
                    f'主流{"、".join(ctx.mainline)}；分歧性质：{nat}（总龙今日{num(m["leader_ret"])}%，'
                    f'昨日连板股今日均{num(m["prem_lb_raw"])}%，炸板率{m["zb_rate"]:.0%}）')

    def candidates(self, ctx, v):
        e = ctx.pool
        top_ind = ctx.mainline[0]
        indep = (e.ret_s - e.ind_mean_s) >= 3

        def tail(r):
            gap = r.ret_s - r.ind_mean_s
            return f'独立性：比行业均值强{gap:.1f}（10cm口径）' + ('，板块分歧时能保持独立强度' if gap >= 3 else '')

        parts = [
            ((e.industry == top_ind) & e.ind_leader & (e.ind_lu >= 3) & (e.lu_now | ((e.ret_s >= 5) & (e.vs_vwap > 0))),
             10 + e.board_now.clip(upper=6) / 2 + indep, '市场总核心',
             lambda r: f'第一主流{ind(r)}里的{role(r)}，{streak(r)}，{now(r)}（{seal(r)}），带动{r.ind_lu}家触板；{tail(r)}'),
            (e.theme & e.ind_leader & e.lu_now & (e.ind_lu >= 2), 8 + e.board_now.clip(upper=6) / 2 + indep, '主线龙头',
             lambda r: f'主流{ind(r)}的{role(r)}，{streak(r)}，{seal(r)}；{tail(r)}'),
            (e.theme & e.cap_core & (e.ret_s >= 5) & (e.amount_rank <= 100), 7 + indep, '容量核心',
             lambda r: f'主流{ind(r)}里成交最大的强势股，{amt(r)}，{now(r)}；{tail(r)}'),
        ]
        out = build(e, parts)
        if self.nature(ctx) == '退潮分歧' and not out.empty:
            out = out[e.loc[out.index, 'boards_y'].to_numpy() < 3]
        return out

    def objection(self, ctx, r, v):
        if not r.theme:
            if not (r.ind_leader or r.cluster_leader or r.cap_core):
                return Review('hard', 'non_mainline', f'{r.industry}不是当前主流，它今天也不是板块的领涨核心（主流优先，禁止倒过来）')
            return Review('soft', 'non_mainline', f'{r.industry}不是当前主流（主流：{"、".join(ctx.mainline) or "无"}）')
        if self.nature(ctx) == '退潮分歧' and r.boards_y >= 3:
            return Review('hard', 'retreat_high', f'退潮分歧中接{streak(r)}的高位股')
        if not (r.ind_leader or r.is_top_leader or r.cap_core or r.cluster_leader) and (r.ret_s - r.ind_mean_s) < 2:
            return Review('soft', 'follower', f'跟风：只比行业均值强{(r.ret_s - r.ind_mean_s):.1f}（10cm口径），主要靠核心带动')
        return None


# ---------------------------------------------------------------- 4 赵老哥

class ZhaoLaoge(Debater):
    key, name, school = 'zhao-laoge', '赵老哥', '龙头接力'
    models = ('龙头接力', '二板竞争', '分歧转一致（回封）', '龙头二波', '弱转强')
    aliases, short = ('老哥',), '老哥'

    def hard_to_trade(self, ctx):
        m = ctx.market
        return m['zb_rate'] >= .4 or (m['prem'] is not None and m['prem'] < -1)

    def view(self, ctx):
        m = ctx.market
        hard = self.hard_to_trade(ctx)
        facts = f'炸板率{m["zb_rate"]:.0%}，昨日涨停今日均{num(m["prem_raw"])}%'
        return View(.2 if hard else .8, f'{"板难做，减少操作（只提1只）" if hard else "可以接力"}：{facts}',
                    max_nominations=1 if hard else 2)

    def candidates(self, ctx, v):
        e = ctx.pool
        base = (e.amount_rank <= 800) & ((e.ind_lu >= 2) | (e.cluster_size >= 3))
        parts = [
            (base & e.is_top_leader & e.lu_now, 10 + e.boards_y.clip(upper=8), '龙头接力',
             lambda r: f'全市场最高板，{streak(r)}，今日{seal(r)}，{amt(r)}；{ind(r)}'),
            (base & (e.boards_y == 1) & e.lu_now & e.second_board_first, 8 + e.lu_first.between(0, 29) * 1.0, '二板竞争',
             lambda r: f'{r.industry}昨日首板里今天最先封二板（{seal(r)}），{amt(r)}，板块触板{r.ind_lu}家'),
            (base & e.lu_reseal & ((e.boards_y >= 1) | e.ind_leader), 7 + e.boards_y.clip(upper=5) / 2, '分歧转一致（回封）',
             lambda r: f'{streak(r)}，今日{seal(r)}：砸开后封回；{ind(r)}'),
            (base & (e.max_run20 >= 3) & (e.boards_y == 0) & (e.days_since_lu >= 2) & (e.ret_s >= 7) & (e.vs_vwap > 0)
             & (e.amt_ratio >= 1.2) & (e.ind_mean_s >= 0), 9, '龙头二波',
             lambda r: f'20日内最多{r.max_run20}连板的老龙头，{r.days_since_lu}天前最后一次涨停，今天重新走强：'
                       f'{now(r)}，{vw(r)}，{ratio(r)}'),
            (base & e.weak_y & (e.open_s < 5) & (e.ret_s >= 5) & (e.vs_vwap > 0) & (e.amt_ratio >= 1.2) & (e.ind_mean_s > 0),
             7, '弱转强',
             lambda r: f'昨日弱（{streak(r)}），今日开{num(r.open_ret)}%后走强到{num(r.ret)}%，{vw(r)}，{ratio(r)}，板块同步'),
        ]
        return build(e, parts)

    def objection(self, ctx, r, v):
        if ctx.stage == '高潮' and r.back_row:
            return Review('hard', 'climax_back_row', f'高潮日接最后排（{seal(r)}）：MUST NOT')
        if r.lu_now and r.ind_lu <= 1 and not r.in_cluster:
            return Review('soft', 'isolated', f'孤立涨停：{r.industry}只有它触板，也不在任何联动簇里')
        if r.one_word and ok(r.turnover) and r.turnover < 1:
            return Review('soft', 'one_word', f'缩量一字（换手{r.turnover:.2f}%），承接无法判断')
        if r.back_row:
            return Review('soft', 'back_row', f'后排跟风（{seal(r)}），不是板块最强')
        if r.weak_y and r.open_s >= 5 and not r.lu_now and (r.ret_s - r.open_s) < 1:
            return Review('soft', 'gap_only', f'{streak(r)}，今日开{num(r.open_ret)}%后没有继续走强：单纯高开不算弱转强')
        return None


# ---------------------------------------------------------------- 5 章盟主

class ZhangMengzhu(Debater):
    key, name, school = 'zhang-mengzhu', '章盟主', '大资金趋势'
    models = ('模式A 主线容量核心突破', '模式B 趋势核心首次大分歧（缩量回踩）')
    aliases, short = ('章建平', '盟主'), '盟主'

    def view(self, ctx):
        m = ctx.market
        av = m['amount_vs5']
        good = (av is None or av >= .95) and m['idx_above_ma10']
        ratio_txt = '' if av is None else f'（前5日同时点的{av:.2f}倍）'
        return View(.7 if good else .4, f'两市成交{yi(m["amount_total"])}{ratio_txt}，大盘代理'
                                        f'{"在10日线上方" if m["idx_above_ma10"] else "跌破10日线"} → 大资金{"可以" if good else "谨慎"}参与')

    def candidates(self, ctx, v):
        e = ctx.pool
        big = e.amount >= ctx.amt(3e8, 1e9)
        trend = (e.prev_close_adj > e.ma5) & (e.ma5 > e.ma10) & (e.ma10 > e.ma20)
        pending = '（突破待收盘确认）' if ctx.mode == 'intraday' else ''
        parts = [
            (big & (e.amount_rank <= 100) & trend & e.ret20.between(5, 60) & (e.vs_vwap > 0) & (e.price > e.high20)
             & e.amt_ratio.between(1.2, 3.5) & (e.ind_mean_s > .5),
             6 + np.log10(e.amount / 1e8).clip(lower=0), '模式A 主线容量核心突破' + pending,
             lambda r: f'{amt(r)}；5/10/20日线多头排列，20日涨{num(r.ret20, "{:+.0f}")}%（趋势建立到主升）；'
                       f'现价{r.price:.2f}突破20日高点{r.high20:.2f}，{ratio(r)}（放量不爆量）；{r.industry}同步走强'),
            ((e.amount_rank <= 150) & (e.ma20 > e.ma20_prev) & (e.price_vs_ma10.abs() <= 2) & (e.price > e.ma20)
             & (e.amt_ratio <= .9) & e.ret_s.between(-5, 2),
             5, '模式B 趋势核心首次大分歧（缩量回踩）',
             lambda r: f'{amt(r)}；20日线上行，回踩10日线（偏离{num(r.price_vs_ma10)}%），缩量（{ratio(r)}），'
                       f'仍在20日线上方；回流放量只能作为次日确认条件'),
        ]
        return build(e, parts)

    def objection(self, ctx, r, v):
        if r.amount < ctx.amt(5e7, 1e8):
            return Review('hard', 'illiquid', f'{amt(r)}，极低成交额小票，大资金无法进出')
        if ok(r.ret5) and r.ret5 >= 40:
            return Review('soft', 'climax', f'5日涨{r.ret5:.0f}%：连续加速后的一致高潮是风险最高区间')
        if r.one_word:
            return Review('soft', 'one_word', '连续一字无法正常换手')
        return None


# ---------------------------------------------------------------- 6 方新侠

class FangXinxia(Debater):
    key, name, school = 'fang-xinxia', '方新侠', '容量趋势节点'
    models = ('主升加速', '趋势突破', '强分歧承接')
    aliases, short = ('新侠',), '新侠'

    def view(self, ctx):
        up = ctx.market['idx_above_ma10']
        return View(.7 if up else .4, f'大盘代理{"在10日线上方" if up else "跌破10日线"}，指数与流动性'
                                      f'{"支持" if up else "不支持"}容量股加速')

    def candidates(self, ctx, v):
        e = ctx.pool
        core = (e.amount_rank <= 50) & (e.theme | (e.ind_rank <= 5)) & (e.ma5 > e.ma20)
        pending = '（待收盘确认）' if ctx.mode == 'intraday' else ''

        def late(r):
            return f'；注意5日已涨{r.ret5:.0f}%，越后期风险越高' if ok(r.ret5) and r.ret5 >= 25 else ''

        parts = [
            (core & (e.amt_ratio >= 1.5) & (e.ret_s >= 5) & (e.vs_vwap > 0) & (e.prev_close_adj > e.ma10),
             7 - 2 * (e.ret5 >= 25), '主升加速',
             lambda r: f'{amt(r)}，{ratio(r)}，{now(r)}，{vw(r)}；{ind(r)}{late(r)}'),
            (core & (e.price > e.high20) & (e.amt_ratio >= 1.3) & (e.ind_mean_s > 0), 6, '趋势突破' + pending,
             lambda r: f'现价{r.price:.2f}突破20日高点{r.high20:.2f}，{amt(r)}，{ratio(r)}，板块共振（{ind(r)}）'),
            (core & (e.low_s <= -5) & (e.vs_vwap > 0) & (e.amt_ratio >= 1.3) & (e.prev_close_adj > e.ma20), 6, '强分歧承接',
             lambda r: f'盘中最低{num(r.low_ret)}%后收复均价线（{vw(r)}），{amt(r)}，{ratio(r)}；板块回流要次日确认'),
        ]
        return build(e, parts)

    def objection(self, ctx, r, v):
        floor = ctx.amt(1e8, 3e8)
        core = r.ind_leader or r.is_top_leader or r.cluster_leader or r.cap_core
        if r.amount < floor and not core:
            return Review('hard', 'illiquid_back', f'{amt(r)}且不是板块核心：追逐低流动性后排（MUST NOT）')
        if r.ret_s > 0 and r.above_vwap < .4:
            return Review('soft', 'one_off', f'只有{r.above_vwap:.0%}时间在均价线上，像一次性拉升，不能当持续资金')
        return None


# ---------------------------------------------------------------- 7 作手新一

class ZuoshouXinyi(Debater):
    key, name, school = 'zuoshou-xinyi', '作手新一', '强势股低吸·接力'
    models = ('模式A 热门强势股低吸', '模式B 核心突破接力（板上确认）', '模式C 分歧回封')
    aliases, short = ('新一',), '新一'

    def level(self, ctx):
        if not ctx.mainline:
            return '无主线'
        e = ctx.pool
        if not bool((e.theme & (e.ind_leader | e.is_top_leader) & (e.lu_now | (e.ret_s >= 7))).any()):
            return '观察'
        return '试错' if ctx.stage in ('退潮', '冰点') else '标准'

    def view(self, ctx):
        lv = self.level(ctx)
        reason = {'标准': '主线、核心地位确认，情绪未退潮', '试错': '主线和核心确认，但情绪在退潮/冰点',
                  '观察': '只确认了主线，核心不明', '无主线': '没有主线'}[lv]
        exp = {'标准': .8, '试错': .5, '观察': .2, '无主线': 0.0}[lv]
        weak = not ctx.market['idx_above_ma10']
        if weak:
            exp *= .7
        acting = lv in ('标准', '试错')
        return View(round(exp, 2), f'证据档位：{lv}（{reason}）{"；弱市下调上限" if weak else ""}',
                    max_nominations=2 if acting else 0, flat_reason='' if acting else f'{reason}，只观察不出手')

    def candidates(self, ctx, v):
        e = ctx.pool
        hot = ((e.ret10_rank <= 100) | (e.lu10 >= 2)) & (e.amount_rank <= 300) & e.theme
        parts = [
            (hot & e.ret_s.between(-6, 2) & ((e.vs_vwap.abs() <= 1.5) | (e.rebound_s >= 3)) & (e.prev_close_adj >= e.ma10)
             & (e.ind_mean_s > -2), 9 + e.lu10.clip(upper=4) / 2, '模式A 热门强势股低吸',
             lambda r: f'人气核心（10日涨{num(r.ret10, "{:+.0f}")}%、10日内{r.lu10}个涨停，{amt(r)}）分歧回落到{num(r.ret)}%，'
                       f'{vw(r)}；结构未破（昨收在10日线上方），{ind(r)}没有集体负反馈'),
            ((e.is_top_leader | (e.ind_leader & (e.ind_rank == 1))) & e.lu_now & (e.amount_rank <= 300) & e.theme,
             8, '模式B 核心突破接力（板上确认）',
             lambda r: f'{role(r)}，{streak(r)}，{seal(r)}，{amt(r)}；行业内高度最高、封板最早'),
            (e.lu_reseal & e.theme & (e.ind_leader | (e.boards_y >= 1)) & (e.ind_lu >= 2) & (e.lu_opens <= 2),
             7, '模式C 分歧回封',
             lambda r: f'{streak(r)}，{seal(r)}：炸板后有承接并回封，{ind(r)}板块支持'),
        ]
        return build(e, parts)

    def objection(self, ctx, r, v):
        if r.ret_s < 0 and not r.theme and not (ok(r.ret20) and r.ret20 >= 5):
            return Review('hard', 'weak_dip', f'{r.industry}不是主线，{pos(r)}：MUST NOT 把非主线、弱势股的下跌当低吸')
        if r.one_word:
            return Review('soft', 'one_word', '一字板：MUST NOT 假设一定能按涨停价成交')
        lead = ctx.market['leader_ret_s']
        if lead is not None and lead <= -7 and r.boards_y >= 1:
            return Review('soft', 'leader_crash', f'总龙头今日{num(ctx.market["leader_ret"])}%（准天地板），接力降级')
        return None


# ---------------------------------------------------------------- 8 乔帮主

class QiaoBangzhu(Debater):
    key, name, school = 'qiao-bangzhu', '乔帮主', '强势股低吸'
    models = ('龙头第一次强分歧', '恐慌低吸（情绪错杀）', '核心趋势回踩')
    aliases, short = ('帮主', '玉米'), '帮主'

    def view(self, ctx):
        strong_mkt = ctx.stage in ('发酵', '加速', '高潮')
        return View(.4 if strong_mkt else .6, f'情绪{ctx.stage}：' +
                    ('强市里“主升是低吸不来的”，低吸只作辅助' if strong_mkt else '弱/中性市，低吸更合适'))

    def candidates(self, ctx, v):
        e = ctx.pool
        strong = (((e.ret20 >= 20) | (e.lu20 >= 2)) & (e.amount_rank <= 800) & ((e.max_run20 >= 2) | (e.lu20 >= 2))
                  & ((e.ind_yday_lu >= 1) | (e.ind_lu >= 1)))
        panic = (strong & ((e.low_s <= -6) | (e.drop_from_open_s <= -6)) & (e.rebound_s >= 2) & (e.ret_s <= 0)
                 & (e.low_minute <= ctx.k - 4) & (e.ind_mean_s > -3) & ~e.ld_now)
        first_div = panic & (e.boards_y >= 3) & ~(e.opens_y >= 2)
        parts = [
            (first_div, 9 + e.boards_y.clip(upper=6) / 2, '龙头第一次强分歧',
             lambda r: f'{streak(r)}的强势核心，今日最低{num(r.low_ret)}%（{tm(r.low_minute)}）后止跌，现{num(r.ret)}%；'
                       f'{ind(r)}未退潮；次日除非涨停按计划了结'),
            (panic & ~first_div, 8, '恐慌低吸（情绪错杀）',
             lambda r: f'强势股（20日涨{num(r.ret20, "{:+.0f}")}%、20日内{r.lu20}个涨停）急杀到{num(r.low_ret)}%'
                       f'（{tm(r.low_minute)}），抛压后止跌回到{num(r.ret)}%；{ind(r)}未崩；次日除非涨停按计划了结'),
            (strong & ((e.price_vs_ma5.abs() <= 2) | (e.price_vs_ma10.abs() <= 2)) & (e.amt_ratio <= .9) & (e.ret_s <= 0)
             & (e.ind_mean_s > -1) & (e.prev_close_adj > e.ma20), 6, '核心趋势回踩',
             lambda r: f'强势股缩量（{ratio(r)}）回踩{"5" if abs(r.price_vs_ma5) <= 2 else "10"}日线，仍在20日线上方，'
                       f'板块未进入调整'),
        ]
        return build(e, parts)

    def objection(self, ctx, r, v):
        if r.ld_now:
            return Review('hard', 'limit_down_buy', '正封在跌停：仅因为跌停就买（MUST NOT），须同时有强势身份、恐慌盘和可见承接')
        if ok(r.ret20) and r.ret20 < 0 and r.ret_s < -5 and r.lu20 == 0:
            return Review('hard', 'falling_knife', f'{pos(r)}的弱势股继续下跌到{num(r.ret)}%：越跌越买（MUST NOT）')
        if ctx.stage == '退潮' and r.boards_y >= 3 and r.ret_s < 0:
            return Review('hard', 'retreat_leader', f'退潮中{streak(r)}的龙头今日{num(r.ret)}%：对退潮龙头补仓（MUST NOT）')
        return None


# ---------------------------------------------------------------- 9 瑞鹤仙

class Ruihexian(Debater):
    key, name, school = 'ruihexian', '瑞鹤仙', '逻辑·大盘定仓位'
    models = ('模式C 逻辑与情绪共振（仅资金验证）',)
    aliases, short = ('鹤仙',), '鹤仙'

    def cap(self, ctx):
        a5, a10 = ctx.market['idx_above_ma5'], ctx.market['idx_above_ma10']
        return 1.0 if (a5 and a10) else (.5 if (a5 or a10) else .2)

    def view(self, ctx):
        cap = self.cap(ctx)
        where = {1.0: '站上5、10日线', .5: '只站上5、10日线中的一条', .2: '跌破5、10日线'}[cap]
        return View(cap * .5, f'大盘代理{where} → 仓位上限{cap:.0%}；回测里没有公告/政策/业绩数据，逻辑链前半段无法验证，'
                              f'逻辑强度一律记“低”，只做资金验证，意愿减半为{cap * .5:.0%}')

    def candidates(self, ctx, v):
        e = ctx.pool
        mask = (e.theme & (e.ind_leader | e.cluster_leader | e.cap_core) & (e.amt_ratio >= 1.5) & (e.above_vwap >= .8)
                & (e.prev_close_adj >= e.ma20) & (e.ind_yday_lu >= 2))
        return build(e, [(mask, 6 + e.ind_yday_lu.clip(upper=6) / 3, '模式C 逻辑与情绪共振（仅资金验证）',
                          lambda r: f'假设（开盘前可写）：{r.industry}昨日{r.ind_yday_lu}家涨停，资金继续验证则延续；'
                                    f'验证：{role(r)}，{now(r)}，{vw(r)}，{ratio(r)}；反证：板块涨停家数回落或它跌破均价线')])

    def objection(self, ctx, r, v):
        if self.cap(ctx) <= .2 and (r.boards_y >= 2 or (ok(r.ret5) and r.ret5 >= 20)):
            return Review('soft', 'index_weak', '大盘代理跌破5、10日线：“‘重个股、轻指数’是绝对错误的”，不重仓高位')
        if ok(r.ret5) and r.ret5 >= 30 and ok(r.amt_ratio) and r.amt_ratio >= 3:
            return Review('soft', 'crowded', f'5日涨{r.ret5:.0f}%且{ratio(r)}：交易拥挤度高')
        return None


def all_debaters():
    return [ChaoguYangjia(), Kobe92(), ChenXiaoqun(), ZhaoLaoge(), ZhangMengzhu(), FangXinxia(), ZuoshouXinyi(),
            QiaoBangzhu(), Ruihexian()]
