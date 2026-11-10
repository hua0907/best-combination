"""Consensus roundtable for one day and one mode (v2.0).

Each debater states its market read and chooses up to NOMINATIONS_PER_DEBATER stocks that match its own framework, or
chooses none. Nobody votes on, supports or objects to anyone else's choice. The moderator only counts: a stock chosen
by at least CONSENSUS_MIN debaters is a consensus pick. Picks are ranked by how many debaters chose them, then by how
many took them as their first choice, then by turnover; there is no cap and nothing is filled in. A meeting without a
consensus pick stays in cash (空仓). In an intraday meeting a pick that sits at the limit-up price at the cutoff can only
be bought by queuing at the limit; it stays in the ranking, marked as sealed.
"""
from dataclasses import dataclass, field

from . import config
from .debaters import EMPTY


@dataclass
class Choice:
    """One stock that at least one debater chose."""
    idx: int
    code: str
    name: str
    choosers: list = field(default_factory=list)   # [(key, model, why, score, order)]; order 1 = that debater's first choice
    sealed: bool = False                            # intraday: at the limit-up price at the cutoff, so it can only queue
    rank: int = 0                                   # 1, 2, ... among the consensus picks; 0 when it is not one

    @property
    def count(self):
        return len(self.choosers)

    @property
    def firsts(self):
        return sum(1 for *_, order in self.choosers if order == 1)

    @property
    def keys(self):
        return [k for k, *_ in self.choosers]


@dataclass
class Debate:
    ctx: object
    debaters: list
    views: dict
    opening: list          # [(debater, [(idx, model, why, score)], why it chose nothing)]
    choices: dict          # idx -> Choice, every stock somebody chose
    picks: list            # consensus picks, best first
    position: float        # mean willingness to buy across the debaters taking part in this mode (空仓 = 0)
    flat_votes: int        # debaters that chose nothing
    voters: int            # debaters taking part in this mode
    cands: dict = None     # debater key -> its candidate frame (stocks that fit its framework, best model each)
    note: str = ''         # why the meeting has no consensus pick; '' when it has one


def run(ctx, debaters):
    stocks = ctx.stocks
    views, cands, opening, choices = {}, {}, [], {}
    for d in debaters:
        v = d.view(ctx)
        views[d.key] = v
        active = not v.flat_reason and v.max_nominations > 0 and v.exposure > 0
        cands[d.key] = c = d.candidates(ctx, v) if active else EMPTY
        if v.flat_reason:
            opening.append((d, [], v.flat_reason))
            continue
        if v.max_nominations == 0 or v.exposure <= 0:
            opening.append((d, [], '本阶段不新增'))
            continue
        if c.empty:
            opening.append((d, [], '没有股票满足本框架的条件'))
            continue
        noms = []
        for order, (idx, row) in enumerate(c.head(v.max_nominations).iterrows(), 1):
            noms.append((idx, row['model'], row['why'], float(row['score'])))
            ch = choices.setdefault(idx, Choice(int(idx), stocks.at[idx, 'code'], stocks.at[idx, 'name']))
            ch.choosers.append((d.key, row['model'], row['why'], float(row['score']), order))
        opening.append((d, noms, ''))

    intraday = ctx.mode == 'intraday'
    for ch in choices.values():
        ch.sealed = intraday and bool(stocks.at[ch.idx, 'lu_now'])
    picks = sorted((ch for ch in choices.values() if ch.count >= config.CONSENSUS_MIN),
                   key=lambda ch: (-ch.count, -ch.firsts, -float(stocks.at[ch.idx, 'amount']), ch.code))
    for rank, ch in enumerate(picks, 1):
        ch.rank = rank
    note = ''
    if not picks:
        note = (f'本场没有共识票：被选中的 {len(choices)} 只票都不到 {config.CONSENSUS_MIN} 位辩手，空仓' if choices
                else '本场没有辩手选票，空仓')

    taking_part = [d for d in debaters if not views[d.key].mode_out]
    exposures = [0.0 if (views[d.key].flat_reason or views[d.key].max_nominations == 0) else views[d.key].exposure
                 for d in taking_part]
    return Debate(ctx=ctx, debaters=debaters, views=views, opening=opening, choices=choices, picks=picks,
                  position=round(sum(exposures) / max(len(taking_part), 1), 3),
                  flat_votes=sum(1 for d, noms, _ in opening if not noms and not views[d.key].mode_out),
                  voters=len(taking_part), cands=cands, note=note)
