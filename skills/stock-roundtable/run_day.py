"""Hold the roundtable on one trading day at one time of day, using only data visible at that time.

    python -X utf8 run_day.py 2026-09-24                        # the 10:00 meeting and the 15:10 review
    python -X utf8 run_day.py 2026-09-24 --time 10:30           # one meeting at 10:30
    python -X utf8 run_day.py 2026-09-24 --time 14:00 --blind   # leave out what happened afterwards
    python -X utf8 run_day.py 2026-09-24 --time 10:30 --audit   # also re-run with every later value randomised

Each meeting is written to output/days/<date>_<HHMM>.md: the moderator's briefing, every debater's own picks (or
why it picks nothing), the moderator's tally, the consensus picks ranked by how many debaters chose them (空仓 when
there are none), and (unless --blind) what happened after the decision: the buy day and the A-share T+1 sale on the
trading day after the buy. Times from 09:31 to 14:59 are intraday meetings (buy at that minute; a consensus pick
already sealed at the limit can only queue at the limit price); 15:00 or later is the after-close review, held at
15:10 (buy at the next day's open). On the day itself the review waits for 15:10 Beijing time and for a file that has
the closing bar.

The emotion-stage rules compare the day with every earlier full-market day at the same time of day, so the
first meeting at a new time walks through the whole history once (a few minutes) and caches it in
cache/market_<HHMM>.json; a full backtest run fills the 10:00 and 15:10 caches. The dialog window
(chat_app.py) holds the same meetings interactively.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine import config, meeting  # noqa: E402


def log(msg):
    print(msg, flush=True)


def hold(data, date, time_str, out_dir, blind=False, audit=False):
    try:
        m = meeting.convene(data, date, time_str, blind=blind, audit=audit, log=log)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    path = meeting.write_record(m, out_dir)
    db, ctx = m.db, m.ctx
    names = {d.key: d.name for d in m.debaters}
    log('')
    log(f'{date} {m.spec["label"]} 圆桌 → {path}')
    log(f'  情绪阶段 {ctx.stage}；92科比周期 {ctx.kobe_stage}；主线 {"、".join(ctx.mainline) or "无"}；'
        f'仓位意见 {db.position:.0%}（{db.flat_votes}/{db.voters} 位不选）')
    for ch in db.picks:
        who = '、'.join(f'{names[k]}{"★" if order == 1 else ""}' for k, _, _, _, order in ch.choosers)
        seal = f'｜{ctx.cutoff} 已封涨停，只能排队' if ch.sealed else ''
        log(f'  共识 {ch.rank}. {ch.name}({ch.code}) {ch.count} 票｜{who}{seal}')
    if not db.picks:
        log(f'  {db.note}')
    for d, noms, abstain in db.opening:
        pick = '；'.join(f'{db.choices[idx].name}（{model}）' for idx, model, _, _ in noms) if noms else f'不选（{abstain}）'
        log(f'  {d.name}：{pick}')
    if audit:
        log(f'  防偷看审计：{"通过（截止时点之后的数据全部换成随机数，结论不变）" if m.audit_passed else "未通过！结论随未来数据变化"}')
    return path


def main():
    ap = argparse.ArgumentParser(description='在某个交易日的某个时点召开圆桌')
    ap.add_argument('date', help='交易日，例如 2026-09-24（D:/data 里要有这一天的分钟数据）')
    ap.add_argument('--time', action='append',
                    help=f'开会时点 HH:MM，可重复；09:31–14:59 为盘中会议，15:00 以后为收盘后复盘（记作 {config.CLOSE_REVIEW}）；'
                         f'默认 10:00 和 {config.CLOSE_REVIEW}')
    ap.add_argument('--blind', action='store_true', help='不写事后结果')
    ap.add_argument('--audit', action='store_true', help='再做一次防偷看审计')
    ap.add_argument('--out', default=str(config.OUTPUT_DIR / 'days'), help='记录写到哪个文件夹')
    args = ap.parse_args()
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = config.ROUNDTABLE_ROOT / out_dir
    times = args.time or ['10:00', config.CLOSE_REVIEW]
    try:
        specs = [config.meeting(t) for t in times]  # reject a bad time before loading anything
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    early = config.too_early(args.date)
    if early and any(s['mode'] == 'eod' for s in specs):
        if args.time:
            raise SystemExit(f'{early}：收盘后复盘 {config.CLOSE_REVIEW} 以后再开；盘中会议用 --time HH:MM（09:31–14:59）')
        log(f'{early}：这次只开盘中会议，收盘后复盘 {config.CLOSE_REVIEW} 以后再开')
        times = [t for t, s in zip(times, specs) if s['mode'] != 'eod']
    data = meeting.Data()
    if args.date not in data.days:
        raise SystemExit(f'D:/data 里没有 {args.date}.csv（现有 {data.days[0]} – {data.days[-1]}）')
    for t in times:
        hold(data, args.date, t, out_dir, blind=args.blind, audit=args.audit)


if __name__ == '__main__':
    main()
