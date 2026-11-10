"""Dialog-window logic: typed instructions become messages from the moderator and the nine debaters.

A meeting plays out as a scene (engine/scene.py): the moderator runs the agenda and each debater speaks in
role, one at a time, at a reading pace the user can pause, speed up or skip. Every debater message comes from
the rule engine (the same rules as the backtest) and only uses data visible at the meeting's cutoff. What happened after the decision is shown only when asked for ("结果").
The debaters are style models of the skills, not the people.
"""
import re
import threading
import time
import traceback
from datetime import datetime

import pandas as pd

from . import config, scene, store
from . import meeting as mt
from .debaters import all_debaters, amt, ind, label, now, role, seal, streak, vw
from .report import briefing_lines, buy_way

COLORS = {
    'chaogu-yangjia': '#e8590c', '92-kobe-a-share': '#7048e8', 'chen-xiaoqun-a-share': '#d6336c',
    'zhao-laoge': '#c92a2a', 'zhang-mengzhu': '#1971c2', 'fang-xinxia': '#0c8599', 'zuoshou-xinyi': '#2f9e44',
    'qiao-bangzhu': '#364fc7',
    'ruihexian': '#495057',
}
# Skills that left the roundtable: a word that names one -> (name, when and why). A question about them gets this.
GONE_V13 = 'v1.3 起：T+1 回测里各项都最差'
GONE_V14 = 'v1.4 起：T+1 回测里前后两半都跑输市场'
RETIRED = {'小鳄鱼': ('小鳄鱼', GONE_V13), '鳄鱼': ('小鳄鱼', GONE_V13), '北京炒家': ('北京炒家', GONE_V14),
           '佛山无影脚': ('佛山无影脚', GONE_V14), '无影脚': ('佛山无影脚', GONE_V14)}
KIND = {'nominate': '选票', 'support': '符合框架', 'soft': '有顾虑', 'hard': '不认可', 'neutral': '中立', 'view': '表态'}
EXAMPLES = ['开会 2026-09-24 10:30', '9月24日 下午两点 开会', '开会 最新 收盘', '大家怎么看 鼎阳科技',
            '乔帮主怎么看 688112', '炒股养家怎么看', '对比 陈小群 乔帮主', '选票', '唱票', '共识票', '结果', '战绩 乔帮主', '帮助']
HELP_ROWS = [
    ['开会', '开会 2026-09-24 10:30｜9月24日 下午两点 开会｜开会 最新 收盘｜下一个交易日 开会',
     '开一场正式的圆桌会：主持人通报盘面 → 每人只按自己的框架选票（不互相投票）→ 主持人唱票 → 宣布共识票，辩手一位一位发言。'
     f'只写时间就沿用当前日期；加“盲测”不算事后结果。“收盘”“盘后”或 15:00 以后的时间都是收盘后 {config.CLOSE_REVIEW} 复盘'
     f'（当天要到 {config.CLOSE_REVIEW} 以后、收盘数据到齐才开）'],
    ['会议进行中', '暂停｜继续｜快一点｜慢一点｜快进｜跳到唱票｜结束会议',
     '控制会议节奏；会中随时可以提问（举手），主持人会暂停、先回答问题再继续'],
    ['完整材料', '完整材料', '把本场的全部材料（每人的选票和理由、唱票表、共识票）一次性展开'],
    ['盘面 / 主线', '盘面｜主线', '主持人简报：涨跌、涨停结构、情绪阶段、主线行业、联动簇'],
    ['问某位辩手', '炒股养家怎么看｜问 乔帮主', '这位辩手本场的判断、买入意愿、选票和理由，以及其中哪些成了共识票'],
    ['问某位辩手怎么看某只股', '乔帮主怎么看 688112｜陈小群 鼎阳科技', '这位辩手按自己的框架对这只股票的看法（只是看法，不影响共识）'],
    ['大家讨论某只股', '大家怎么看 鼎阳科技｜讨论 600641', '每位辩手按自己的框架表态（选了 / 符合框架 / 有顾虑 / 不认可 / 中立），只是看法'],
    ['选票 / 唱票 / 共识票', '选票｜唱票｜共识票｜排板', '每人的选票；每只票几票、谁选的；共识票排名或空仓；共识票里开会时已封涨停的'],
    ['对比', '对比 陈小群 乔帮主', '两位辩手的判断、选票，以及对彼此选票的看法'],
    ['战绩', '战绩｜战绩 乔帮主', '这一天之前已经知道结果的历史选票表现（回测记录：买入日收盘、T+1 均价卖出）'],
    ['结果', '结果', '决策之后发生了什么：买入时涨幅、买入日收盘、T+1 最高/均价/最低价卖出（开会时看不到；盲测不显示）'],
    ['名单 / 导出 / 清空', '名单｜导出｜清空', '辩手名单；把对话存成 Markdown；清空窗口'],
]


def _pc(x):
    return f'{x:+.2f}%' if x is not None and pd.notna(x) else '—'


def text(t):
    return {'type': 'text', 'text': t}


def items(lst):
    return {'type': 'list', 'items': list(lst)}


def table(headers, rows):
    return {'type': 'table', 'headers': list(headers), 'rows': [list(r) for r in rows]}


# ---------------------------------------------------------------- parsing

CN_DIGIT = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}


def _cn_number(s):
    if s.isdigit():
        return int(s)
    if s == '十':
        return 10
    if s.startswith('十'):
        return 10 + CN_DIGIT.get(s[1:], 0)
    if '十' in s:
        a, _, b = s.partition('十')
        return CN_DIGIT.get(a, 0) * 10 + (CN_DIGIT.get(b, 0) if b else 0)
    return CN_DIGIT.get(s)


def parse_time(t):
    """'10:30', '10点半', '下午两点', '14点05分', '收盘' -> 'HH:MM' (None when no time is given)."""
    m = re.search(r'(?<!\d)(\d{1,2})[:：](\d{2})(?!\d)', t)
    if m:
        return f'{int(m[1]):02d}:{m[2]}'
    m = re.search(r'([0-9零一二两三四五六七八九十]{1,3})\s*点\s*(半|([0-9零一二两三四五六七八九十]{1,3})\s*分?)?', t)
    if m:
        hour = _cn_number(m[1])
        if hour is None:
            return None
        minute = 30 if m[2] == '半' else (_cn_number(m[3]) or 0) if m[3] else 0
        if hour <= 3 or (('下午' in t or '午后' in t) and hour < 12):
            hour += 12
        return f'{hour:02d}:{minute:02d}'
    for words, value in ((('收盘', '盘后'), config.CLOSE_REVIEW), (('尾盘',), '14:30'), (('午盘', '中午'), '11:30')):
        if any(w in t for w in words):
            return value
    return None


def parse_date(t, days, current=None):
    """(day, explicit). '2026-09-24', '20260924', '9月24日', '09-24' are explicit; '最新', '下一个交易日'
    are relative and only count together with a word like 开会. An unknown day comes back as '?...'."""
    m = re.search(r'(20\d{2})\s*[-/.年]\s*(\d{1,2})\s*[-/.月]\s*(\d{1,2})', t)
    if m:
        return f'{m[1]}-{int(m[2]):02d}-{int(m[3]):02d}', True
    m = re.search(r'(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)', t)
    if m:
        return f'{m[1]}-{m[2]}-{m[3]}', True
    m = re.search(r'(?<![\d:：-])(\d{1,2})\s*[-/月]\s*(\d{1,2})(?![\d:：])', t)
    if m:
        tail = f'-{int(m[1]):02d}-{int(m[2]):02d}'
        hits = [d for d in days if d.endswith(tail)]
        return (hits[-1] if hits else '?' + tail[1:]), True
    base = current or (days[-1] if days else None)
    if any(w in t for w in ('下一个交易日', '下一交易日', '后一天', '下一天', '次日', '明天')) and base in days:
        i = days.index(base)
        return (days[i + 1] if i + 1 < len(days) else '?次日'), False
    if any(w in t for w in ('上一个交易日', '上一交易日', '前一天', '前一交易日', '昨天', '上一天')) and base in days:
        i = days.index(base)
        return (days[i - 1] if i > 0 else '?前日'), False
    if any(w in t for w in ('最新', '最近', '最后一天', '今天', '今日')):
        return (days[-1] if days else None), False
    return None, False


# ---------------------------------------------------------------- meeting playback

SPEEDS = {'慢': 1.0, '正常': 1.8, '快': 3.5}
TALLY_STAGE = scene.TALLY_STAGE
VERDICT_STAGE = scene.VERDICT_STAGE


class Player:
    """Releases a meeting's lines one at a time, as if spoken: pause, speed, fast-forward, jump."""

    def __init__(self, session, lines, speed='正常'):
        self.session = session
        self.lines = lines
        self.i = 0
        self.speed = speed
        self.paused = False
        self.stopped = False
        self.rush = False
        self.jump_to = None
        self.next_who = None
        self.done = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    @property
    def active(self):
        return not self.done

    def _delay(self, ln):
        if ln['who'] in ('divider', 'call'):
            return 0.8 / SPEEDS[self.speed]
        return min(10.0, 1.2 + len(ln['text']) / 20.0) / SPEEDS[self.speed]

    def _run(self):
        try:
            while self.i < len(self.lines) and not self.stopped:
                if self.jump_to:
                    target = next((j for j in range(self.i, len(self.lines))
                                   if self.lines[j]['stage'] == self.jump_to), None)
                    self.jump_to = None
                    if target is not None and target > self.i:
                        self.session.system(f'（跳过 {target - self.i} 段发言，完整内容见会议记录或「完整材料」）')
                        self.i = target
                ln = self.lines[self.i]
                self.next_who = ln['who']
                waited, need = 0.0, self._delay(ln)
                while not self.stopped and not self.rush and not self.jump_to and (self.paused or waited < need):
                    time.sleep(0.1)
                    if not self.paused:
                        waited += 0.1
                if self.stopped:
                    break
                if self.jump_to:
                    continue
                self.session.post_line(ln)
                self.i += 1
        finally:
            self.next_who = None
            self.done = True


# ---------------------------------------------------------------- session

class ChatSession:
    def __init__(self, out_dir=None):
        self.lock = threading.Lock()
        self.messages = []
        self.next_id = 1
        self.reset_seq = 0
        self.busy = False
        self.data = None
        self.meeting = None
        self.player = None
        self.speed = '正常'
        self.seq = 0
        self.debaters = all_debaters()
        self.by_key = {d.key: d for d in self.debaters}
        self.out_dir = out_dir or config.OUTPUT_DIR
        self.days = store.data_dates()
        self.full_days = mt.full_market_days(self.days)
        self._welcome()

    # ---- plumbing
    def post(self, role, blocks, speaker=None, key=None, kind=None, group=None):
        with self.lock:
            msg = {'id': self.next_id, 'time': datetime.now().strftime('%H:%M:%S'), 'role': role, 'blocks': blocks,
                   'speaker': speaker, 'key': key, 'kind': kind, 'group': group}
            self.next_id += 1
            self.messages.append(msg)
            if len(self.messages) > 6000:
                del self.messages[:1000]
        return msg

    def system(self, t):
        return self.post('system', [text(t)])

    def moderator(self, blocks, group=None):
        return self.post('moderator', blocks, speaker='主持人', group=group)

    def say(self, d, blocks, kind=None, group=None):
        return self.post('debater', blocks, speaker=d.name, key=d.key, kind=kind, group=group)

    def post_line(self, ln):
        """One line of a meeting scene becomes a chat message."""
        blocks = ln.get('blocks') or [text(ln['text'])]
        who = ln['who']
        if who == 'divider':
            return self.post('divider', blocks)
        if who == 'call':
            return self.post('system', blocks)
        if who == 'moderator':
            return self.moderator(blocks)
        return self.say(self.by_key[who], blocks, kind=ln.get('kind'))

    def since(self, after):
        with self.lock:
            msgs = [m for m in self.messages if m['id'] > after]
        return {'messages': msgs, 'busy': self.busy, 'state': self.state()}

    def state(self):
        m = self.meeting
        debaters = []
        for d in self.debaters:
            info = {'key': d.key, 'name': d.name, 'school': d.school, 'short': d.short, 'color': COLORS[d.key]}
            if m is not None:
                v = m.db.views[d.key]
                info.update(exposure=round(v.exposure, 3), flat=bool(v.flat_reason), mode_out=v.mode_out)
            debaters.append(info)
        meeting = None
        if m is not None:
            db, ctx = m.db, m.ctx
            meeting = {'date': m.date, 'time': m.spec['key'], 'label': m.label, 'stage': ctx.stage,
                       'kobe': ctx.kobe_stage, 'position': db.position, 'flat': db.flat_votes, 'voters': db.voters,
                       'mainline': ctx.mainline, 'blind': m.blind, 'picks': [f'{ch.name}×{ch.count}' for ch in db.picks]}
        p = self.player
        playing = p is not None and p.active
        speaking = None
        if playing and p.next_who in self.by_key:
            d = self.by_key[p.next_who]
            speaking = {'name': d.name, 'short': d.short, 'color': COLORS[d.key]}
        elif playing and p.next_who == 'moderator':
            speaking = {'name': '主持人', 'short': '主持', 'color': '#343a40'}
        play = {'playing': playing, 'paused': bool(playing and p.paused), 'speed': self.speed,
                'speaking': speaking, 'stage': p.lines[min(p.i, len(p.lines) - 1)]['stage'] if playing else None,
                'progress': [p.i, len(p.lines)] if p is not None else None}
        return {'busy': self.busy, 'reset': self.reset_seq, 'meeting': meeting, 'debaters': debaters,
                'dates': self.full_days, 'examples': EXAMPLES, 'play': play}

    def submit(self, raw, quiet=False):
        """quiet: a meeting-control button press, not echoed into the chat as the user's message."""
        raw = (raw or '').strip()
        if not raw:
            return {'ok': False, 'error': '空指令'}
        with self.lock:
            if self.busy:
                return {'ok': False, 'error': '上一条指令还在处理，请稍等'}
            self.busy = True
        if not quiet:
            self.post('user', [text(raw)])
        threading.Thread(target=self._run, args=(raw,), daemon=True).start()
        return {'ok': True}

    def _run(self, raw):
        try:
            self.handle(raw)
        except Exception as exc:  # report to the window, keep the session alive
            traceback.print_exc()
            self.system(f'出错了：{type(exc).__name__}: {exc}')
        finally:
            with self.lock:
                self.busy = False

    def _ensure_data(self):
        if self.data is None:
            self.system('第一次开会：读取日线面板和除权数据（约 10 秒）…')
            self.data = mt.Data()
            self.days = self.data.days

    # ---- dispatch
    def handle(self, raw):
        """Meeting controls first; any other instruction during a meeting pauses it, gets answered, resumes."""
        t = raw.replace(chr(0x3000), ' ').strip()
        has = lambda *words: any(w in t for w in words)  # noqa: E731
        p = self.player
        if p is None or not p.active or has('开会', '开圆桌', '召开', '开一场', '开个会', '再开', '换到', '切到'):
            return self._dispatch(t)
        if has('暂停', '等一下', '停一下'):
            p.paused = True
            return self.system('会议暂停。说「继续」接着开。')
        if t in ('继续', '继续开会', '接着开', '继续吧'):
            p.paused = False
            return self.system('会议继续。')
        if has('快进', '直接看结果', '快点结束', '全部显示'):
            p.rush, p.paused = True, False
            return self.system('快进：剩下的发言一次显示。')
        if has('跳到唱票', '直接唱票', '开始唱票', '跳到表决', '直接表决'):
            p.jump_to, p.paused = TALLY_STAGE, False
            return None
        if has('结束会议', '散会', '停止会议'):
            p.jump_to, p.paused = VERDICT_STAGE, False
            return None
        speed = '快' if has('快一点', '加速', '说快点') else '慢' if has('慢一点', '减速', '说慢点') else None
        for word in ('慢', '正常', '快'):
            if t in (f'速度{word}', f'速度 {word}'):
                speed = word
        if speed:
            self.speed = p.speed = speed
            return self.system(f'发言速度：{speed}。')
        was_paused = p.paused
        p.paused = True
        self.system('（主持人：有人举手，先回答问题）')
        try:
            self._dispatch(t)
        finally:
            if not was_paused and p.active:
                self.system('（主持人：好，回到会议）')
                p.paused = False
        return None

    def _dispatch(self, raw):
        t = raw.replace('\u3000', ' ')
        has = lambda *words: any(w in t for w in words)  # noqa: E731
        if has('帮助', '怎么用', '指令', '命令') or t.strip().lower() in ('?', '？', 'help'):
            return self.help()
        if has('清空', '清屏'):
            return self.clear()
        if has('导出', '保存对话', '保存记录'):
            return self.export()
        if has('名单', '都有谁', '成员', '辩手列表'):
            return self.roster()
        gone = sorted({RETIRED[w] for w in RETIRED if w in t})
        if gone:
            return self.system('；'.join(f'{name}已经退出圆桌（{why}）' for name, why in gone)
                               + f'。现在圆桌是 {len(self.debaters)} 位辩手；退出的辩手 Skill 目录还在，可以单独使用。')
        date, explicit = parse_date(t, self.days, self.meeting.date if self.meeting else None)
        time_ = parse_time(t)
        if has('开会', '开圆桌', '召开', '开一场', '开个会', '再开', '换到', '切到') or (
                (explicit or time_) and not has('怎么看', '讨论', '辩论', '战绩', '结果', '投票', '选票', '唱票', '共识', '裁决')):
            return self.convene(date, time_, blind=has('盲测', '不看结果'), audit=has('审计'))
        if self.meeting is None:
            return self.need_meeting()
        if has('完整材料', '全部材料', '会议材料', '全部展开'):
            return self.materials()
        m = self.meeting
        stocks = mt.find_stocks(m, t)
        rest = t
        for idx in stocks:
            r = mt.row(m, idx)
            rest = rest.replace(r.code, ' ').replace(r.name or '\0', ' ')
        ds = self.find_debaters(rest)
        codes = re.findall(r'(?<!\d)\d{6}(?!\d)', t)
        if codes and not stocks:
            return self.system(f'{m.ctx.cutoff} 之前没有看到 {"、".join(codes)} 的成交（代码不存在、停牌，或还没开盘成交）')
        if has('结果', '复盘', '后来', '事后', '收益'):
            return self.outcomes()
        if has('战绩', '胜率', '历史表现', '回测表现', '靠谱'):
            return self.record(ds[0] if ds else None)
        if has('对比', '比较', 'PK', 'pk', 'VS', 'vs') and len(ds) >= 2:
            return self.compare(ds[0], ds[1])
        if stocks and ds:
            for d in ds:
                self.ask(d, stocks[0])
            return None
        if stocks:
            for idx in stocks[:3]:
                self.discuss(idx)
            return None
        if ds:
            for d in ds:
                self.view(d)
            return None
        if has('唱票', '统计', '票数'):
            return self.tally()
        if has('投票', '选票', '谁投', '谁选'):
            return self.votes()
        if has('裁决', '结论', '共识', '选股', '入选', '推荐', '选了'):
            return self.verdict()
        if has('排板'):
            return self.watch()
        if has('主线', '板块', '题材', '联动'):
            return self.mainline()
        if has('简报', '盘面', '大盘', '市场', '情绪'):
            return self.briefing()
        if has('大家怎么看', '大家', '各位', '表态'):
            return self.overview()
        return self.unknown(raw)

    def find_debaters(self, t):
        hits = []
        for d in self.debaters:
            pos = min((t.find(a) for a in (d.name,) + d.aliases if a in t), default=-1)
            if pos >= 0:
                hits.append((pos, d))
        return [d for _, d in sorted(hits, key=lambda x: x[0])]

    # ---- helpers about the current meeting
    def _lab(self, idx):
        return label(mt.row(self.meeting, idx))

    def _name(self, idx):
        return mt.row(self.meeting, idx).name

    def _names(self, keys):
        return '、'.join(self.by_key[k].name for k in keys) or '—'

    def _opening(self, d):
        return next((noms, abstain) for b, noms, abstain in self.meeting.db.opening if b.key == d.key)

    def _debated(self):
        return list(self.meeting.db.choices.values())

    def _status(self, idx):
        db = self.meeting.db
        ch = db.choices.get(idx)
        if ch is None:
            return '本场没有辩手选它'
        who = self._names(ch.keys)
        if ch.rank:
            sealed = f'；{self.meeting.ctx.cutoff} 已封涨停，只能排队' if ch.sealed else ''
            return f'本场共识票第 {ch.rank} 名：{ch.count} 票（{who}）{sealed}'
        return f'本场 {ch.count} 票（{who}），不到 {config.CONSENSUS_MIN} 票，不是共识票'

    def _choice_text(self, d):
        db = self.meeting.db
        noms, abstain = self._opening(d)
        if db.views[d.key].mode_out:
            return f'不参加（{abstain}）'
        if not noms:
            return f'不选（{abstain}）'
        return '、'.join(db.choices[i].name + ('（共识）' if db.choices[i].rank else '') for i, *_ in noms)

    def _opening_blocks(self, d):
        m = self.meeting
        v = m.db.views[d.key]
        noms, abstain = self._opening(d)
        head = '本场不参加' if v.mode_out else f'买入意愿 {v.exposure:.0%}'
        blocks = [text(f'**{head}**｜{v.stance}')]
        if noms:
            blocks.append(items([f'选票 **{self._lab(idx)}**「{model}」：{why}' for idx, model, why, _ in noms]))
        else:
            blocks.append(text(f'不选：{abstain}'))
        return blocks

    def convene(self, date, time_, blind=False, audit=False):
        if date and date.startswith('?'):
            return self.system(f'D:\\data 里没有对应的交易日（{date[1:]}）。现有 {self.days[0]} – {self.days[-1]}')
        cur = self.meeting
        if date is None:
            date = cur.date if cur else (self.full_days[-1] if self.full_days else None)
        if time_ is None:  # keep the current meeting's time of day when only the day changes
            time_ = cur.spec['key'] if cur else '10:00'
        time_ = {'09:30': '09:31', '13:00': '13:01'}.get(time_, time_)  # session breaks have no bar
        try:
            spec = config.meeting(time_)
        except ValueError as exc:
            return self.system(str(exc))
        if date not in self.days:
            return self.system(f'D:\\data 里没有 {date} 的分钟数据（现有 {self.days[0]} – {self.days[-1]}）')
        if self.player is not None and self.player.active:
            self.player.stopped = True
            self.player.thread.join(timeout=2)
            self.system('（上一场会议到此结束）')
        self.system(f'准备 {date} {spec["label"]} 圆桌{"（盲测）" if blind else ""}：主持人正在整理盘面和各位的发言…')
        self._ensure_data()
        try:
            m = mt.convene(self.data, date, time_, blind=blind, audit=audit, log=self.system)
        except ValueError as exc:
            return self.system(str(exc))
        path = mt.write_record(m, self.out_dir / 'days')
        self.meeting = m
        if audit:
            self.system('防偷看审计：' + ('通过（截止时点之后的数据换成随机数，结论不变）' if m.audit_passed else '未通过！'))
        lines = scene.build(m)
        lines.append(scene.line('call', f'会议记录已保存：{path}。会后可以继续问：大家怎么看 <股票>｜<辩手>怎么看｜'
                                         '唱票｜结果｜完整材料', stage=VERDICT_STAGE))
        self.player = Player(self, lines, speed=self.speed)
        self.player.start()
        return None

    def materials(self):
        """The whole record at once (every debater's choices, the count, the consensus picks), grouped; on request."""
        m = self.meeting
        db, ctx = m.db, m.ctx
        self.seq += 1

        def grp(n, title, fold=False):
            return {'id': f'g{self.seq}-{n}', 'title': f'{m.label} · {title}', 'fold': fold}

        self.moderator([items(briefing_lines(ctx))], group=grp(0, '主持人盘面简报'))
        g1 = grp(1, '各位辩手的选票')
        for d, noms, _ in db.opening:
            self.say(d, self._opening_blocks(d), kind='nominate' if noms else 'neutral', group=g1)
        self.moderator([self._tally_table()], group=grp(2, '主持人统计（唱票）'))
        self.moderator(self._verdict_blocks(), group=grp(3, '共识票'))

    def _choices_table(self):
        db = self.meeting.db
        rows = []
        for d, noms, _ in db.opening:
            v = db.views[d.key]
            why = '；'.join(f'{db.choices[i].name}：{model}' for i, model, *_ in noms)
            rows.append([d.name, '—' if v.mode_out else f'{v.exposure:.0%}', self._choice_text(d), why or '—'])
        return table(['辩手', '买入意愿', '选票', '依据（各自框架的模式）'], rows)

    def _tally_table(self):
        m = self.meeting
        ranked = sorted(m.db.choices.values(), key=lambda ch: (-ch.count, -ch.firsts, ch.rank or 999, ch.code))
        if not ranked:
            return text('本场没有辩手选票')
        rows = [[self._lab(ch.idx), ch.count,
                 '、'.join(self.by_key[k].name + ('★' if order == 1 else '') for k, _, _, _, order in ch.choosers),
                 now(mt.row(m, ch.idx)), f'第 {ch.rank} 名' if ch.rank else '—'] for ch in ranked]
        return table(['股票', '票数', '选它的辩手（★第一选择）', '开会时', '共识票'], rows)

    def _verdict_blocks(self):
        m = self.meeting
        db, ctx = m.db, m.ctx
        blocks = []
        if db.picks:
            rows = [[str(ch.rank), self._lab(ch.idx), ch.count, self._names(ch.keys), buy_way(ctx, ch)] for ch in db.picks]
            blocks.append(table(['名次', '股票', '票数', '选它的辩手', '买入方式'], rows))
        lines = [db.note] if db.note else []
        lines.append(f'规则：{config.CONSENSUS_MIN} 位及以上辩手选中的是共识票，按票数从多到少排；没有共识票就空仓')
        lines.append(f'圆桌仓位意见：{db.voters} 位辩手平均买入意愿 {db.position:.0%}，{db.flat_votes}/{db.voters} 位不选')
        blocks.append(items(lines))
        reasons = [f'**{ch.rank}. {ch.name}**（{ch.count} 票）｜'
                   + '；'.join(f'{self.by_key[k].name}「{model}」：{why}' for k, model, why, *_ in ch.choosers)
                   for ch in db.picks]
        if reasons:
            blocks.append(items(reasons))
        return blocks

    def help(self):
        self.moderator([text('在下面输入指令，辩手们会按各自 Skill 的规则发言（规则引擎生成，只用开会时点之前的数据）。'),
                        table(['指令', '例子', '作用'], HELP_ROWS)])

    def _welcome(self):
        days = f'{self.full_days[0]} – {self.full_days[-1]}' if self.full_days else '（D:\\data 里还没有数据）'
        n = len(self.debaters)
        self.moderator([
            text(f'**游资圆桌 · 对话窗口**：{n} 位辩手是按各自 Skill 规则化的风格模型（不是本人），'
                 '发言全部由规则引擎生成，只用开会时点之前的数据；仅供研究，不构成投资建议。'),
            text(f'可开会的交易日：{days}。先开一场会，比如「开会 {self.full_days[-1] if self.full_days else "2026-09-24"} 10:30」。'
                 f'会议按议程一项一项开：主持人通报盘面，{n} 位辩手依次说判断和选票（只按自己的框架，最多 '
                 f'{config.NOMINATIONS_PER_DEBATER} 只，不互相投票），主持人唱票，{config.CONSENSUS_MIN} 位及以上选中的是共识票，'
                 '按票数排，没有就空仓；会中可以暂停、调速、快进，也可以随时举手提问。'
                 '会后可以问「唱票」「大家怎么看 某只股票」「某位辩手怎么看」「结果」。输入「帮助」看全部指令。'),
        ])

    def need_meeting(self):
        latest = self.full_days[-1] if self.full_days else '2026-09-24'
        self.system(f'还没有开会。先说「开会 {latest} 10:30」（或任意交易日：盘中 09:31–14:59 的任意分钟，'
                    f'或收盘后 {config.CLOSE_REVIEW} 复盘）。')

    def unknown(self, raw):
        self.moderator([text(f'没听懂「{raw}」。可以试试：'), items(EXAMPLES[:8])])

    def roster(self):
        m = self.meeting
        rows = []
        for d in self.debaters:
            if m is None:
                state = '—'
            else:
                v = m.db.views[d.key]
                state = '不参加' if v.mode_out else ('空仓' if v.flat_reason else f'{v.exposure:.0%}')
            rows.append([d.name, d.school, state, '、'.join(d.models)])
        self.moderator([table(['辩手', '风格', '本场买入意愿', '看什么（模式）'], rows)])

    def briefing(self):
        self.moderator([text(f'**{self.meeting.label}**'), items(briefing_lines(self.meeting.ctx))])

    def mainline(self):
        ctx = self.meeting.ctx
        ind_ = ctx.industries
        rows = [[x, int(ind_.at[x, 'lu_touch']), int(ind_.at[x, 'strong']), int(ind_.at[x, 'yday_lu']),
                 f'{ind_.at[x, "mean_ret_s"]:+.1f}'] for x in ind_.index[:8]]
        blocks = [text(f'主线行业：{"、".join(ctx.mainline) or "无"}（按 触板×2 + 强势家数 排序的前 8 个行业）'),
                  table(['行业', '触板', '强势', '昨日涨停', '平均涨幅（10cm口径）'], rows)]
        if ctx.clusters:
            blocks.append(items(
                f'联动簇 #{c["rank"]}：{c["size"]} 只、触板 {c["lu"]}，{"/".join(c["industries"])}，新面孔 {c["new_share"]:.0%}：'
                + '、'.join(n for _, n in c['members'][:10]) for c in ctx.clusters[:5]))
        self.moderator(blocks)

    def votes(self):
        self.moderator([text(f'**{self.meeting.label} · 各位选票**'), self._choices_table()])

    def tally(self):
        self.moderator([text(f'**{self.meeting.label} · 唱票**'), self._tally_table()])

    def verdict(self):
        self.moderator([text(f'**{self.meeting.label} · 共识票**')] + self._verdict_blocks())

    def watch(self):
        m = self.meeting
        sealed = [ch for ch in m.db.picks if ch.sealed]
        if not sealed:
            return self.moderator([text('本场共识票里没有开会时已封涨停的（收盘后复盘是次日开盘买入，不涉及）')])
        self.moderator([text(f'共识票里 {m.ctx.cutoff} 已封涨停的：只能按涨停价排队，之后开板才成交'),
                        items(f'**{ch.name}** {ch.count} 票（{self._names(ch.keys)}）' for ch in sealed)])

    def overview(self):
        db = self.meeting.db
        rows = []
        for d in self.debaters:
            v = db.views[d.key]
            noms, _ = self._opening(d)
            cons = [db.choices[i].name for i, *_ in noms if db.choices[i].rank]
            rows.append([d.name, '不参加' if v.mode_out else f'{v.exposure:.0%}', self._choice_text(d), '、'.join(cons) or '—'])
        self.moderator([text(f'**{self.meeting.label} · 表态总览**（问「某位辩手怎么看」看理由）'),
                        table(['辩手', '买入意愿', '选票', '其中的共识票'], rows)])

    def view(self, d):
        blocks = self._opening_blocks(d)
        db = self.meeting.db
        noms, _ = self._opening(d)
        lines = []
        for i, *_ in noms:
            ch = db.choices[i]
            others = [self.by_key[k].name for k in ch.keys if k != d.key]
            lines.append(f'**{ch.name}**：' + (f'共识票第 {ch.rank} 名，{ch.count} 票（同选的还有 {"、".join(others)}）'
                                               if ch.rank else '只有自己选，不是共识票'))
        if lines:
            blocks.append(items(lines))
        self.say(d, blocks, kind='view')

    def ask(self, d, idx):
        kind, model, why = mt.opinion(self.meeting, d, idx)
        tag = f'「{model}」' if kind in ('nominate', 'support') else ''
        blocks = [text(f'关于 **{self._lab(idx)}**：{KIND[kind]}{tag}——{why}')]
        pool = mt.pool_status(self.meeting, idx)
        if pool:
            blocks.append(text(f'（主持人注：{pool}）'))
        self.say(d, blocks, kind=kind)

    def discuss(self, idx):
        m = self.meeting
        r = mt.row(m, idx)
        notes = [self._status(idx)]
        pool = mt.pool_status(m, idx)
        if pool:
            notes.append(pool)
        self.moderator([text(f'讨论 **{label(r)}**｜{m.ctx.cutoff}：{now(r)}，{vw(r)}，{amt(r)}；{seal(r)}；{streak(r)}；'
                             f'{role(r)}；{ind(r)}'), items(notes)])
        tally = {k: [] for k in ('nominate', 'support', 'soft', 'hard', 'neutral')}
        neutral = []
        for d in self.debaters:
            kind, model, why = mt.opinion(m, d, idx)
            tally[kind].append(d.name)
            if kind == 'neutral':
                neutral.append(f'{d.name}：{why}')
                continue
            tag = f'「{model}」' if kind in ('nominate', 'support') else ''
            self.say(d, [text(f'{KIND[kind]}{tag}：{why}')], kind=kind)
        summary = (f'小结：选了 {len(tally["nominate"])}、符合框架 {len(tally["support"])}、有顾虑 {len(tally["soft"])}、'
                   f'不认可 {len(tally["hard"])}、中立 {len(tally["neutral"])}（只是各自的看法，共识只看选票）')
        blocks = [text(summary)]
        if neutral:
            blocks.append(items(neutral))
        self.moderator(blocks)

    def compare(self, a, b):
        m = self.meeting
        db = m.db

        def col(d):
            v = db.views[d.key]
            return [v.stance, '不参加' if v.mode_out else f'{v.exposure:.0%}', self._choice_text(d)]

        rows = [[k, x, y] for k, x, y in zip(['判断', '买入意愿', '选票'], col(a), col(b))]
        na, nb = self._opening(a)[0], self._opening(b)[0]
        both = {i for i, *_ in na} & {i for i, *_ in nb}
        lines = ['两人都选了：' + '、'.join(db.choices[i].name for i in both)] if both else []
        for x, y, noms in ((a, b, na), (b, a, nb)):
            for i, *_ in noms:
                if i not in both:
                    kind, model, why = mt.opinion(m, y, i)
                    tag = f'「{model}」' if kind == 'support' else ''
                    lines.append(f'{x.name}选的 **{db.choices[i].name}**，{y.name}的看法：{KIND[kind]}{tag}——{why}')
        blocks = [table(['', a.name, b.name], rows)]
        blocks.append(items(lines) if lines else text('两人今天都没有选票'))
        self.moderator(blocks)

    def record(self, d=None):
        m = self.meeting
        mode = 'intraday' if m.spec['mode'] == 'intraday' else 'eod'
        path = config.OUTPUT_DIR / mode / 'nominations.csv'
        if not path.exists():
            return self.system('还没有回测记录：先运行 run_backtest.py')
        noms = pd.read_csv(path, dtype={'code': str})
        days = self.days
        if mode == 'intraday':
            known = noms['date'] < m.date
        else:
            nxt = dict(zip(days, days[1:]))
            known = noms['date'].map(lambda x: nxt.get(x, '9999-99-99') < m.date)
        sub = noms[known & (noms['filled'] == True)]  # noqa: E712
        has_t1 = 'sell_date' in sub and 't1_mid' in sub
        basis = '10:00 回测' if mode == 'intraday' else f'收盘后 {config.CLOSE_REVIEW} 复盘、次日开盘买入的回测'
        note = (f'只统计 {m.date} 之前已经知道结果的选票（{basis}）：买入日收盘收益，以及卖出日在 {m.date} 之前的 '
                'T+1 均价卖出收益')
        if m.spec['key'] != config.MODES[mode]['key']:
            note += f'；本场是 {m.spec["key"]}，借用 {config.MODES[mode]["key"]} 回测的记录作参考'

        def stats(g):
            t1 = g.loc[g['sell_date'].fillna('9999-99-99') < m.date, 't1_mid'].dropna() if has_t1 else pd.Series(dtype=float)
            return [len(g), _pc(g['ret_close'].mean()), f'{g["win"].mean():.0%}', _pc(t1.mean() if len(t1) else None),
                    f'{(t1 > 0).mean():.0%}' if len(t1) else '—']

        head = ['成交笔数', '买入日收盘收益', '买入日胜率', 'T+1 均价卖出收益', 'T+1 胜率']
        if d is None:
            rows = [[name] + stats(g) for name, g in sub.groupby('debater')]
            rows.sort(key=lambda r: -r[1])
            return self.moderator([text(note), table(['辩手'] + head, rows)])
        g = sub[sub['debater'] == d.name]
        if g.empty:
            return self.moderator([text(note), text(f'{d.name} 在这之前没有成交的选票')])
        rows = [['合计'] + stats(g)] + [[model] + stats(x) for model, x in g.groupby('model')]
        self.moderator([text(note), table([f'{d.name} 的模式'] + head, rows)])

    def outcomes(self):
        m = self.meeting
        if m.blind:
            return self.system('这一场是盲测，没有计算事后结果。想看结果：重新开会，不加“盲测”。')
        if m.out is None:
            return self.system(m.note or '还没有事后结果')
        o = m.out
        day_txt = '当日' if m.spec['mode'] == 'intraday' else f'次日（{m.note}）'
        rows = []
        for ch in m.db.picks:
            r = o.loc[ch.idx] if ch.idx in o.index else None
            if r is None or not r['filled']:
                why = '涨停排队未开板' if (r is not None and r['queued']) else '无成交数据'
                rows.append([str(ch.rank), ch.name, ch.count, f'未成交（{why}）'] + ['—'] * 7)
                continue
            sell = r['sell_date'] if isinstance(r['sell_date'], str) else '待数据'
            if r.get('t1_stuck') == True:  # noqa: E712
                sell += '（一字跌停，实际卖不掉）'
            rows.append([str(ch.rank), ch.name, ch.count, '排板成交' if r['queued'] else '成交', f'{r["entry"]:.2f}',
                         _pc(r['buy_chg']), _pc(r['ret_close']), sell, _pc(r['t1_best']), _pc(r['t1_mid']), _pc(r['t1_worst'])])
        blocks = [text(f'以下是决策之后才发生的事（开会时看不到）。买入日：{day_txt}；卖出：买入后的下一个交易日（T+1），'
                       '按最高价、高低均价、最低价三种算，收益相对买入价。'),
                  table(['名次', '股票', '票数', '成交', '买入价', '买入时涨幅', '买入日收盘收益', '卖出日', '最高价卖出',
                         '均价卖出', '最低价卖出'], rows) if rows else text('本场没有共识票，空仓。')]
        if m.base:
            b = m.base
            blocks.append(text(f'对照：同日全部可选股 {b["all_n"]} 只，买入日收盘 {_pc(b["all_ret"])}、T+1 均价卖出 '
                               f'{_pc(b["all_t1"])}；被选中的 {b["pool_n"]} 只，买入日收盘 {_pc(b["pool_ret"])}、T+1 均价卖出 '
                               f'{_pc(b["pool_t1"])}'))
        self.moderator(blocks)

    def export(self):
        path = self.out_dir / 'chats' / f'{datetime.now().strftime("%Y%m%d-%H%M%S")}.md'
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            msgs = list(self.messages)
        L = ['# 游资圆桌 · 对话记录', '', '> 辩手是按各自 Skill 规则化的风格模型，不是本人；仅供研究。', '']
        group = None
        for msg in msgs:
            if msg['role'] == 'divider':
                L += [f'## {msg["blocks"][0]["text"]}', '']
                continue
            g = msg.get('group')
            if g and g['id'] != group:
                L += [f'## {g["title"]}', '']
            group = g['id'] if g else None
            who = {'user': '我', 'system': '系统'}.get(msg['role'], msg['speaker'])
            kind = f'（{KIND[msg["kind"]]}）' if msg.get('kind') in KIND and msg['role'] == 'debater' else ''
            L.append(f'**{who}{kind}** {msg["time"]}')
            for b in msg['blocks']:
                if b['type'] == 'text':
                    L.append(b['text'])
                elif b['type'] == 'list':
                    L += [f'- {x}' for x in b['items']]
                elif b['type'] == 'table':
                    L.append('| ' + ' | '.join(map(str, b['headers'])) + ' |')
                    L.append('|' + '---|' * len(b['headers']))
                    L += ['| ' + ' | '.join(str(c).replace('|', '／') for c in r) + ' |' for r in b['rows']]
            L.append('')
        path.write_text('\n'.join(L), encoding='utf-8')
        self.system(f'对话已导出：{path}')

    def clear(self):
        with self.lock:
            self.messages = []
            self.reset_seq += 1
        self.system('窗口已清空（当前会议仍然保留）' if self.meeting else '窗口已清空')
