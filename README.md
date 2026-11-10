# 最优组合每日复盘（D:\review）

每个交易日 15:10 自动运行：把当天的 1 分钟线拉到 D:\data → 刷新除权除息 → 用冻结的圆桌引擎回测全部历史并开当天的
15:10 复盘 → 按最优组合的规则算出 T+1 买入清单 → 生成固定格式的复盘页。没有符合规则的票就空仓，不凑数。

## 规则（冻结，2026-10-17 指定为最优组合）

- 成员：炒股养家、92科比、陈小群、赵老哥、乔帮主；章盟主、方新侠、作手新一、瑞鹤仙的选票不看。
- 只用收盘后 15:10 复盘，只算每位成员的第一选择。
- 同一只股票被至少 2 位成员列为第一选择 → 下一个交易日开盘买入（开盘一字涨停的挂涨停价排队，全天不开板就买不到；
  几只同时符合就平分资金；没有就空仓）。
- 卖出：再下一个交易日，按 config.json 的 `sell_rule`（open = 开盘价，close = 收盘价，mid = 最高/最低均价，即回测口径）。
- 结果一律给三种情况（2026-10-18 起，复盘页、index.html、台账都一样）：最好 = 卖出日最高价卖出，最差 = 最低价卖出，
  平均 = 最高价和最低价的均价卖出（回测口径）。那天不管什么时候卖，收益都在最好和最差之间。
- 规则写在冻结的 `skills\stock-roundtable\optimal_combo.py`，复盘页和回测记录用的是同一份定义。

## 每天的流程

| 时间 | 做什么 |
|---|---|
| 15:10 | 计划任务 Review-OptimalCombo 运行 `run_review.ps1` |
| 约 15:12 | 当天 1 分钟线写入 D:\data（共用的下载器 `--early` 模式：≥99% 的股票有 15:00 那根才写；没收完每 2 分钟重试，最晚 15:40） |
| 约 15:13 | 刷新除权除息 |
| 约 15:22 | 回测完成（约 9 分钟；历史每多一天约多 5 秒） |
| 约 15:23 | 复盘页生成，浏览器自动打开 `reports\latest.html`（桌面快捷方式“最优组合复盘”） |
| 16:40 | 再运行一次：今天已经完成就只做数据核对（重新下载当天分钟线对比，不改 D:\data）；15:10 没完成就重做 |

周末和节假日（按 TDX 交易日历）不运行。电脑在 15:10 关机或睡眠时，开机后会补跑。TDX 的交易日历只知道已经交易过的日子，
所以页面上的 T+1 日期按“下一个工作日”推算：交易所公布休市安排后（例如 2027 年的），把休市日加进 config.json 的
`holidays`（格式 `"2027-01-01"`），T+1 日期就会顺延。

## 复盘页（每天同样的几部分）

1. 状态：数据完整度、文件指纹、引擎版本（冻结副本有没有被改）、防偷看审计、两套计算是否一致。
2. T+1 买入清单（或“空仓”）。
3. 五位成员的选票：① 第一选择（算）、② 第二选择（不算）、理由；另外四位折叠在下面。
4. 持仓与卖出提醒：今天买入的、明天要卖的；今天卖出的结果（最好 / 最差 / 平均）。
5. 盘面简报、圆桌九位的共识票（只作参考）。
6. 战绩：选定后（2026-10-13 之后）和回测全期，按最好 / 最差 / 平均三种情况各算一遍；最近 10 笔。
7. 规则与风险、运行步骤和耗时。

## 目录

| 路径 | 内容 |
|---|---|
| `run_review.ps1` | 入口（计划任务和手动都用它） |
| `install_task.ps1` | 注册计划任务、放桌面快捷方式 |
| `config.json` | Python 路径、D:\data、下载器、卖出规则、数据截止时间、重试间隔 |
| `pipeline\` | `run.py`（总流程）、`pull_today.py`、`review_day.py`、`render.py`、`verify_data.py`、`freeze.py` |
| `skills\` | 冻结的圆桌引擎 v2.2（`stock-roundtable\`，自己的 cache 和 output）和九位辩手的 SKILL.md；`FROZEN.json` 是指纹清单 |
| `reports\` | `YYYY-MM-DD.html` / `.json`、`latest.html`、`index.html`（全部复盘：三种情况的战绩、每只信号的最好 / 最差 / 平均） |
| `ledger\signals.csv` | 每只信号股和之后的结果：最好 / 最差 / 平均收益%，另列开盘、收盘卖出收益%（每次运行重写） |
| `logs\` | 每天的日志、各步骤的输出、一致性核对报告 |
| `tests\check_consistency.py` | 和圆桌当前结果逐项核对 |
| `timing\` | 买卖点：研究脚本、方案、盘前计划和盘中提示（见下一节） |

## 手动运行

最简单：双击桌面上的“立即复盘”（或 D:\review\立即复盘.cmd）。窗口里显示每一步，完成后打开最新的复盘页。
交易日 15:10 以后才会复盘；周末、节假日和 15:10 以前只会打开上一个交易日的复盘页。当天已经复盘过的不会重做。

```
powershell -ExecutionPolicy Bypass -File D:\review\run_review.ps1 -Show                   # 和“立即复盘”一样
powershell -ExecutionPolicy Bypass -File D:\review\run_review.ps1 -Show -Date 2026-10-14  # 回放过去某天（用现有回测结果）
python -X utf8 D:\review\pipeline\run.py --full                                       # 今天已完成也强制重跑
python -X utf8 D:\review\pipeline\run.py --date 2026-10-16 --full                     # 过去某天也跑全部步骤
python -X utf8 D:\review\tests\check_consistency.py                                   # 和圆桌当前结果核对
python -X utf8 D:\review\pipeline\render.py                                           # 只重新生成全部复盘页、index.html 和台账（不重跑复盘，约 1 分钟）
```

用 AI 助手（Claude Code、DeepSeek、GLM 等）在这个文件夹里工作时，它们会读 AGENTS.md（Claude Code 通过 CLAUDE.md）：
只运行上面的命令、原样转告结果，不自己选股。

## 买卖点（timing\）

选股不变，只定执行时点。2026-10-26 对已结算的 69 笔信号逐分钟回放了 200 多条买法、180 多条卖法，结论和每天的动作在
`timing\买卖点方案.md`，完整对比表在 `timing\output\report.md`：

- 买入日：09:30 开盘价买（现行不变）。等回调、定时、挂低价、突破、半仓低吸都更差；跌停买只在少数地天板上占便宜。
- 卖出日：开盘前挂 **昨收 +7%** 的卖单（不超过涨停价），**14:50** 没成交就市价卖。69 笔里每笔 +1.16%（开盘卖）→ +3.47%，胜率 59% → 70%。

```
python -X utf8 D:\review\timing\plan.py            # 盘前：今天要买 / 要卖的票和价格台阶（昨收、涨停、跌停、挂单价）
python -X utf8 D:\review\timing\watch.py           # 盘中 09:15–15:01：轮询 TDX 行情，到价 / 到时 / 开盘提示，只提示不下单
python -X utf8 D:\review\timing\timing_study.py    # 重新验证（有缓存约 1 分钟），结果写 timing\output\
```

参数在 `timing\rules.json`。复盘页和台账仍按三种情况展示；要让页面高亮收盘卖（最接近新规则的现成口径），把 config.json 的
`sell_rule` 改成 `close` 再运行 `pipeline\render.py`。

## 冻结的引擎

`skills\stock-roundtable` 是 2026-10-17 从 D:\skills\stock\stock-roundtable 原样复制的 v2.2，策略就以它为准；圆桌那边
以后怎么改都不影响这里。每次运行都会核对 `FROZEN.json` 里的指纹，有文件被改，复盘页会警告。要换新版本：
`python pipeline\freeze.py --force`，换完必须重新验证（回测 + 一致性核对）再用。

## 依赖

- D:\data：1 分钟线（和 D:\youlong 的研究共用）。
- D:\youlong\scripts\tdx-daily-download.py：共用的下载器；本项目用它 2026-10-17 新加的 `--early` 模式（默认行为不变，
  16:30 的 Youlong-Overnight-Forward 任务照常运行，遇到已经下好的日子会跳过）。
- D:\youlong\.catdesk-output 下的日线和名称快照（引擎只读）。
- 网络：TDX 行情服务器（分钟线、交易日历、除权除息；timing\watch.py 的盘中行情）。

## 出问题时

- 看 `logs\<日期>.log` 和 `reports\<日期>.json` 的 steps；各步骤的完整输出在 `logs\<日期>-*.out`。
- 当天数据 15:40 还没收完：页面写“今天没有复盘结果”；16:30 的 Youlong 任务会补下载，16:40 会再跑一次。
- 防偷看审计没通过或两套计算不一致：当天不给信号，请把日志发给 Claude 查。

## 卸载

```
Unregister-ScheduledTask -TaskName Review-OptimalCombo -Confirm:$false
```
再删除桌面上的“最优组合复盘”快捷方式和 D:\review 文件夹。
