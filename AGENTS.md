# 给 AI 助手的说明（Claude Code、DeepSeek、GLM 等都适用）

这个项目每个交易日收盘后，用冻结的规则引擎算出“最优组合”的 T+1 买入清单。**选股全部由程序完成，AI 不参与选股。**
最优组合：炒股养家、92科比、陈小群、赵老哥、乔帮主五位，只算各自的第一选择，同一只票 ≥2 位相同才买，没有就空仓。

## 用户要“最新的复盘 / 明天买什么”时

1. 交易日 15:10 以后：运行
   `powershell -NoProfile -ExecutionPolicy Bypass -File D:\review\run_review.ps1 -Show`（全部步骤约 12 分钟）。
   15:10 以前、周末和节假日它不会复盘，这时最新的结果就是上一个交易日的 `reports\latest.html`。
   当天已经复盘过（计划任务 15:10 跑过）就不用再跑，直接读结果；确实要重跑用 `python -X utf8 pipeline\run.py --full`。
2. 读 `reports\<复盘日>.json`：`status`（只有 ok 才算数）、`next_day`（T+1 日期）、`review.t1`（T+1 清单，空 = 空仓）、
   `review.audit_passed`、`cross_check`、`steps`。
3. 原样转告：T+1 日期、清单（或“空仓”）、每只的票数和投第一票的成员。不要补票、换票，也不要自己看行情另选。
4. 页面：`reports\latest.html`（最新一期）、`reports\index.html`（全部）；台账 `ledger\signals.csv`。
5. 说到交易结果（收益、战绩）时，一律给三种情况（用户 2026-10-18 指定，以后都用这个格式；页面和台账已经这样输出）：
   最好 = 卖出日最高价卖出（`t1_best`），最差 = 最低价卖出（`t1_worst`），平均 = 最高价和最低价的均价卖出
   （`t1_mid`，回测口径）。不要只给开盘卖出一个数。

## 用户问“今天怎么买 / 怎么卖、买点卖点”时

买卖点规则（2026-10-26 定，依据 69 笔逐分钟回放）在 `timing\买卖点方案.md`：买入日 09:30 开盘价买（一字涨停挂涨停价排队）；
卖出日开盘前挂昨收 +7% 的卖单，14:50 没成交就市价卖。参数在 `timing\rules.json`。

- 盘前：`python -X utf8 timing\plan.py` 打印今天要买 / 要卖的票和价格台阶（昨收、涨停、跌停、挂单价），原样转告。
- 盘中：`python -X utf8 timing\watch.py` 轮询 TDX 行情，到价 / 到时 / 开盘提示，只提示不下单。
- 重新验证：`python -X utf8 timing\timing_study.py`（约 1 分钟，结果在 `timing\output\report.md`）。
- 这些只管执行时点，不改选股；不要因为盘中走势建议用户改票、加仓或低吸。

## 不要做的事

- 不要自己选股或“分析出”买入建议；不要把 `review.roundtable`（圆桌九位的共识票）当成买入信号，只有 `review.t1` 是。
- 不要改 `skills\` 里冻结的引擎和规则（指纹记在 `skills\FROZEN.json`，改了页面会警告，策略要重新验证）。
- 不要删除或覆盖 `D:\data` 里的文件；当天数据只由 `pipeline\pull_today.py` 调用共用下载器写入。

## 其他命令

| 做什么 | 命令 |
|---|---|
| 回放过去某天（用现有回测结果，约 20 秒） | `run_review.ps1 -Show -Date 2026-10-14` |
| 过去某天全部步骤重跑 | `python -X utf8 pipeline\run.py --date 2026-10-16 --full` |
| 和圆桌的结果逐项核对 | `python -X utf8 tests\check_consistency.py` |
| 只重新生成全部复盘页、index.html 和台账（不重跑复盘，约 1 分钟） | `python -X utf8 pipeline\render.py` |
| 出错时看 | `logs\<日期>.log`、`logs\<日期>-*.out`、`reports\<日期>.json` 的 steps |

细节见 README.md。
