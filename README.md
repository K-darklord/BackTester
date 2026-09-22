# Backtester

Fund 多 agent 量化项目中的**回测引擎**角色仓库。对 Strategist 的 regime 信号、Analyst 的选股、Trader 的执行决策做统一的 walk-forward 回测，产出标准绩效指标与净值曲线，并把历史回测记录写入 SQLite 供 dashboard 读取。

## 安装

```bash
pip install -e .
```

安装后获得 `backtester` 命令；依赖 `pandas` / `numpy` / `requests` / `pyarrow`（见 [setup.py](setup.py)）。

## 快速开始

```bash
# 运行单次回测（market: US | HK | ASHARE）
backtester run US --years 5 --strategy strategist_regime --save

# 策略 vs 买入持有对比
backtester compare US --years 5

# 三种策略对比（buy_hold / strategist_regime / hmm_strategist）
backtester compare-strategies ASHARE --years 3

# 查看历史回测记录（SQLite）
backtester list --limit 20
```

`run` 支持 `--cost-bps`（单边交易成本，默认 10bp）与 `--notes`（备注，随结果落库）。

## 架构

```
backtester/
├── cli.py                      # 命令行入口（run / compare / compare-strategies / list）
├── core.py                     # BacktestEngine + compute_metrics + BacktestResult
├── signal.py                   # Signal 契约（目标仓位，而非买卖指令）
├── data.py                     # 数据层：FundTeam API ↔ 本地 CSV ↔ Strategist 直连
├── storage.py                  # SQLite 回测历史（metrics + 降采样净值曲线）
├── _paths.py                   # Fund 子仓库 sys.path 管理（__file__ 相对路径）
└── strategies/
    ├── base.py                 # 策略基类
    ├── buy_hold.py             # 买入持有基准
    ├── strategist_regime.py    # 基于 Strategist regime 信号
    ├── hmm_strategist.py        # 基于 HMM 隐状态信号
    └── analyst_picks.py         # 基于 Analyst 选股
```

## 核心设计

**信号契约（signal contract）**：一个策略的 `signal` 是**目标仓位**（`target_position ∈ [0,1]`），而不是买卖指令。这样能避免 `-1` 被误解为"做空"而非"空仓"的经典 bug：

- `1.0` = 满仓
- `0.0` = 空仓
- `0.5` = 半仓

**无前视偏差**：日期 T 生成的信号作用于 T+1 的收益（next-day execution）。

**交易成本**：`--cost-bps` 为单边成本，一次完整的 `0 → 1 → 0` 往返计 `2 * cost_bps`，以每日收益拖累的形式按仓位变动幅度计提。

## 数据源

优先走 FundTeam 的 `/api/data/etf_basket` HTTP 接口；不可用时回退到直接导入 Strategist 的 `data_fetcher`（ETF 篮子：`US_ETFS` / `HK_ETFS` / `ASHARE_ETFS`）。也支持本地 CSV/parquet。连接地址由环境变量 `FUNDTEAM_URL` 控制。

## 绩效指标

`total_return` / `cagr` / `sharpe` / `sortino` / `max_drawdown` / `volatility` / `win_rate` / `n_trades`，并额外输出 `excess_return` / `excess_sharpe`（相对买入持有基准）与 `transaction_costs_total`。