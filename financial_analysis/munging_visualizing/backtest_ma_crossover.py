import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def backtest_ma_crossover(df: pd.DataFrame, fast: int, slow: int, fee_bps: float = 0.0):
    """
    fee_bps: cost per change in position in basis points (e.g. 10 = 0.10%)
    """
    x = df[['Close']].copy()
    x['ret'] = x['Close'].pct_change()

    x['ma_fast'] = x['Close'].rolling(fast).mean()
    x['ma_slow'] = x['Close'].rolling(slow).mean()
    x = x.dropna()

    # Señal: 1 si fast>slow, 0 si no
    x['pos'] = (x['ma_fast'] > x['ma_slow']).astype(int)

    # To avoid ‘looking to the future’: today's position applies to tomorrow's return.
    x['pos_lag'] = x['pos'].shift(1).fillna(0)

    # Cost when changing position (0->1 or 1->0)
    x['trade'] = x['pos'].diff().abs().fillna(0)  #1 when there is change
    cost = (fee_bps / 10_000.0) * x['trade']

    x['strat_ret'] = x['pos_lag'] * x['ret'] - cost
    x['bh_ret'] = x['ret']

    # Equity curves
    x['strat_eq'] = (1 + x['strat_ret']).cumprod()
    x['bh_eq'] = (1 + x['bh_ret']).cumprod()

    # Simple metrics
    def max_drawdown(equity):
        peak = equity.cummax()
        dd = equity / peak - 1.0
        return dd.min()

    total = x['strat_eq'].iloc[-1] - 1
    bh_total = x['bh_eq'].iloc[-1] - 1

    sharpe = np.sqrt(252) * x['strat_ret'].mean() / (x['strat_ret'].std(ddof=0) + 1e-12)

    return {
        'fast': fast,
        'slow': slow,
        'total_return': float(total),
        'bh_return': float(bh_total),
        'sharpe': float(sharpe),
        'max_dd': float(max_drawdown(x['strat_eq'])),
        'trades': int(x['trade'].sum()),
        'final_eq': float(x['strat_eq'].iloc[-1]),
    }

def grid_search(df, fast_list, slow_list, fee_bps=0.0):
    rows = []
    for fast in fast_list:
        for slow in slow_list:
            if fast >= slow:
                continue
            rows.append(backtest_ma_crossover(df, fast, slow, fee_bps=fee_bps))
    res = pd.DataFrame(rows).sort_values(['sharpe', 'total_return'], ascending=False)
    return res

# Ejemplo:
# fast_list = range(5, 31, 5)      # 5,10,15,20,25,30
# slow_list = range(20, 201, 10)   # 20..200
# results = grid_search(df, fast_list, slow_list, fee_bps=10)
# print(results.head(10))

# Carga (Date como índice)
ms = pd.read_csv("../data/microsoft.csv", parse_dates=["Date"], index_col="Date")

ms2015 = ms['2015-01-01':'2015-12-01']
# 1) Un backtest con parámetros concretos
r = backtest_ma_crossover(ms2015, fast=10, slow=30, fee_bps=10)
print(r)

# 2) Grid search de parámetros
results = grid_search(ms2015, fast_list=range(5, 31, 5), slow_list=range(20, 201, 20), fee_bps=10)
print(results.head(10))

fig = plt.plot()
x = ms2015[["Close"]].copy()
x["ret"] = x["Close"].pct_change()
x["pos"] = (x["Close"].rolling(10).mean() > x["Close"].rolling(30).mean()).astype(int).shift(1).fillna(0)
x["eq"] = (1 + x["pos"] * x["ret"]).cumprod()
x["eq"].plot()

plt.show()
