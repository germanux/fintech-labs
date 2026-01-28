import numpy as np
import pandas as pd

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
    x['trade'] = x['pos'].diff().abs().fillna(0)  # 1 cuando hay cambio
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
