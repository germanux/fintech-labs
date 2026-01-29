import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def backtest_ma_crossover(df: pd.DataFrame, fast: int, slow: int, fee_bps: float = 0.0):
    """
    Moving-average crossover backtest (long-or-flat).

    Parameters
    ----------
    df : pd.DataFrame
        Input OHLCV-like DataFrame. Must contain at least a 'Close' column.
        Ideally indexed by datetime for easy slicing, but not strictly required.
    fast : int
        Window length (in trading days) for the fast moving average.
    slow : int
        Window length (in trading days) for the slow moving average. Must be > fast.
    fee_bps : float
        Trading cost in basis points (bps) applied whenever the position changes.
        Example: 10 bps = 0.10% = 0.001 in decimal return.
        This is a simplified way to model commissions + spread + slippage.

    Returns
    -------
    dict
        Summary metrics of the strategy:
        - total_return: final strategy return
        - bh_return: buy-and-hold return
        - sharpe: simple daily Sharpe annualized by sqrt(252)
        - max_dd: maximum drawdown of the strategy equity curve
        - trades: number of position changes (entries/exits)
        - final_eq: final equity (starting at 1.0)
    """

    # Work on a copy with only the 'Close' price to keep things simple and explicit
    x = df[['Close']].copy()

    # Daily simple returns:
    # ret[t] = Close[t] / Close[t-1] - 1
    x['ret'] = x['Close'].pct_change()

    # Compute moving averages (simple moving averages, SMA)
    x['ma_fast'] = x['Close'].rolling(fast).mean()
    x['ma_slow'] = x['Close'].rolling(slow).mean()

    # Drop the initial rows where moving averages are NaN
    # (you can't generate signals until enough history exists)
    x = x.dropna()

    # Raw position signal:
    # 1 means "in the market / long 1 unit" when fast MA is above slow MA
    # 0 means "flat / out of the market" otherwise
    x['pos'] = (x['ma_fast'] > x['ma_slow']).astype(int)

    # IMPORTANT (anti-lookahead):
    # We shift the position by 1 day so that today's signal is applied to tomorrow's return.
    # Otherwise you would be implicitly using today's close to decide and profit from today's move.
    x['pos_lag'] = x['pos'].shift(1).fillna(0)

    # Trade detection:
    # pos.diff().abs() is 1 when you switch between 0 and 1 (enter or exit), 0 otherwise.
    x['trade'] = x['pos'].diff().abs().fillna(0)

    # Convert bps (basis points) to decimal and apply only on trade days
    # Example: 10 bps => 10/10000 = 0.001 = 0.1%
    cost = (fee_bps / 10_000.0) * x['trade']

    # Strategy return:
    # - If pos_lag = 1, you earn the market return for that day.
    # - If pos_lag = 0, you earn 0 that day.
    # - Subtract costs on trade days.
    x['strat_ret'] = x['pos_lag'] * x['ret'] - cost

    # Buy & hold return (baseline)
    x['bh_ret'] = x['ret']

    # Equity curves (starting at 1.0):
    # equity[t] = Π(1 + return[t])
    x['strat_eq'] = (1 + x['strat_ret']).cumprod()
    x['bh_eq'] = (1 + x['bh_ret']).cumprod()

    # Helper: maximum drawdown computed from the equity curve
    def max_drawdown(equity: pd.Series) -> float:
        peak = equity.cummax()            # running peak
        dd = equity / peak - 1.0          # drawdown series (<= 0)
        return float(dd.min())            # worst drawdown (most negative)

    # Total returns relative to 1.0 initial equity
    total = x['strat_eq'].iloc[-1] - 1
    bh_total = x['bh_eq'].iloc[-1] - 1

    # Simple annualized Sharpe:
    # sqrt(252) * mean(daily_return) / std(daily_return)
    # NOTE: This is a basic Sharpe; in real work you'd handle risk-free rate, stability, etc.
    sharpe = np.sqrt(252) * x['strat_ret'].mean() / (x['strat_ret'].std(ddof=0) + 1e-12)

    return {
        'fast': fast,
        'slow': slow,
        'total_return': float(total),
        'bh_return': float(bh_total),
        'sharpe': float(sharpe),
        'max_dd': max_drawdown(x['strat_eq']),
        'trades': int(x['trade'].sum()),
        'final_eq': float(x['strat_eq'].iloc[-1]),
    }


def grid_search(df: pd.DataFrame, fast_list, slow_list, fee_bps: float = 0.0) -> pd.DataFrame:
    """
    Brute-force parameter sweep over (fast, slow) combinations.

    Returns a DataFrame sorted by Sharpe then total return.
    """
    rows = []

    for fast in fast_list:
        for slow in slow_list:
            # Enforce the usual constraint: fast < slow
            if fast >= slow:
                continue

            # Run the backtest and collect metrics
            rows.append(backtest_ma_crossover(df, fast, slow, fee_bps=fee_bps))

    # Convert list of dicts to a DataFrame for easy sorting/inspection
    res = pd.DataFrame(rows).sort_values(['sharpe', 'total_return'], ascending=False)
    return res


# -------------------------
# Example usage with Microsoft data (2015)
# -------------------------

# Load Microsoft CSV, parse 'Date' as datetime, set it as index
ms = pd.read_csv("../data/microsoft.csv", parse_dates=["Date"], index_col="Date")

# Slice one year (2015); adjust end date as needed
ms2015 = ms.loc["2015-01-01":"2015-12-01"]

# 1) Run one backtest with chosen parameters (fast=10, slow=30)
r = backtest_ma_crossover(ms2015, fast=10, slow=30, fee_bps=10)
print(r)

# 2) Grid search across ranges of parameters
results = grid_search(ms2015, fast_list=range(5, 31, 5), slow_list=range(20, 201, 20), fee_bps=10)
print(results.head(10))

# -------------------------
# Plot an equity curve (make sure parameters match what you want!)
# -------------------------

# Build a quick equity curve for plotting
x = ms2015[["Close"]].copy()
x["ret"] = x["Close"].pct_change()

# NOTE: Your original file used slow=60 here, which may not match the printed backtest above.
# Choose the same windows if you want consistency.
fast = 10
slow = 30

x["pos"] = (x["Close"].rolling(fast).mean() > x["Close"].rolling(slow).mean()).astype(int)

# Shift to avoid lookahead: today's signal applied to tomorrow's return
x["pos_lag"] = x["pos"].shift(1).fillna(0)

# Equity curve for the strategy (no costs in this quick plot unless you also subtract them)
x["eq"] = (1 + x["pos_lag"] * x["ret"]).cumprod()

# Plot
x["eq"].plot()
plt.title(f"Equity curve (MA{fast} vs MA{slow})")
plt.xlabel("Date")
plt.ylabel("Equity (starts at 1.0)")
plt.show()
