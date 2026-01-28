import pandas as pd
import matplotlib.pyplot as plt

# ---- Load data ----
fb = pd.read_csv('../data/facebook.csv')

# If the CSV has a Date column, use it as index for better plots
if 'Date' in fb.columns:
    fb['Date'] = pd.to_datetime(fb['Date'])
    fb = fb.set_index('Date')

# ---- Features: moving averages ----
fb['MA10'] = fb['Close'].rolling(10).mean()
fb['MA50'] = fb['Close'].rolling(50).mean()

# Drop rows where MAs are NaN
fb = fb.dropna()

# ---- Strategy signal: 1 share if fast > slow else 0 ----
fb['Shares'] = (fb['MA10'] > fb['MA50']).astype(int)

# ---- Profit: tomorrow close - today close if Shares==1 else 0 ----
fb['Close1'] = fb['Close'].shift(-1)
fb['Profit'] = (fb['Close1'] - fb['Close']).where(fb['Shares'].eq(1), 0)

# ---- Wealth (accumulated P&L) ----
fb['wealth'] = fb['Profit'].cumsum()

# Save
fb.to_csv('../data_aux/facebook_features.csv')

# ---- Plots (subplots) ----
fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(10, 10), sharex=True,
    gridspec_kw={"height_ratios": [3, 1, 2]}
)

# 1) Price + MAs
fb['Close'].plot(ax=ax1, label='Close')
fb['MA10'].plot(ax=ax1, label='MA10', linestyle=':')
fb['MA50'].plot(ax=ax1, label='MA50', linestyle='--')
ax1.set_title('Facebook: Close + Moving Averages')
ax1.set_ylabel('Price')
ax1.legend()

# 2) Signal (Shares)
fb['Shares'].plot(ax=ax2, label='Shares (0/1)', drawstyle='steps-post')
ax2.set_ylabel('Shares')
ax2.set_ylim(-0.1, 1.1)
ax2.legend()

# 3) Profit + Wealth
fb['Profit'].plot(ax=ax3, label='Daily Profit')
fb['wealth'].plot(ax=ax3, label='Wealth (cumsum)', linestyle='--')
ax3.axhline(0, linewidth=1)
ax3.set_title(f"Total money you win is {fb['wealth'].iloc[-2]:.2f}")
ax3.set_xlabel('Date' if fb.index.name == 'Date' else 'Index')
ax3.set_ylabel('USD (per 1 share)')
ax3.legend()

plt.tight_layout()
plt.show()
