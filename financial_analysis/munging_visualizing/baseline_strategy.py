import pandas as pd
import matplotlib.pyplot as plt

# ---- Load data ----
fb = pd.read_csv('../data/facebook.csv')

# If the CSV has a Date column, use it as index for better plots
if 'Date' in fb.columns:
    fb['Date'] = pd.to_datetime(fb['Date'])
    fb = fb.set_index('Date')

df = fb.loc['2015-01-01':'2015-12-31'].copy()


# ---- Features: moving averages ----
df['MA10'] = df['Close'].rolling(10).mean()
df['MA50'] = df['Close'].rolling(30).mean()

# Drop rows where MAs are NaN
df = df.dropna()

# ---- Strategy signal: 1 share if fast > slow else 0 ----
df['Shares'] = (df['MA10'] > df['MA50']).astype(int)

# ---- Profit: tomorrow close - today close if Shares==1 else 0 ----
df['Close1'] = df['Close'].shift(-1)
# df['Profit'] = (df['Close1'] - df['Close']).where(df['Shares'].eq(1), 0)
df['Profit'] = [df.loc[ei, 'Close1'] - df.loc[ei, 'Close'] if df.loc[ei, 'Shares']==1 else 0 for ei in df.index]

# This is the same than:
# profit = []
# for ei in fb.index:
#     if fb.loc[ei, 'Shares'] == 1:
#         profit.append(fb.loc[ei, 'Close1'] - fb.loc[ei, 'Close'])
#     else:
#         profit.append(0)
# fb['Profit'] = profit

# ---- Wealth (accumulated P&L) ----
df['wealth'] = df['Profit'].cumsum()

# Save
df.to_csv('../data_aux/facebook_features.csv')

# ---- Plots (subplots) ----
fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(10, 10), sharex=True,
    gridspec_kw={"height_ratios": [3, 1, 2]}
)

# 1) Price + MAs
df['Close'].plot(ax=ax1, label='Close')
df['MA10'].plot(ax=ax1, label='MA10', linestyle=':')
df['MA50'].plot(ax=ax1, label='MA50', linestyle='--')
ax1.set_title('Facebook: Close + Moving Averages')
ax1.set_ylabel('Price')
ax1.legend()

# 2) Signal (Shares)
df['Shares'].plot(ax=ax2, label='Shares (0/1)', drawstyle='steps-post')
ax2.set_ylabel('Shares')
ax2.set_ylim(-0.1, 1.1)
ax2.legend()

# 3) Profit + Wealth
df['Profit'].plot(ax=ax3, label='Daily Profit')
df['wealth'].plot(ax=ax3, label='Wealth (cumsum)', linestyle='--')
ax3.axhline(0, linewidth=1)
ax3.set_title(f"Total money you win is {df['wealth'].iloc[-2]:.2f}")
ax3.set_xlabel('Date' if df.index.name == 'Date' else 'Index')
ax3.set_ylabel('USD (per 1 share)')
ax3.legend()

plt.tight_layout()
plt.show()
