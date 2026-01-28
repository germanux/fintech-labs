import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

# Load Microsoft stock data from CSV file
ms = pd.read_csv('../data/microsoft.csv', index_col='Date', parse_dates=['Date'])

# Calculate the price difference between the next day's close and today's close
ms['PriceDiff'] = ms['Close'].shift(-1) - ms['Close']

# Calculate the daily return
ms['Return'] = ms['PriceDiff'] / ms['Close']

# Determine the price movement direction (1 for up, 0 for down/flat)
ms['Direction'] = [1 if ms['PriceDiff'].loc[el] > 0 else 0 for el in ms.index]

# Print price difference and direction for a specific date
print('Price difference on {} is {}. direction is {}'.format('2015-01-06', ms['PriceDiff'].loc['2015-01-06'], ms['Direction'].loc['2015-01-06']))

# Calculate the 60-day moving average (MA60) of the closing price
ms['ma30'] = ms['Close'].rolling(30).mean()
ms['ma60'] = ms['Close'].rolling(60).mean()
ms['ma120'] = ms['Close'].rolling(120).mean()

# Save the processed DataFrame to a new CSV file
ms.to_csv('../data_aux/microsoft_features.csv')


# Plot the 60-day moving average and the closing price for the year 2015
# plt.figure(figsize=(8, 7))

# --- Plot con 2 subplots: (1) Close + MAs, (2) Return ---
df = ms.loc['2015-01-01':'2015-12-31'].copy()

fig, (ax1, ax2) = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

# ===== Subplot 1: Price + averages =====
df['Close'].plot(ax=ax1, label='Close')
df['ma30'].plot(ax=ax1, label='MA30', linestyle=':')
df['ma60'].plot(ax=ax1, label='MA60', linestyle='-.')
df['ma120'].plot(ax=ax1, label='MA120', linestyle='--')
ax1.set_title('Microsoft (2015): Price + Moving Averages')
ax1.set_ylabel('Price')
ax1.set_ylim(35, 60)
ax1.yaxis.set_major_locator(mticker.MultipleLocator(5))
ax1.legend()

# ===== Subplot 2: Return (%) + PriceDiff ($) =====
df['Return'].plot(ax=ax2, label='Return (%)', linestyle=':', linewidth=1.5)
ax2.axhline(0, linewidth=1)
ax2.set_ylabel('Return (%)')

ax2b = ax2.twinx()
df['Direction'].plot(ax=ax2b, label='PriceDiff ($)', linestyle='--', linewidth=1.0, alpha=0.8)
ax2b.set_ylabel('PriceDiff ($)')
ax2b.grid(False)

# Leyenda combinada (ambos ejes)
l1, lab1 = ax2.get_legend_handles_labels()
l2, lab2 = ax2b.get_legend_handles_labels()
ax2.legend(l1 + l2, lab1 + lab2, loc='upper left')

# X ticks cada 2 meses empezando en febrero
ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[2, 4, 6, 8, 10, 12], bymonthday=1))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()
