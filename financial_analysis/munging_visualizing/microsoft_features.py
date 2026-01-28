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
plt.figure(figsize=(8, 7))


# X: display ‘month-year’ such as Jan-2015
ax = plt.gca()
#ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2, ))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[2, 4, 6, 8, 10, 12], bymonthday=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
plt.xticks(rotation=45, ha='right')

# And: start at 20, or at 0, Axis limits:
plt.ylim(35, 60)
ax.yaxis.set_major_locator(mticker.MultipleLocator(5))

ms['Close'].loc['2015-01-01':'2015-12-31'].plot(label='Close')
ms['ma30'].loc['2015-01-01':'2015-12-31'].plot(label='MA30', linestyle=':')
ms['ma60'].loc['2015-01-01':'2015-12-31'].plot(label='MA60', linestyle='-.')
# Vlaues for linestyle: '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
ms['ma120'].loc['2015-01-01':'2015-12-31'].plot(label='MA120', linestyle='--')
ms['Return'].loc['2015-01-01':'2015-12-31'].plot(label='Return')
plt.title('Microsoft Stock Price and 60-day Moving Average (2015)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
