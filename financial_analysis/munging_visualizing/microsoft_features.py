import pandas as pd
import matplotlib.pyplot as plt
# Load Microsoft stock data from CSV file
ms = pd.read_csv('../data/microsoft.csv', index_col='Date', parse_dates=['Date'])

# Calculate the price difference between the next day's close and today's close
ms['PriceDiff'] = ms['Close'].shift(-1) - ms['Close']

# Calculate the daily return
ms['Return'] = ms['PriceDiff'] / ms['Close']

# Determine the price movement direction (1 for up, 0 for down/flat)
ms['Direction'] = [1 if ms['PriceDiff'].loc[el] > 0 else 0 for el in ms.index]

# Save the processed DataFrame to a new CSV file
ms.to_csv('../data/microsoft_with_diff.csv')

# Print price difference and direction for a specific date
print('Price difference on {} is {}. direction is {}'.format('2015-01-06', ms['PriceDiff'].loc['2015-01-06'], ms['Direction'].loc['2015-01-06']))

# Calculate the 60-day moving average (MA60) of the closing price
ms['ma60'] = ms['Close'].rolling(60).mean()

# Plot the 60-day moving average and the closing price for the year 2015
plt.figure(figsize=(10, 8))
ms['ma60'].loc['2015-01-01':'2015-12-31'].plot(label='MA60')
ms['Close'].loc['2015-01-01':'2015-12-31'].plot(label='Close')
plt.title('Microsoft Stock Price and 60-day Moving Average (2015)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
