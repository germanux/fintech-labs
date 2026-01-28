import pandas as pd
import matplotlib.pyplot as plt

# Load Facebook stock data and calculate moving averages
# MA10: 10-day moving average, MA50: 50-day moving average
fb = pd.read_csv('../data/facebook.csv')
fb['MA10'] = fb['Close'].rolling(10).mean()
fb['MA50'] = fb['Close'].rolling(50).mean()

# Remove rows with missing values (NaN) resulting from moving average calculation
fb = fb.dropna()
fb.head()

# Trading Strategy:
# If MA10 > MA50, buy one share (1). Otherwise, do nothing (0).
fb['Shares'] = [1 if fb.loc[ei, 'MA10']>fb.loc[ei, 'MA50'] else 0 for ei in fb.index]

# Calculate profit:
# If a share is held (Shares=1), profit is tomorrow's close price minus today's close price.
# Otherwise, profit is 0.
plt.figure(figsize=(10, 8))
fb['Close1'] = fb['Close'].shift(-1)
fb['Profit'] = [fb.loc[ei, 'Close1'] - fb.loc[ei, 'Close'] if fb.loc[ei, 'Shares']==1 else 0 for ei in fb.index]

# Plot the daily profit/loss and a horizontal line at 0 for reference
fb['Profit'].plot()
plt.axhline(y=0, color='red')
plt.title('Daily Profit/Loss from Moving Average Crossover Strategy')
plt.xlabel('Index')
plt.ylabel('Profit')
plt.legend()
plt.show()
