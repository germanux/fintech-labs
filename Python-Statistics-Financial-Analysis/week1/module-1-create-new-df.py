import pandas as pd
import matplotlib.pyplot as plt
plt.show()
ms = pd.read_csv('../data/microsoft.csv', index_col='Date', parse_dates=['Date'])

#%%
#Your turn to create PriceDiff in the DataFrame ms
ms['PriceDiff'] = ms['Close'].shift(-1) - ms['Close']

#%%
#Your turn to create a new column Return in the DataFrame MS
ms['Return'] = ms['PriceDiff'] / ms['Close']

ms['Direction'] = [1 if ms['PriceDiff'].loc[el] > 0 else 0 for el in ms.index]

ms.to_csv('../data/microsoft_with_diff.csv')
print('Price difference on {} is {}. direction is {}'.format('2015-01-06', ms['PriceDiff'].loc['2015-01-06'], ms['Direction'].loc['2015-01-06']))

# You can use .rolling() to calculate any numbers of days' Moving Average. This is your turn to calculate "60 days"
# moving average of Microsoft, rename it as "ma60". And follow the codes above in plotting a graph

ms['ma60'] = ms['Close'].rolling(60).mean()

#plot the moving average
plt.figure(figsize=(10, 8))
ms['ma60'].loc['2015-01-01':'2015-12-31'].plot(label='MA60')
ms['Close'].loc['2015-01-01':'2015-12-31'].plot(label='Close')
plt.legend()
plt.show()
