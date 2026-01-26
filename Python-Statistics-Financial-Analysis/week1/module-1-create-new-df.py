import pandas as pd
import matplotlib.pyplot as plt
plt.show()
ms = pd.DataFrame.from_csv('../data/microsoft.csv')

#%%
#Your turn to create PriceDiff in the DataFrame ms
ms['PriceDiff'] = ms['Close'].shift(-1) - ms['Close']

#%%
#Your turn to create a new column Return in the DataFrame MS
ms['Return'] = ms['PriceDiff'] / ms['Close']

ms['Direction'] = [1 if ms['PriceDiff'].loc[el] > 0 else 0 for el in ms.index]
