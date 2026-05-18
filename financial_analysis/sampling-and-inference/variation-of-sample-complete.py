import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

meanlist = []
varlist = []

# Generate 1000 samples
for t in range(1000):
    # Each sample has 30 random values from N(10, 5)
    sample = pd.DataFrame(np.random.normal(loc=10, scale=5, size=30))

    # Save the sample mean
    meanlist.append(sample[0].mean())

    # Save the sample variance
    varlist.append(sample[0].var(ddof=1))

# Store results in a DataFrame
collection = pd.DataFrame()
collection["meanlist"] = meanlist
collection["varlist"] = varlist

print(collection.head())


# Plot histogram of sample variances
collection["varlist"].hist(bins=50, color='red', label='Sample variance')
plt.title("Histogram of sample variances")
plt.show()

# Plot histogram of sample means
collection["meanlist"].hist(bins=50, color='cyan')
plt.title("Histogram of sample means")
plt.show()
