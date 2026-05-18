import pandas as pd
import numpy as np

# Create one random sample of 5 values from a Normal distribution
# mean = 10, standard deviation = 5
sample = pd.DataFrame(np.random.normal(loc=10, scale=5, size=5))

print("Complete sample:")
print(sample)
print()

# sample[0] means: take the whole first column, not the first element
print("Column 0:")
print(sample[0])
print()

# Compute sample mean
sample_mean = sample[0].mean()
print("Sample mean:", sample_mean)

# Compute sample variance (sample variance uses n-1, so ddof=1)
sample_var = sample[0].var(ddof=1)
print("Sample variance:", sample_var)

# Compute sample standard deviation
sample_std = sample[0].std(ddof=1)
print("Sample standard deviation:", sample_std)

