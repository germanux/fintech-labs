import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# -----------------------------
# 1) Load data
# -----------------------------
ms = pd.read_csv('../data/microsoft.csv')  # Load Microsoft's historical prices
print(ms.head())                          # In a .py script you must print to see it

# Ensure the Close column is numeric (sometimes CSVs load as strings)
ms['Close'] = pd.to_numeric(ms['Close'], errors='coerce')

# -----------------------------
# 2) Compute returns
# -----------------------------
# Simple return (aka arithmetic return): r_t = (P_{t+1} - P_t) / P_t
ms['SimpleReturn'] = ms['Close'].shift(-1) / ms['Close'] - 1.0
# ms['SimpleReturn'] = ms['SimpleReturn'].sort_values().to_numpy()

# Log return (aka continuously-compounded return): ln(P_{t+1}/P_t)
ms['LogReturn'] = np.log(ms['Close'].shift(-1)) - np.log(ms['Close'])
# ms['LogReturn'] = ms['LogReturn'].sort_values().to_numpy()


# Drop last row because shift(-1) creates NaN at the end
ms = ms.dropna(subset=['SimpleReturn', 'LogReturn']).copy()

# -----------------------------
# 3) Fit a Normal distribution to LogReturn
# -----------------------------
# mu: sample mean of daily log returns (average daily log return)
mu = ms['LogReturn'].mean()

# mur: sample mean of daily returns (average daily return)
mur = ms['SimpleReturn'].mean()

# sigma: sample standard deviation of daily log returns (volatility)
# ddof=1 => divide by (n-1), i.e. sample std
sigma = ms['LogReturn'].std(ddof=1)

print(f"mu (mean log return, daily) = {mu:.6f}")
print(f"mur (mean return, daily) = {mur:.6f}")
print(f"sigma (std log return, daily) = {sigma:.6f}")

# Create an x-grid to draw the fitted Normal PDF/CDF
x_min = ms['LogReturn'].min() - 0.01
x_max = ms['LogReturn'].max() + 0.01
x = np.arange(x_min, x_max, 0.001)

pdf = norm.pdf(x, loc=mu, scale=sigma)  # Probability Density Function (height, not probability)
cdf = norm.cdf(x, loc=mu, scale=sigma)  # Cumulative Distribution Function P(X <= x)

# -----------------------------
# 4) Plots (more explanatory)
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# (A) Price series
axes[0, 0].plot(ms['Close'].values)
axes[0, 0].set_title("Close price (time series)")
axes[0, 0].set_xlabel("Day index")
axes[0, 0].set_ylabel("Price")

# (C) Histogram of LogReturn + fitted Normal PDF overlay
axes[0, 1].hist(ms['LogReturn'].values, bins=50, density=True, alpha=0.7)
axes[0, 1].plot(x, pdf, linewidth=2)  # Fitted Normal PDF
axes[0, 1].set_title("LogReturn histogram + fitted Normal PDF")
axes[0, 1].set_xlabel("LogReturn")
axes[0, 1].set_ylabel("Density")

# (B) Compare SimpleReturn vs LogReturn (time series, zoomed)
axes[1, 0].plot(ms['SimpleReturn'].values, label="SimpleReturn")
axes[1, 0].plot(ms['LogReturn'].values, label="LogReturn", alpha=0.8)
axes[1, 0].set_title("Daily returns (Simple vs Log)")
axes[1, 0].set_xlabel("Day index")
axes[1, 0].set_ylabel("Return")
axes[1, 0].legend()

# (D) CDF of the fitted Normal (helps interpret probabilities)
axes[1, 1].plot(x, cdf, linewidth=2)
axes[1, 1].set_title("Fitted Normal CDF  (P(X <= x))")
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("Cumulative probability")

plt.tight_layout()
plt.show()
