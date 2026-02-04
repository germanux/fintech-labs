import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ms = pd.read_csv("../data/microsoft.csv")
ms["Date"] = pd.to_datetime(ms["Date"])
ms = ms.sort_values("Date").set_index("Date")

ms["Close"] = pd.to_numeric(ms["Close"], errors="coerce")
ms = ms.dropna(subset=["Close"])

# daily log return
ms["LogReturn"] = np.log(ms["Close"]).diff()
ms = ms.dropna(subset=["LogReturn"])

# --- 6-month blocks: 6MS = starts in Jan and Jul by default ---
g = ms.groupby(pd.Grouper(freq="6MS"))["LogReturn"]

# step series aligned to every day in that block
ms["mu_6m"] = g.transform("mean")
ms["sigma_6m"] = g.transform("std")   # sample std (ddof=1)
ms["var_6m"] = g.transform("var")     # sample variance (ddof=1)

# Plot
fig, (ax_price, ax_stats) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax_price.plot(ms.index, ms["Close"])
ax_price.set_title("Microsoft Close price")
ax_price.set_ylabel("Price")

ax_stats.plot(ms.index, ms["mu_6m"], label="μ (mean log-return, 6M block)")
ax_stats.plot(ms.index, ms["sigma_6m"], label="σ (std log-return, 6M block)")
ax_stats.axhline(0, linewidth=1)
ax_stats.set_ylabel("Return (log)")
ax_stats.legend(loc="upper left")

ax_var = ax_stats.twinx()
ax_var.plot(ms.index, ms["var_6m"], linestyle="--", label="σ² (variance, 6M block)")
ax_var.set_ylabel("Variance")
ax_var.legend(loc="upper right")

plt.tight_layout()
plt.show()
