import numpy as np
import matplotlib.pyplot as plt

mu, sigma = 0.0, 1.0
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 12)
f = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-((x-mu)**2)/(2*sigma**2))

plt.plot(x, f)
plt.show()
