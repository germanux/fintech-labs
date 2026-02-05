import numpy as np
import matplotlib.pyplot as plt

# 3 poblaciones
A = np.array([1,3,5,7,9])
B = np.array([1,2,5,8,9])
C = np.array([1,4,5,7,8])
D = np.array([3,4,5,6,7])
pops = [A, B, C, D]
names = ["A: 1,3,5,7,9", "B: 1,2,5,8,9", "C: 1,4,5,7,8", "D: 3,4,5,6,7"]

Y_MAX = 0.3

def pdf(x, mu, sigma):
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-((x-mu)**2)/(2*sigma**2))

fig, axes = plt.subplots(4, 1, sharex=True)

for ax, data, name in zip(axes, pops, names):
    mu = data.mean()
    sigma = np.sqrt(((data - mu)**2).mean())  # var poblacional (divide entre N)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 400)
    ax.plot(x, pdf(x, mu, sigma))
    ax.set_title(f"{name} | mu={mu:.1f}, sigma={sigma:.3f}")
    ax.set_ylim(0, Y_MAX)         # ylim: rango mostrado en el eje Y (densidad siempre >= 0)

plt.show()
