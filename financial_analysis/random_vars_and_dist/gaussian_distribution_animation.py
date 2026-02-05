import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sigma = 1.0
x = np.linspace(-6, 6, 800)

fig, ax = plt.subplots()
(line,) = ax.plot([], [])
ax.set_xlim(x.min(), x.max())
ax.set_ylim(0, 0.45)

def pdf(x, mu, sigma):
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-((x-mu)**2)/(2*sigma**2))

def update(frame):
    mu = -3 + frame*0.05
    y = pdf(x, mu, sigma)
    line.set_data(x, y)
    ax.set_title(f"mu={mu:.2f}, sigma={sigma}")
    return line,

FuncAnimation(fig, update, frames=140, interval=300, blit=True)
plt.show()
