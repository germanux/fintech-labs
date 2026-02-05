import numpy as np
import matplotlib
matplotlib.use("TkAgg")   # o "Qt5Agg" si tienes Qt instalado

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sigma = 1.0
x = np.linspace(-6, 6, 800)

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(-6, 6)
ax.set_ylim(0, 0.45)

def pdf(x, mu, sigma):
    """Defines Gaussian probability density function"""
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-((x-mu)**2)/(2*sigma**2))

def init():
    line.set_data(x, pdf(x, -3, sigma))  # pinta algo inicial
    ax.set_title("mu=-3.00, sigma=1.0")
    return line,

def update(frame):
    mu = -3 + frame * 0.05
    y = pdf(x, mu, sigma)
    line.set_data(x, y)
    ax.set_title(f"mu={mu:.2f}, sigma={sigma}")
    return line,

anim = FuncAnimation(fig, update, init_func=init, frames=140, interval=30, blit=False)
plt.show()
