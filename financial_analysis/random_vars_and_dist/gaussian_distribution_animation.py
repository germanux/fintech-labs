import numpy as np

# Fuerza un backend "interactivo" para que la ventana de Matplotlib se muestre bien
# (TkAgg suele funcionar en muchos entornos; Qt5Agg si tienes Qt instalado).
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# =========================
# CONFIGURACIÓN (toca aquí)
# =========================

# Dominio del eje X (rango de valores donde evaluamos la normal)
X_MIN, X_MAX = -10, 10

# Cuántos puntos (resolución) para dibujar la curva
N_POINTS = 800

# Rango de mu (media) que quieres animar: de MU_START a MU_END
MU_START, MU_END = -1.0, -1.0

# Rango de sigma (desviación típica) que quieres animar: de SIGMA_START a SIGMA_END
# Si no quieres animar sigma, pon SIGMA_START == SIGMA_END (ej: 1.0 y 1.0)
SIGMA_START, SIGMA_END = 0.001, 5.0

# Animación: número de frames y tiempo entre frames (ms)
FRAMES = 1000
INTERVAL_MS = 20

# Estética del gráfico
LINE_WIDTH = 2  # equivale a lw=2
Y_MAX = 2    # límite superior del eje Y (ajústalo si animas sigma mucho)


# =========================
# DATOS BASE
# =========================

# linspace devuelve un numpy.ndarray (array) con N_POINTS valores equiespaciados
x = np.linspace(X_MIN, X_MAX, N_POINTS)


# =========================
# FIGURA Y EJES
# =========================

# plt.subplots() devuelve una tupla: (fig, ax)
# - fig: el objeto Figure (la "ventana"/lienzo completo)
# - ax: el objeto Axes (los ejes donde dibujas la curva)
# La asignación "fig, ax = ..." es asignación múltiple (unpacking) en Python.
fig, ax = plt.subplots()

# ax.plot(...) devuelve una lista de objetos Line2D (aunque sea una sola línea).
# "line, = ..." (con coma) hace unpacking del primer (y único) elemento de esa lista.
# lw=2 significa line width = grosor de la línea.
line, = ax.plot([], [], lw=LINE_WIDTH)

# Límites visibles de los ejes
ax.set_xlim(X_MIN, X_MAX)     # xlim: rango mostrado en el eje X
ax.set_ylim(0, Y_MAX)         # ylim: rango mostrado en el eje Y (densidad siempre >= 0)


# =========================
# MODELO: PDF Normal
# =========================

def pdf(x, mu, sigma):
    """PDF (densidad) de la Normal N(mu, sigma^2).
    - x puede ser escalar o array (NumPy vectoriza).
    - Devuelve un array si x es array.
    """
    return (1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-((x - mu)**2) / (2 * sigma**2))


# =========================
# UTIL: interpolación lineal de parámetros
# =========================

def lerp(a, b, t):
    """Linear interpolation: a -> b, con t en [0,1]."""
    return a + (b - a) * t


# =========================
# ANIMACIÓN
# =========================

def init():
    """Se llama una vez al inicio para pintar algo."""
    mu0 = MU_START
    sigma0 = SIGMA_START
    y0 = pdf(x, mu0, sigma0)
    line.set_data(x, y0)
    ax.set_title(f"mu={mu0:.2f}, sigma={sigma0:.2f}")
    return line,  # Matplotlib espera un iterable de artistas

def update(frame):
    """Se llama en cada frame.
    - frame va de 0 a FRAMES-1
    """
    # Normalizamos frame a t en [0,1] para que el rango sea independiente de FRAMES
    t = frame / (FRAMES - 1) if FRAMES > 1 else 1.0

    # Ahora mu y sigma van de inicio a fin automáticamente
    mu = lerp(MU_START, MU_END, t)
    sigma = lerp(SIGMA_START, SIGMA_END, t)

    y = pdf(x, mu, sigma)
    line.set_data(x, y)

    # Si animas sigma mucho, el pico cambia (1/(sigma*sqrt(2pi))).
    # Si quieres auto-ajuste del eje Y, descomenta estas 2 líneas:
    # peak = 1 / (sigma * np.sqrt(2*np.pi))
    # ax.set_ylim(0, 1.2 * peak)

    ax.set_title(f"mu={mu:.2f}, sigma={sigma:.2f}")
    return line,

# FuncAnimation crea el "bucle de frames" llamando update(frame)
# - frames=FRAMES: cuántos frames
# - interval=INTERVAL_MS: ms entre frames
# - blit=False: más compatible (refresca todo el eje)
anim = FuncAnimation(fig, update, init_func=init, frames=FRAMES, interval=INTERVAL_MS, blit=False)

plt.show()
