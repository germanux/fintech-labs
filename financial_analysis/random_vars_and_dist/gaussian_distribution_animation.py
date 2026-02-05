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
X_MIN, X_MAX = -5, 5
Y_MAX = 2    # límite superior del eje Y (ajústalo si animas sigma mucho)

# Cuántos puntos (resolución) para dibujar la curva
N_POINTS = 800

# Rango de mu (media) que quieres animar: de MU_START a MU_END
MU_START, MU_END = -1.0, -1.0

# Rango de sigma (desviación típica) que quieres animar: de SIGMA_START a SIGMA_END
# Si no quieres animar sigma, pon SIGMA_START == SIGMA_END (ej: 1.0 y 1.0)
SIGMA_START, SIGMA_END = 0.01, 3.0

# Animación: número de frames y tiempo entre frames (ms)
FRAMES = 400
INTERVAL_MS = 30

# Estética del gráfico
LINE_WIDTH = 2  # equivale a lw=2


# Áreas teóricas (exactas para Normal)
AREA_1SIGMA = 0.682689492  # P(|Z|<=1)
AREA_2SIGMA = 0.954499736  # P(|Z|<=2)

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

# Mostrar varianza
txt_var = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")


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
# Elementos extra: líneas y áreas
# =========================

# Líneas verticales: mu, mu±sigma, mu±2sigma
v_mu   = ax.axvline(0, lw=1)
v_m1   = ax.axvline(0, lw=1, ls="--")
v_p1   = ax.axvline(0, lw=1, ls="--")
v_m2   = ax.axvline(0, lw=1, ls=":")
v_p2   = ax.axvline(0, lw=1, ls=":")

# Rellenos (guardamos el "handle" para poder borrarlo y redibujar cada frame)
fill_1 = None
fill_2 = None

# =========================
# ANIMACIÓN
# =========================

def init():
    """
    init() se ejecuta UNA vez al arrancar la animación.
    Su objetivo es dejar el gráfico en un estado inicial consistente:
    - la curva con datos (x, y)
    - las líneas verticales colocadas en mu, mu±sigma, mu±2sigma
    - los rellenos (áreas) dibujados para ±1σ y ±2σ
    """
    global fill_1, fill_2  # vamos a reasignar estos objetos (rellenos) aquí

    # Estado inicial: usa el inicio de los rangos configurados
    mu0 = MU_START
    sigma0 = SIGMA_START

    # Texto varianza
    txt_var.set_text(f"sigma={sigma0:.3f}   Var=sigma^2={sigma0**2:.3f}")

    # Calcula y = f(x) para la normal N(mu0, sigma0^2)
    y0 = pdf(x, mu0, sigma0)

    # Carga los datos en el objeto Line2D ya creado por ax.plot(...)
    # Esto es lo que realmente "dibuja" la curva.
    line.set_data(x, y0)

    # --- NUEVO: líneas de referencia (verticales) ---
    # set_xdata([a,a]) en una línea vertical hace que la línea se coloque en x=a
    v_mu.set_xdata([mu0, mu0])  # línea en el centro (mu)

    # líneas discontinuas en mu±sigma
    v_m1.set_xdata([mu0 - sigma0, mu0 - sigma0])
    v_p1.set_xdata([mu0 + sigma0, mu0 + sigma0])

    # líneas punteadas en mu±2sigma
    v_m2.set_xdata([mu0 - 2*sigma0, mu0 - 2*sigma0])
    v_p2.set_xdata([mu0 + 2*sigma0, mu0 + 2*sigma0])

    # --- NUEVO: rellenos (áreas) ---
    # En Matplotlib, fill_between devuelve un PolyCollection.
    # Para "animarlo" simple, lo habitual es borrarlo (remove) y recrearlo.
    if fill_1:
        fill_1.remove()
    if fill_2:
        fill_2.remove()

    # Calcula intervalos en x que corresponden a ±1σ y ±2σ
    m1, p1 = mu0 - sigma0, mu0 + sigma0
    m2, p2 = mu0 - 2*sigma0, mu0 + 2*sigma0

    # Máscaras booleanas para quedarnos solo con los puntos x dentro del intervalo
    mask1 = (x >= m1) & (x <= p1)
    mask2 = (x >= m2) & (x <= p2)

    # Rellena el área bajo la curva dentro de ±2σ y ±1σ.
    # Ojo: el alpha no es "probabilidad"; es solo transparencia visual.
    fill_2 = ax.fill_between(x[mask2], y0[mask2], alpha=0.15)  # ±2σ (≈0.9545)
    fill_1 = ax.fill_between(x[mask1], y0[mask1], alpha=0.30)  # ±1σ (≈0.6827)

    # Título inicial (informativo)
    ax.set_title(f"mu={mu0:.2f}, sigma={sigma0:.2f} | ±1σ≈0.683, ±2σ≈0.954")

    # IMPORTANTE: devolver los "artists" que cambian (especialmente si usaras blit=True).
    return line, v_mu, v_m1, v_p1, v_m2, v_p2, txt_var


def update(frame):
    """
    update(frame) se ejecuta en cada frame.
    'frame' es un entero: 0, 1, 2, ..., FRAMES-1

    Aquí actualizamos:
    - mu y sigma (interpolando de inicio a fin)
    - la curva line (set_data)
    - las líneas verticales
    - los rellenos del área
    """
    global fill_1, fill_2  # porque vamos a eliminar y recrear los rellenos

    # Convertimos frame a t en [0,1] para que el rango no dependa del número de frames
    t = frame / (FRAMES - 1) if FRAMES > 1 else 1.0

    # mu y sigma evolucionan linealmente en el tiempo (configurable con START/END)
    mu = lerp(MU_START, MU_END, t)
    sigma = lerp(SIGMA_START, SIGMA_END, t)

    # Texto varianza
    txt_var.set_text(f"sigma={sigma:.3f}   Var=sigma^2={sigma**2:.3f}")

    # Recalcula la curva
    y = pdf(x, mu, sigma)
    line.set_data(x, y)

    # Mueve líneas verticales
    v_mu.set_xdata([mu, mu])
    v_m1.set_xdata([mu - sigma, mu - sigma])
    v_p1.set_xdata([mu + sigma, mu + sigma])
    v_m2.set_xdata([mu - 2*sigma, mu - 2*sigma])
    v_p2.set_xdata([mu + 2*sigma, mu + 2*sigma])

    # Actualiza rellenos: borrar y recrear (simple y fiable)
    if fill_1:
        fill_1.remove()
    if fill_2:
        fill_2.remove()

    m1, p1 = mu - sigma, mu + sigma
    m2, p2 = mu - 2*sigma, mu + 2*sigma

    mask1 = (x >= m1) & (x <= p1)
    mask2 = (x >= m2) & (x <= p2)

    fill_2 = ax.fill_between(x[mask2], y[mask2], alpha=0.15)
    fill_1 = ax.fill_between(x[mask1], y[mask1], alpha=0.30)

    # Si sigma cambia mucho, el pico cambia mucho: pico = 1/(sigma*sqrt(2pi))
    # Esto ajusta el eje Y para que siempre se vea bien el "top" de la campana.
    # peak = 1 / (sigma * np.sqrt(2*np.pi))
    # ax.set_ylim(0, 1.2 * peak)

    ax.set_title(f"mu={mu:.2f}, sigma={sigma:.2f} | ±1σ≈0.683, ±2σ≈0.954")

    return line, v_mu, v_m1, v_p1, v_m2, v_p2, txt_var


# FuncAnimation crea el "bucle de frames" llamando update(frame)
# - frames=FRAMES: cuántos frames
# - interval=INTERVAL_MS: ms entre frames
# - blit=False: más compatible (refresca todo el eje)
anim = FuncAnimation(fig, update, init_func=init, frames=FRAMES, interval=INTERVAL_MS, blit=False)

plt.show()
