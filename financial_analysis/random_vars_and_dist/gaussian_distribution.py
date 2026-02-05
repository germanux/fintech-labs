import numpy as np
import matplotlib.pyplot as plt

# Parámetros de la normal:
# mu = media (centro de la campana)
# sigma = desviación típica (ancho de la campana)
mu, sigma = 0.0, 1.0

# linspace(a, b, n) crea un array de n puntos igualmente espaciados entre a y b (incluye extremos)
# Aquí generamos valores de x desde mu-4σ hasta mu+4σ (cubre casi toda la campana)
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)

# Esto NO guarda una "función" como en JS.
# f es un array (misma longitud que x) con los valores de la fórmula evaluada en cada x.
# Es decir: f[i] = densidad normal en x[i]
f = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-((x-mu)**2)/(2*sigma**2))

# plot(x, f): dibuja una línea uniendo los puntos (x[i], f[i]) para i=0..n-1
plt.plot(x, f)
plt.show()
