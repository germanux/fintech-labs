import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Leer el CSV ---
ms = pd.read_csv("../data/microsoft.csv")          # Carga el fichero con datos históricos
ms["Date"] = pd.to_datetime(ms["Date"])           # Convierte la columna Date a tipo fecha
ms = ms.sort_values("Date").set_index("Date")     # Ordena por fecha y pone Date como índice (eje X)

# --- Limpiar precios ---
ms["Close"] = pd.to_numeric(ms["Close"], errors="coerce")  # Asegura que Close es numérico (si algo falla, NaN)
ms = ms.dropna(subset=["Close"])                           # Elimina filas sin precio

# --- Calcular log-returns diarios ---
# LogReturn_t = ln(P_t) - ln(P_{t-1})  == ln(P_t / P_{t-1})
ms["LogReturn"] = np.log(ms["Close"]).diff()     # diff() resta con el día anterior; el primer día queda NaN
ms = ms.dropna(subset=["LogReturn"])             # Quitamos el primer NaN (y cualquier otro)

# --- Agrupar en bloques de 6 meses (enero-junio, julio-diciembre) ---
# pd.Grouper(freq="6MS") = grupos que empiezan cada 6 meses (Month Start): Jan 1, Jul 1, Jan 1, ...
g = ms.groupby(pd.Grouper(freq="6MS"))["LogReturn"]  # g es un “grupo” de log-returns por semestres

# --- Calcular estadísticas por bloque, pero alineadas a cada día del bloque ---
# transform("mean") pone a cada fila el valor medio de SU bloque
ms["mu_6m"] = g.transform("mean")       # μ = media de log-return en ese semestre (valor constante por bloque)
ms["sigma_6m"] = g.transform("std")     # σ = desviación típica en ese semestre (volatilidad diaria)
ms["var_6m"] = g.transform("var")       # σ² = varianza en ese semestre (volatilidad al cuadrado)

# --- Dibujar gráficos ---
fig, (ax_price, ax_stats) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)  # 2 filas, mismo eje X (fecha)

# Gráfico 1: precio
ax_price.plot(ms.index, ms["Close"])       # Serie temporal del precio de cierre
ax_price.set_title("Microsoft Close price")
ax_price.set_ylabel("Price")

# Gráfico 2: métricas por semestres (eje izquierdo)
ax_stats.plot(ms.index, ms["mu_6m"], label="μ (media log-return, bloque 6M)")
ax_stats.plot(ms.index, ms["sigma_6m"], label="σ (desv típica log-return, bloque 6M)")
ax_stats.axhline(0, linewidth=1)           # Línea horizontal en 0 para ver si μ está por encima o por debajo
ax_stats.set_ylabel("Return (log)")
ax_stats.legend(loc="center left")

# Eje derecho: varianza (porque numéricamente es mucho más pequeña que σ)
ax_var = ax_stats.twinx()                  # Segundo eje Y a la derecha (misma X)
ax_var.plot(ms.index, ms["var_6m"], linestyle="--", label="σ² (varianza, bloque 6M)")
ax_var.set_ylabel("Variance")
ax_var.legend(loc="upper right")

plt.tight_layout()
plt.show()
