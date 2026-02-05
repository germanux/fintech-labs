import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
rolls = [1, 2, 3, 5, 10, 15, 100]
trials = 50000  # simulaciones por caso

for n in rolls:
    s = np.random.randint(1, 7, size=(trials, n)).sum(axis=1)
    plt.figure()
    plt.hist(s, bins="auto", density=True)
    plt.title(f"Suma de {n} dado(s) (trials={trials})")
    plt.xlabel("Suma")
    plt.ylabel("Densidad")
    plt.show()
