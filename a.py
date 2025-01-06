import matplotlib.pyplot as plt
import math
import numpy as np


def g(x, t):
    return 1 / math.sqrt(4 * math.pi * D * t) * math.exp(-x**2 / (4 * D * t))


def w(x, t):
    return g(x, t) * g(x1 - x, t1 - t) / g(x1, t1)


D = 1 / 3

x1 = 10
t1 = 20
t = t1 / 2

x = np.linspace(-1.5 * x1, 1.5 * x1 + 1, 100)
p0 = [g(x, t1) for x in x]
p1 = [w(x, t) for x in x]
plt.plot(x, p0, x, p1)
plt.savefig("a.png")
