import numpy as np
import matplotlib.pyplot as plt
import statistics
import math


def w0(x):
    return statistics.NormalDist(0, 2).pdf(x)


def w1(x):
    return statistics.NormalDist(10, 2).pdf(x)


def gf(x, t):
    return 1 / math.sqrt(4 * math.pi * D * t) * math.exp(-x**2 / (4 * D * t))


x1 = 10
lo = -3 * x1 // 2
hi = 3 * x1 // 2
t1 = 20
D = 1 / 3

xx = range(lo, hi + 1)
g = [[gf(l - k, t1) for l in xx] for k in xx]
a = [w0(x) for x in xx]
b = [w1(x) for x in xx]

psi = [1 for x in xx]
phi = [1 for x in xx]

for step in range(20):
    psi, phi = np.divide(a, np.dot(g, phi)), np.divide(b, np.dot(g, psi))

t = 0.5 * t1
psi0 = [sum(gf(x - x0, t) * f for f, x0 in zip(psi, xx)) for x in xx]
phi0 = [sum(gf(x - x0, t1 - t) * f for f, x0 in zip(phi, xx)) for x in xx]
w = np.multiply(phi0, psi0)

plt.plot(xx, [w0(x) for x in xx], 'k-')
plt.plot(xx, [w1(x) for x in xx], 'r-')

plt.plot(xx, w / sum(w), color=[t / t1, 0, 0])

plt.savefig("sinkhorn.png", bbox_inches="tight")
