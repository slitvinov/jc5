import matplotlib.pyplot as plt
import math


def g(x, t):
    return 1 / math.sqrt(4 * math.pi * D * t) * math.exp(-x**2 / (4 * D * t))


def w(x, t):
    return g(x - x0, t - t0) * g(x1 - x, t1 - t) / g(x1 - x0, t1 - t0)


D = 1 / 3
x0 = 0
t0 = 0

x1 = 10
t1 = 20
t = t1 / 2

x = range(-2 * x1, 2 * x1 + 1)
p0 = [g(x, t1) for x in x]
p1 = [w(x, t) for x in x]
plt.plot(x, p0, x, p1)
plt.savefig("a.png")
