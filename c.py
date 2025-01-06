import matplotlib.pyplot as plt
import numpy as np
import math
import collections
import random


def g(x, t):
    return 1 / math.sqrt(4 * math.pi * D * t) * math.exp(-x**2 / (4 * D * t))


def w(x, t):
    return g(x - x0, t - t0) * g(x1 - x, t1 - t) / g(x1 - x0, t1 - t0)


random.seed(123)
D = 1 / 3
p0 = collections.Counter()
p1 = collections.Counter()
x0 = 0
t0 = 0

x1 = 10
t1 = 20
t = t1 // 2
N = 1000
i = 0
while True:
    x = x0
    for j in range(t0, t):
        x += random.randint(-1, 1)
    y = x
    for j in range(t, t1):
        x += random.randint(-1, 1)
    p0[x] += 1
    if x == x1:
        p1[y] += 1
        i += 1
        if i == N:
            break

x = range(-3 * x1 // 2, 3 * x1 // 2 + 1)
plt.step(x, [p0[x] / sum(p0.values()) for x in x], where='mid')
plt.plot(x, [g(x, t1) for x in x])

plt.step(x, [p1[x] / sum(p1.values()) for x in x], where='mid')
plt.plot(x, [w(x, t) for x in x])

plt.savefig("c.png")
