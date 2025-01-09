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
M = 0
N = 500
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
    M += 1
plt.rcParams.update({'lines.linewidth': 3})
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'figure.figsize': (12, 6)})

plt.axis((-3 * x1 // 2, 3 * x1 // 2, None, None))
plt.xlabel("position")
xd = range(-3 * x1 // 2, 3 * x1 // 2 + 1)
xc = np.linspace(-3 * x1 // 2, 3 * x1 // 2 + 1, 100)
plt.step(xd, [p0[x] / M for x in xd], color='k', where='mid')
plt.plot(xc, [g(x, t1) for x in xc], 'k')

plt.step(xd, [p1[x] / N for x in xd], 'r', where='mid')
plt.plot(xc, [w(x, t) for x in xc], 'r')

plt.savefig("one.dist.pdf", bbox_inches="tight")
