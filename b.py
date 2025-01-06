import matplotlib.pyplot as plt
import numpy as np
import math
import collections
import random


def g(x, t):
    return 1 / math.sqrt(4 * math.pi * D * t) * math.exp(-x**2 / (4 * D * t))


random.seed(123)
D = 1 / 3
p0 = collections.Counter()
x0 = 0
t0 = 0

x1 = 10
t1 = 20
N = 100000
for i in range(N):
    x = x0
    for j in range(t0, t1):
        x += random.randint(-1, 1)
    p0[x] += 1
x = range(-2 * x1, 2 * x1 + 1)

plt.step(x, [p0[x] / N for x in x], where='mid')
plt.plot(x, [g(x, t1) for x in x])
plt.savefig("b.png")
