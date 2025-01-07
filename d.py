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
N = 5
i = 0
Bad = [ ]
Good = [ ]
while True:
    x = x0
    trace = [ x ]
    for j in range(t0, t):
        x += random.randint(-1, 1)
        trace.append(x)
    y = x
    for j in range(t, t1):
        x += random.randint(-1, 1)
        trace.append(x)
    p0[x] += 1
    if x == x1:
        Good.append((trace, y))
        p1[y] += 1
        i += 1
    else:
        Bad.append(trace)
    if i == N:
        break
    M += 1

plt.axis((-3 * x1 // 2, 3 * x1 // 2, None, None))    
for trace in Bad:
    plt.plot([x + random.uniform(-0.1, 0.1) for x in trace], range(t1 + 1), 'k-', alpha=0.1)

for trace, y in Good:
    plt.plot([x + random.uniform(-0.1, 0.1) for x in trace], range(t1 + 1), 'r-')
    print(y, t)
    plt.plot([y], [t], 'or')

plt.savefig("d.png")
