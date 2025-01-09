import matplotlib.pyplot as plt
import math
import numpy as np


def g(x, t):
    return 1 / math.sqrt(4 * math.pi * D * t) * math.exp(-x**2 / (4 * D * t))


def w(x, t):
    return g(x, t) * g(x1 - x, t1 - t) / g(x1, t1)


plt.rcParams.update({'lines.linewidth': 3})
plt.rcParams.update({'lines.markersize': 12})
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'figure.figsize': (12, 6)})
D = 1 / 3
x1 = 10
t1 = 20
x = np.linspace(-3 * x1 // 2, 3 * x1 // 2 + 1, 200)
plt.axis((-3 * x1 // 2, 3 * x1 // 2, None, 0.75))
plt.xlabel("position")

plt.plot(x, [w(x, 0.001 * t1) for x in x], 'r')
plt.plot(x, [w(x, 0.999 * t1) for x in x], 'r')
for t in range(1, t1 - 1, 4):
    plt.plot(x, [w(x, t) for x in x], 'r')
plt.savefig("one.bridge.pdf", bbox_inches="tight")
