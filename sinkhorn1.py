import numpy as np
import matplotlib.pyplot as plt
import statistics
import math


def w0(x):
    x += 150
    x /= 120
    return \
        statistics.NormalDist(0.3 , 0.5).pdf(x) + \
        0.5 * statistics.NormalDist(0.6 , 0.15).pdf(x)


def w1(x):
    x += 60
    x /= 300
    return \
        .5*statistics.NormalDist(0.88,2*0.024).pdf(x) + \
        .8*statistics.NormalDist(0.70,4*0.030).pdf(x) + \
        .5*statistics.NormalDist(0.52,2*0.024).pdf(x)


def gf(x, t):
    return 1 / math.sqrt(4 * math.pi * D * t) * math.exp(-x**2 / (4 * D * t))

def dist(t):
    psi0 = [sum(gf(x - x0, t) * f for f, x0 in zip(psi, xx)) for x in xx]
    phi0 = [sum(gf(x - x0, t1 - t) * f for f, x0 in zip(phi, xx)) for x in xx]
    w = np.multiply(phi0, psi0)
    return w / sum(w)

plt.rcParams.update({'lines.linewidth': 3})
plt.rcParams.update({'lines.markersize': 12})
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'figure.figsize': (12, 6)})
lo = -300
hi = 300
t1 = 160
D = 1 / 3

xx = range(lo, hi + 1)
g = [[gf(l - k, t1) for l in xx] for k in xx]
a = [w0(x) for x in xx]
b = [w1(x) for x in xx]
psi = [1 for x in xx]
phi = [1 for x in xx]
for step in range(100):
    psi, phi = np.divide(a, np.dot(g, phi)), np.divide(b, np.dot(g, psi))

w00 = np.array([w0(x) for x in xx])
w11 = np.array([w1(x) for x in xx])
plt.plot(xx, w00 / sum(w00), color=[0, 0, 0])
plt.plot(xx, w11 / sum(w11), color=[1, 0, 0])
for t in 0.3, 0.8:
    plt.plot(xx, dist(t * t1), color=[t, 0, 0])
plt.savefig("sinkhorn1.pdf", bbox_inches="tight")
