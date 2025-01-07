import numpy as np
import matplotlib.pyplot as plt
import statistics


def rho(t):
    I = 1
    D = (4 * Si**0.5 * Sf * Si**0.5 - eps**2 * I)**0.5
    C = 1 / 2 * (Si**0.5 * D * Si**(-0.5) - eps * I)
    Ct = C
    St = (1 - t)**2 * Si + t**2 * Sf + (1 - t) * t * (C + Ct + eps * I)
    mut = (1 - t) * mui + t * muf
    return statistics.NormalDist(mut, St)


mui = -2
muf = 5
Si = 1
Sf = 2
eps = 1

x = np.linspace(-10, 10, 200)
rhoi = statistics.NormalDist(mui, Si)
rhof = statistics.NormalDist(muf, Sf)
rhot = rho(1 / 2)
yi = [rhoi.pdf(x) for x in x]
yf = [rhof.pdf(x) for x in x]
yt = [rhot.pdf(x) for x in x]

plt.plot(x, yi)
plt.plot(x, yf)
plt.plot(x, yt)

plt.show()
