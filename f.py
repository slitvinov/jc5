import numpy as np
import matplotlib.pyplot as plt
import statistics
import scipy


def rho(t):
    I = np.identity(len(Si))
    D = scipy.linalg.sqrtm(
        4 * scipy.linalg.sqrtm(Si) @ Sf @ scipy.linalg.sqrtm(Si) - eps**2 * I)
    C = 1 / 2 * (scipy.linalg.sqrtm(Si) @ D *
                 np.linalg.inv(scipy.linalg.sqrtm(Si)) - eps * I)
    Ct = C
    St = (1 - t)**2 * Si + t**2 * Sf + (1 - t) * t * (C + Ct + eps * I)
    mut = (1 - t) * mui + t * muf
    return scipy.stats.multivariate_normal(mut, St)


eps = 1
mui = np.array([-2, -2], dtype=float)
muf = np.array([5, 5], dtype=float)
Si = np.array([[4, 0], [0, 1]], dtype=float)
Sf = np.array([[2, 8], [2, 4]], dtype=float)

rhoi = scipy.stats.multivariate_normal(mui, Si)
rhof = scipy.stats.multivariate_normal(muf, Sf)
rhot = rho(1 / 2)

x = np.linspace(-10, 10, 200)
x, y = np.meshgrid(x, x)
pos = np.dstack((x, y))

zi = rhoi.pdf(pos)
zf = rhof.pdf(pos)
zt = rhot.pdf(pos)

plt.axis("equal")
plt.contour(x, y, zi)
plt.contour(x, y, zf)
plt.contour(x, y, zt)

plt.savefig("f.png")
