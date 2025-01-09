import matplotlib.pyplot as plt
import random

random.seed(12345)
plt.rcParams.update({'lines.linewidth': 3})
plt.rcParams.update({'lines.markersize': 45})
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'figure.figsize': (12, 6)})

x1 = 4
lo = -3 * x1 // 2
hi = 3 * x1 // 2
plt.xlabel("position")
plt.axis((lo, hi, 0, 4))
plt.gca().set_xticks([x + 1 / 2 for x in range(lo, hi)])
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])

plt.grid(axis='x')

for x, y in (-2, 1), (-2, 2), (-3, 1), (-3, 2), (-4, 1):
    plt.plot([x], [y], "ok")

for x, y in (4, 1), (4, 2), (5, 1), (5, 2), (5, 3):
    plt.plot([x], [y], "or")

for x, y in (0, 1), (0, 2), (1, 1), (1, 2), (2, 1):
    plt.plot([x], [y], "o", markerfacecolor='none', markeredgecolor='red')

plt.savefig("particles.pdf", bbox_inches="tight")
