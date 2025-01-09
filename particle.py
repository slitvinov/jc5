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
plt.plot([0], [2], "ok")
for x, y in (0, 1), (-1, 1), (1, 1):
    plt.plot([x], [y], "o", markerfacecolor='none', markeredgecolor='black')
plt.savefig("particle.pdf", bbox_inches="tight")
