import matplotlib.pylab as plt
import numpy as np
from antenna_array import AntennaArray
from matplotlib import cm

n = 11
angle = 0

antenna = AntennaArray(n, n, assamble_L=False)
antenna.create_template(0, 0)
antenna.crete_window(n)
excitation = antenna.fourier_series()
poly = np.ones(n)
zeros = np.roots(poly)
print(np.angle(zeros))
zeros = sorted(zeros, key=lambda x: np.abs(np.angle(x)))
deg = np.pi/180 * 1
k = 0
d = [5, -2, -2.5, -1.5, -0.75]
for i in range((n-1)//2):
    zeros[k] = zeros[k] * np.exp(1j*deg * d[i])
    zeros[k+1] = zeros[k+1] * np.exp(-1j*deg * d[i])
    k += 2
excitation = antenna.excitation_from_zeros(zeros)
print(np.angle(zeros))
antenna.calculate(excitation)
antenna.zeros = zeros
#antenna.plot_zeros()
# plt.plot(d)
plt.show()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Re', fontsize=16)
plt.ylabel('Im', fontsize=16)
plt.axis('equal')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('./results/zeros_comparison.pdf')
plt.show()

antenna.plot_2d_elevation(normalize=True)
plt.show()



