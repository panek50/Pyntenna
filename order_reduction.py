import numpy as np
from antenna_array import AntennaArray
import pylab as plt
n = 11
angle = 0
antenna = AntennaArray(n, n, assamble_L=True, n_phi=100, n_theta=100)
excitation = antenna.dolph(theta=angle, phi=0, R=20)
af = antenna.calculate_matrix(excitation)
antenna.plot_2d_elevation(normalize=True, label='Dolph-Chebychev method', markers='-')
af_mag = np.abs(af)
cor = antenna.L.T @ antenna.L
cor = cor / np.max(np.abs(cor))
val, vec = np.linalg.eig(cor)
ex = np.linalg.pinv(antenna.L) @ antenna.data.reshape([antenna.n_theta * antenna.n_phi, 1])
ex = ex / np.max(np.abs(ex))
af = antenna.calculate_matrix(ex)
antenna.plot_2d_elevation(normalize=True, label='Penrose-Moore pseudo-inverse', markers='kx')
plt.savefig('comparison_pinv_cheby.pdf')
plt.legend()
plt.show()
#
plt.plot(np.abs(val), 'x')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(r'$i$[-]', fontsize=16)
plt.ylabel(r'$|\lambda_i|$', fontsize=16)
plt.grid()
plt.savefig('eigen_values.pdf')
plt.show()




