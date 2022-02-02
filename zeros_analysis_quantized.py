import matplotlib.pylab as plt
import numpy as np
from antenna_array import AntennaArray


n = 11
antenna = AntennaArray(n, n, assamble_L=True)
angles = [20]
mag = []
phase = []
mag_z = []
phase_z = []
mag_f = []
phase_f = []
mag_w = []
phase_w = []


plt.figure('zeros')
for angle in angles:
    excitation_z = antenna.dolph(theta=angle, phi=0, R=30)
    # excitation_z = antenna.schelkunoff_zero_placement(theta=angle, phi=0)

    mag_z.append(np.abs(excitation_z[1, 5]))
    phase_z.append(np.angle(excitation_z[1, 5]))
    poly_z = excitation_z[:, 6]
    zeros_z = np.roots(poly_z)
    plt.plot(zeros_z.real, zeros_z.imag, 'yo', markersize=12, label='original')

    mag_z.append(np.abs(excitation_z[1, 5]))
    phase_z.append(np.angle(excitation_z[1, 5]))
    excitation_z = antenna.quantize(excitation_z, 3)
    poly_z = excitation_z[:, 6]
    zeros_z = np.roots(poly_z)
    plt.plot(zeros_z.real, zeros_z.imag, 'kx', markersize=12, label='quantized')

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Re', fontsize=16)
plt.ylabel('Im', fontsize=16)
plt.axis('equal')
plt.legend()
plt.grid()
plt.tight_layout()
plt.tight_layout()
plt.savefig('./results/zeros_comparing_quantized_30db.pdf')
plt.show()




