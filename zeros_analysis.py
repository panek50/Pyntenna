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
    excitation = antenna.dolph(theta=angle, phi=0, R=30)
    excitation_z = antenna.schelkunoff_zero_placement(theta=angle, phi=0)
    antenna.plot_zeros()
    antenna.create_template(angle, 0)
    antenna.crete_window(n, type=None)
    excitation_f = antenna.fourier_series()
    antenna.crete_window(n, type='Hamming')
    excitation_w = antenna.fourier_series()

    mag.append(np.abs(excitation[1, 5]))
    phase.append(np.angle(excitation[1, 5]))
    poly = excitation[:, 6]
    zeros = np.roots(poly)
    plt.plot(zeros.real, zeros.imag, 'o', markersize=12, label='Dolph Chebychev')

    mag_z.append(np.abs(excitation_z[1, 5]))
    phase_z.append(np.angle(excitation_z[1, 5]))
    excitation_z
    poly_z = excitation_z[:, 6]
    zeros_z = np.roots(poly_z)
    plt.plot(zeros_z.real, zeros_z.imag, 'sr', markersize=12, label='Schelunoff zero-placement')

    mag_f.append(np.abs(excitation_z[1, 5]))
    phase_f.append(np.angle(excitation_z[1, 5]))
    poly_f = excitation_f[:, 6]
    zeros_f = np.roots(poly_f)
    plt.plot(zeros_f.real, zeros_f.imag, 'vg', markersize=12, label='Woodward-Lawson')

    mag_w.append(np.abs(excitation_w[1, 5]))
    phase_w.append(np.angle(excitation_w[1, 5]))
    poly_w = excitation_w[:, 6]
    zeros_w = np.roots(poly_w)
    plt.plot(zeros_w.real, zeros_w.imag, '.k', markersize=15, label='Hamming window')

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Re', fontsize=16)
plt.ylabel('Im', fontsize=16)
plt.axis('equal')
plt.legend()
plt.grid()
plt.tight_layout()
plt.tight_layout()
plt.savefig('./results/zeros_comparison.pdf')
plt.show()




