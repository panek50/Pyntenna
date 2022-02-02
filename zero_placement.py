import matplotlib.pylab as plt
import numpy as np
from antenna_array import AntennaArray

n = 11
antenna = AntennaArray(n, n, assamble_L=True)
# angles = np.linspace(0, 89, 30)
angles = [20]

diference = []
for i in range(0, 8):
    diference.append([])

for angle in angles:
    antenna.create_template(angle, 0)
    excitation = antenna.schelkunoff_zero_placement(theta=angle)
    # excitation = excitation / np.max(np.abs(excitation))
    index = antenna.phi_index(0)
    af = antenna.calculate_matrix(excitation)
    maxima_indices = antenna.detect_local_maxima(np.abs(af[:, index]))
    plt.figure(angle)
    antenna.plot_2d_elevation(normalize=True)
    for i in range(0, 8):
        maxima = []
        quantized = antenna.quantize(excitation, i)
        af = antenna.calculate_matrix(quantized)
        antenna.plot_2d_elevation(label='{} bits'.format(i + 2), normalize=True)
        for j in range(len(maxima_indices[0])):
            k = maxima_indices[0][j]
            maxima.append(np.abs(af[k, index]))
        maxima.sort(reverse=True)
        diference[i].append(maxima[0] / maxima[1])

    plt.legend()
    plt.tight_layout()
    plt.savefig('zero_placement.pdf')
    plt.show()

plt.figure('Diference')
for i in range(len(diference)):
    plt.plot(angles, 20 * np.log10(diference[i]), label="{0} bit".format(i))
    plt.xlabel(r'$\vartheta \ [^\circ]$', size=16)
    plt.ylabel(r'$\Delta$ [dB]', size=16)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('./results/zero_placement_steered.pdf')
plt.show()
