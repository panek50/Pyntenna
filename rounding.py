import matplotlib.pylab as plt
import numpy as np
from antenna_array import AntennaArray
import numpy as np
import pylab as plt


n = 11
n_bit = 2
angle = 20
antenna = AntennaArray(n, n, assamble_L=False)
excitation = antenna.dolph(theta=angle, phi=0, R=30)
colours_dot = ['r.', 'b.', 'k.', 'g.' ]
colours = ['k', 'r', 'b', 'k', 'g' ]
F_zeros = []
F_af = []
F_poly = []


for i in range(20):
    poly = excitation[:, 5]
    poly = poly / np.max(poly)
    quantized = antenna.quantize(excitation, n_bit=n_bit)

    if i == 0:
        pass
    else:
        quantized += ((np.random.randn(n) > 0.5) - 0.5 ) * 2 / 2**n_bit

    poly_q = quantized[:, 5]

    L = np.roots(poly)
    L = np.array(sorted(L, key= lambda x: np.angle(x)))
    L_q = np.roots(poly_q)
    L_q = np.array(sorted(L_q, key=lambda x: np.angle(x)))
    F_zeros.append(np.max(np.abs(np.angle(L) - np.angle(L_q))))
    F_poly.append(np.sum(np.abs(excitation - quantized)))
    plt.figure(1)
    plt.grid('on')
    plt.axis('equal')
    for value in L:
        plt.plot(L.real, L.imag, 'kx')
        plt.plot(L_q.real, L_q.imag, colours_dot[i % 4])

    plt.figure(2)
    af_q = antenna.calculate(quantized)
    if i == 0:
        antenna.plot_2d_elevation(markers='kx--', normalize=True)
    else:
        antenna.plot_2d_elevation(normalize=True)

    af = antenna.calculate(excitation)
    sum = 0
    index = antenna.phi_index(0)
    maxima_indices = antenna.detect_local_maxima(np.abs(af[:, index]))
    for j in maxima_indices[0]:
        sum += np.abs(np.abs(af[j, index]) - np.abs(af_q[j, index]))
    F_af.append(sum)
    print(sum)
    # antenna.plot_2d_elevation(markers=colours[i % 4], normalize=True)

# fig, ax1 = plt.subplots()
# print(F_af)
#
# ax1.plot(F_af, 'r-')
# ax2 = ax1.twinx()
# ax2.plot(F_zeros, 'b-')
# # ax2.plot(F_poly, 'bx')
plt.grid()
plt.show()