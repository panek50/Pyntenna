import matplotlib.pylab as plt
from antenna_array import AntennaArray
import numpy as np

n = 11
antenna = AntennaArray(n, n, assamble_L=True)
af = antenna.read_array_factor('./results/array_factor_NSGA_II.txt')
af_zeros = antenna.read_array_factor('./results/array_factor_EPS_MOEA.txt')
af_coefs = antenna.read_array_factor('./results/array_factor_SM_PSO.txt')
antenna.data = af
antenna.plot_2d_elevation(markers='bx-', label='NSGA II')
antenna.data = af_zeros
antenna.plot_2d_elevation(markers='k.-', label='EPS MOEA')
antenna.data = af_coefs
antenna.plot_2d_elevation(markers='r-', label='SMPSO')
plt.legend(prop={'size': 12})
plt.tight_layout()
plt.savefig('array_factor_NSGA_II_EPS_MOEA_SM_PSO_comparison.pdf')
plt.show()

# Plot convergence
file = open('./results/convergence_NSGA_II.txt', 'r')
strings = file.readlines()
x = []
for string in strings:
    x.append(float(string))
plt.plot(x, '-r', label='NSGA II')


file = open('./results/convergence_EPS_MOEA.txt', 'r')
strings = file.readlines()
x = []
for string in strings:
    x.append(float(string))
plt.plot(x, '-k', label='EPS MOEA')

file = open('./results/convergence_SM_PSO.txt', 'r')
strings = file.readlines()
x = []
for string in strings:
    x.append(float(string))
plt.plot(x, '--b', label='SM PSO')


plt.legend(prop={'size': 12})
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('$k [-]$', fontsize=16)
plt.ylabel(r'$S_a [\varphi, \vartheta]$', fontsize=16)
plt.grid()
plt.tight_layout()
plt.savefig('convergence_NSGA_II_EPS_MOEA_comparison.pdf')
plt.show()




