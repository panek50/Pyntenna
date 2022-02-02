import matplotlib.pylab as plt
import numpy as np
from scipy import signal
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
import time


class AntennaArray:

    def __init__(self, n_x, n_y, assamble_L=False, n_phi=200, n_theta=200):
        self.n_theta = n_phi  # division in elevation
        self.n_phi = n_theta  # division in azimuth
        self.zeros = None
        self.window2d = np.ones([self.n_theta, self.n_phi])
        self.c = 3e8  # speed of light in vacuum
        self.f = 3e9  # required frequency
        self.lambd = self.c / self.f  # wave length
        self.d_x = self.lambd / 2  # distance between patches in x direction
        self.d_y = self.lambd / 2
        self.k = 2 * np.pi / self.lambd  # wave number
        self.n_x = n_x  # number of patches in x direction
        self.n_y = n_y  # number of patches in y direction
        self.phi = np.linspace(-np.pi, np.pi, self.n_phi)  # azimuth
        self.theta = np.linspace(-np.pi / 2, np.pi / 2, self.n_theta)  # elevation
        self.cos_phi = np.cos(self.phi)
        self.sin_phi = np.sin(self.phi)
        self.cos_theta = np.cos(self.theta)
        self.sin_theta = np.sin(self.theta)
        self.delta_i = np.zeros([self.n_theta, self.n_phi])
        self.delta_j = np.zeros([self.n_theta, self.n_phi])

        # Matrices for array factor calculation
        start = time.time()
        for i in range(self.n_theta):
            for j in range(self.n_phi):
                self.delta_i[i, j] = (self.d_x * self.sin_theta[i] * self.cos_phi[j])
                self.delta_j[i, j] = (self.d_y * self.sin_theta[i] * self.sin_phi[j])
        end = time.time()
        print("Elapsed time, assambling delta matrices: {0}".format(end - start))

        if assamble_L:
            start = time.time()
            self.L = np.zeros([self.n_theta * self.n_phi, self.n_x**2], dtype=complex)
            for k in range(0, self.n_x*self.n_y):
                m = k % self.n_x
                n = k // self.n_y
                for i,theta in enumerate(self.theta):
                    for j, phi in enumerate(self.phi):
                       self.L[self.n_theta * i + j, k] = -1j * self.k * (m * self.delta_j[i, j] + n * self.delta_i[i, j])
            self.L = np.exp(self.L)
            end = time.time()
            print("Elapsed time, assambling L matrix: {0}".format(end - start))

        else:
            self.L = None

        self.data = None
        self.array_factor = None

    def plot_2d_azimuth(self, theta=0, label=None):
        plt.figure('')
        index = self.phi_index(0)
        # self.data = self.data / np.max(np.abs(self.data))
        plt.plot(self.theta / np.pi * 180, 20 * np.log10(np.abs(self.data[:, index])),
                 label=label)
        plt.ylim([-60, 1])
        plt.xlabel(r'$\theta [^\circ] $', size=16)
        plt.ylabel(r'AF [dB]', size=16)
        plt.grid('on')

    def plot_2d_elevation(self, phi=0, label=None, normalize=False, markers='-'):
        index = self.phi_index(phi)
        if normalize:
            data = self.data / np.max(np.abs(self.data[:, index]))
        else:
            data = self.data
        plt.plot(self.theta / np.pi * 180, 20 * np.log10(np.abs(data[:, index])), markers,
                 label=label)
        plt.ylim([-60, 1])
        plt.xlabel(r'$\vartheta [^\circ] $', size=16)
        plt.ylabel(r'$S_\mathrm{a}$ [dB]', size=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid('on', which='both')

    def plot_polar_elevation(self, phi=None):
        index = self.phi_index(phi)
        plt.polar(-self.theta + np.pi / 2,
                  (20 * np.log10(np.abs(self.data[:, index]))), 'r')
        # plt.ylim([-40, 0])
        plt.show()

    def plot_polar_azimuth(self, theta=0, normalize=True):
        if normalize:
            data = self.data / np.max(np.abs(self.data))
        index = self.theta_index(theta)
        plt.title('Phi')
        plt.polar(self.phi,
                  (20 * np.log10(np.abs(self.data[index, :]))), 'r')
        # plt.ylim([-40, 0])
        plt.show()

    def plot_excitation(self, excitation):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        x = np.linspace(0, self.n_x, self.n_x)
        y = np.linspace(0, self.n_y, self.n_y)
        X, Y = np.meshgrid(x, y)
        surf = ax.plot_surface(X, Y, np.abs(excitation))
        plt.show()

    def lin2db(self, value_lin):
        result = np.where(value_lin > 0.0000000001, value_lin, -10)
        np.log10(result, out=result, where=result > 0)
        return 20 * result

    def plot_3d(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Make data.
        phi_mesh = np.ones((self.n_theta, self.n_phi))
        theta_mesh = np.ones((self.n_theta, self.n_phi))

        for t, theta in enumerate(self.theta):
            for p, phi in enumerate(self.phi):
                theta_mesh[t, p] = theta
                phi_mesh[t, p] = phi

        r = np.abs(self.data) / np.max(np.abs(self.data))
        x = r * np.sin(theta_mesh) * np.cos(phi_mesh)
        y = r * np.sin(theta_mesh) * np.sin(phi_mesh)
        z = r * np.cos(theta_mesh)

        # Plot the surface.
        surf = ax.plot_surface(x, y, z)
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)
        ax.set_zlim(-0.2, 1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()

    def phi_index(self, angle):
        return int((angle + 180) * self.n_phi / 360)

    def theta_index(self, angle):
        return int((angle + 90) * self.n_theta / 180)

    def calculate_matrix(self, excitation):
        af = self.L @ excitation.reshape([self.n_x * self.n_y, 1])
        af = af * 1 / self.n_x / self.n_y
        self.data = af.reshape([self.n_theta, self.n_phi])
        return self.data

    def calculate(self, excitation):
        """ Calculates the array factor for given set of inputs """
        af = np.zeros([self.n_theta, self.n_phi], dtype=complex)
        for i in range(self.n_x):
            for j in range(self.n_y):
                af = af + np.exp(-1j * self.k * (i * self.delta_i + j * self.delta_j)) * \
                     excitation[i, j]
        af = af * 1 / self.n_x / self.n_y
        self.data = af
        return af

    def crete_window(self, n, type=None):
        if type is None:
            window1d = np.ones(n)
        if type == 'Hamming':
            window1d = np.abs(signal.windows.hamming(n))
        if type == 'Gaussian':
            window1d = np.abs(signal.windows.gaussian(n, 0))
        window1d = window1d / np.max(window1d)
        window2d = np.outer(window1d, window1d)
        self.window2d = window2d / np.max(window2d)

    def plot_window(self):
        X, Y = np.meshgrid(self.phi, self.theta)
        Z = self.window2d

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(X, Y, Z, 50, cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z');
        plt.show()

    # ToDo: Make 3D plot
    def plot_zeros(self):
        plt.figure('zeros')
        plt.title('Zeros')
        for zero in self.zeros:
            plt.plot(zero.real, zero.imag, 'x')
        plt.axis('equal')
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)


    def quantize(self, vector, n_bit):
        magnitude = np.abs(vector) / np.max(np.abs(vector))
        phase = np.angle(vector) / np.pi
        quantized_magnitude = np.round(magnitude * (2 ** n_bit)) / (2 ** n_bit)
        quantized_phase = np.round(phase * (2 ** n_bit)) / (2 ** n_bit) * np.pi
        quantized_vector = quantized_magnitude * np.exp(1j * quantized_phase)
        return quantized_vector

    def fourier_series(self):
        ex = np.zeros([self.n_x, self.n_y], dtype=complex)
        # self.data = self.data * self.window2d
        I = np.zeros([self.n_x, self.n_y])
        J = np.zeros([self.n_x, self.n_y])
        for i in range(self.n_x):
            for j in range(self.n_y):
                I[i, j] = i
                J[i, j] += j
        for k in range(self.n_theta):
            for m in range(self.n_phi):
                ex += self.data[k, m] * np.exp(1j * self.k * (I * self.delta_i[k, m] + J * self.delta_j[k, m]))
        ex = ex * self.window2d
        return ex / self.n_theta / self.n_phi

    def steer(self, a, ph0):
        ph0 = ph0 * np.pi / 180 + np.pi / 2
        ps0 = self.k * self.d_x * np.cos(ph0)
        return self.scan(a, ps0)

    def scan(self, a, ps0):
        if type(a) is float:
            N = 1
        else:
            N = len(a)
        m = np.arange(1, N+1, 1)
        return a * np.exp(-1j * m * ps0)

    def schelkunoff_zero_placement(self, theta=0, phi=0):
        theta_rad = theta / 180 * np.pi
        phi_rad = phi / 180 * np.pi
        # j = np.arange(1, self.n_y, dtype=complex)
        # i = np.arange(1, self.n_y, dtype=complex)
        # I, J = np.meshgrid(i, j)
        # k_dx = np.pi * self.d_x
        # k_dy = np.pi * self.d_y
        psi = np.linspace(np.pi / 5, 2 * np.pi - np.pi / 5 , self.n_x-1)
        # psi[1] = psi[1] - np.pi/20
        # psi[8] = psi[8] + np.pi/20
        # psi[0] = psi[0] - np.pi/20
        # psi[9] = psi[9] + np.pi/20

        # psi_deg = psi * 180 / np.pi
        z_i = np.exp(-1j * (psi))
        excitation_psi = np.poly(z_i)[::-1]
        excitation_psi = self.steer(excitation_psi, theta)
        if self.n_y > 1:
            alfa = np.linspace(np.pi / 5, 2 * np.pi - np.pi / 5 , self.n_y-1)
            y_i = -np.exp(1j * alfa)
            excitation_phi = np.poly(y_i)[::-1]
            excitation_phi = self.steer(excitation_phi, phi)
            excitation = np.outer(excitation_psi, excitation_phi)
        else:
            excitation = excitation_psi.reshape([self.n_x, self.n_y])
        self.zeros = [z_i, y_i]

        # excitation = excitation / np.max((excitation))

        # excitation = excitation.reshape([self.n_x, self.n_y])
        return excitation

    def dolph(self, theta=0, phi=0, R=20):
        Ra = 10**(R / 20)                       # sidelobe level in absolute units
        x0 = np.cosh(np.arccosh(Ra) / (self.n_x-1))      # scaling factor
        y0 = np.cosh(np.arccosh(Ra) / (self.n_y-1))

        i = np.arange(1, self.n_x, 1)
        j = np.arange(1, self.n_y, 1)
        x = np.cos(np.pi * (i - 0.5) / (self.n_x - 1)) # N1 zeros of Chebyshev polynomial T_N1(x)
        y = np.cos(np.pi * (j - 0.5) / (self.n_y - 1))
        psi = 2 * np.arccos(x / x0) # N1 array pattern zeros in psi - space
        zeta = 2 * np.arccos(y / y0)
        z = np.exp(1j * psi) # N1 zeros of array polynomial
        w = np.exp(1j * zeta)
        a = np.real(np.poly(z)) # zeros - to - polynomial form, N1 + 1 = N  coefficients
        b = np.real(np.poly(w))
        a = self.steer(a, theta) # steer towards theta
        self.zeros = np.roots(a)
        b = self.steer(b, phi)
        a.resize([len(a), 1])
        b.resize([len(b), 1]);
        excitation = np.outer(a, b)
        return excitation

    def excitation_from_zeros(self, zeros):
        a = np.real(np.poly(zeros))  # zeros - to - polynomial form, N1 + 1 = N  coefficients
        b = np.real(np.poly(zeros))
        # a.resize([len(a), 1])
        # b.resize([len(b), 1]);
        excitation = np.outer(a, b)
        return excitation

    def create_template(self, theta, phi):
        self.phi_r = phi
        self.thera_r = theta
        self.index_phi = self.phi_index(phi)
        self.index_theta = self.theta_index(theta)
        self.template = np.ones([self.n_phi, self.n_theta]) * 0
        self.template[self.index_theta, self.index_phi] = self.n_theta * self.n_phi
        self.data = self.template

    def detect_local_maxima(self, arr):
        neighborhood = morphology.generate_binary_structure(len(arr.shape), 2)
        local_min = (filters.maximum_filter(arr, footprint=neighborhood) == arr)
        background = (arr == 0)  #
        eroded_background = morphology.binary_erosion(
            background, structure=neighborhood, border_value=1)
        detected_minima = local_min ^ eroded_background
        return np.where(detected_minima)

    def save_array_factor(self, filename):
        file = open(filename,'w')
        for i in range(self.n_theta):
            for j in range(self.n_phi):
                file.write("{}, ".format(self.data[i,j]))
            file.write('\n')
        file.close()

    def read_array_factor(self, filename):
        af = np.zeros([self.n_theta, self.n_phi], dtype=complex)
        file = open(filename, 'r')
        file_text = file.read()
        lines = file_text.strip().split('\n')
        numbers = []
        for line in lines:
            numbers.extend(line.strip().split(', '))

        k = 0
        for i in range(self.n_theta):
            for j in range(self.n_phi):
                if numbers[k].strip() != ',':
                    numbers[k] = numbers[k].replace(',','')
                    af[i, j] = complex(numbers[k])
                    k += 1
        return af





if __name__ == '__main__':
   pass
