import matplotlib.pylab as plt
from antenna_array import AntennaArray
from artap.problem import Problem
from artap.results import Results
import numpy as np
from artap.algorithm_genetic import EpsMOEA


class AntennaArrayProblem(Problem):
    """ Describe simple one objective optimization problem. """

    def set(self):
        n = 11
        self.antenna = AntennaArray(n, n, assamble_L=True)
        self.theta_r = 0
        self.phi_r = 0
        self.name = "LocalPythonProblem"
        self.parameters = []
        for i in range(2 * n - 2):
            self.parameters.append(
                {'name': 'alpha_{}'.format(i), 'bounds': [-np.pi/11, np.pi/11], 'parameter_type': 'float'})

        self.costs = [{'name': 'F_1', 'criteria': 'maximize'}, {'name': 'F_2', 'criteria': 'maximize'}]

    def vector_to_excitation(self, vector):
        n_theta = self.antenna.n_x - 1
        n_phi = self.antenna.n_y - 1
        zeros_theta = np.zeros(n_theta, dtype=complex)
        zeros_phi = np.zeros(n_phi, dtype=complex)
        k = 0
        alpha = np.linspace(-np.pi, np.pi, n_theta)
        for i in range(n_theta):
            zeros_theta[i] = (np.cos(vector[k] + alpha[i])
                                                + 1j * np.sin(vector[k] + alpha[i]))
            zeros_phi[i] = (np.cos(vector[k + n_theta] + alpha[i])
                                                 + 1j * np.sin(vector[k + n_theta] + alpha[i]))
            k += 1
        self.antenna.zeros = zeros_theta
        a = np.real(np.poly(zeros_theta))
        b = np.real(np.poly(zeros_phi))
        a = self.antenna.steer(a, self.theta_r) # steer towards theta
        b = self.antenna.steer(b, self.phi_r) # steer towards phi
        a.resize([len(a), 1])
        b.resize([len(b), 1]);
        excitation = np.outer(a, b)
        excitation = excitation / np.max(np.abs(excitation))
        return excitation

    def evaluate(self, individual):
        excitation = self.vector_to_excitation(individual.vector)
        self.antenna.excitation = excitation
        array_factor = self.antenna.calculate_matrix(excitation)
        magnitude = np.abs(array_factor)
        index = self.antenna.phi_index(0)
        maxima_indices = self.antenna.detect_local_maxima(np.abs(array_factor[:, index]))
        maxima = []
        for i in range(len(maxima_indices[0])):
            k = maxima_indices[0][i]
            maxima.append(np.abs(array_factor[k, index]))
        maxima.sort(reverse=True)
        max = np.max(maxima[1:])
        theta_index = self.antenna.theta_index(0)
        phi_index = self.antenna.phi_index(0)
        magnitude_r = magnitude[theta_index, phi_index]
        print(magnitude_r - max)
        return [magnitude_r-max]

problem = AntennaArrayProblem()
algorithm = EpsMOEA(problem)
algorithm.options['max_population_number'] = 60
algorithm.options['max_population_size'] = 50
algorithm.options['max_processes'] = 1
algorithm.run()

population = problem.last_population()
excitation = problem.vector_to_excitation(population[-1].vector)
af = problem.antenna.calculate(excitation)
problem.antenna.plot_2d_elevation()
problem.antenna.plot_3d()
problem.antenna.plot_zeros()
plt.show()

problem.antenna.save_array_factor('./results/array_factor_EPS_MOEA_zeros.txt')

results = Results(problem=problem)
x = []
for i in range(algorithm.options['max_population_number']):
    table = results.goal_on_index('F_1', population_id=i)
    x.append(np.max(np.abs(table[1])))

file = open('./results/convergence_EPS_MOEA_zeros.txt', 'w')
for i in range(len(x)):
    file.write("{} \n".format(x[i]))
file.close()