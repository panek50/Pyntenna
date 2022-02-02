import matplotlib.pylab as plt
from antenna_array import AntennaArray
from artap.problem import Problem
from artap.datastore import SqliteDataStore
import numpy as np
from artap.algorithm_bayesopt import BayesOptSerial
from artap.operators import LHSGenerator, GradientEvaluator

class AntennaArrayProblem(Problem):
    """ Describe simple one objective optimization problem. """

    def set(self):
        n = 11
        self.antenna = AntennaArray(n, n, assamble_L=True)
        self.theta_r = 0
        self.phi_r = 0
        self.name = "LocalPythonProblem"
        self.parameters = []
        for i in range(5):
            self.parameters.append(
                {'name': 'I_{}'.format(i), 'bounds': [-20, 20], 'parameter_type': 'float', 'initial_value': 0})

        self.costs = [{'name': 'F_1', 'criteria': 'maximize'}, {'name': 'F_2', 'criteria': 'minimize'}]

    def vector_to_excitation(self, vector):
        poly = np.ones(self.antenna.n_x)
        zeros = np.roots(poly)
        zeros_theta = sorted(zeros, key=lambda x: np.abs(np.angle(x)))
        k = 0
        deg = np.pi / 180
        for i in range((11 - 1) // 2):
            zeros_theta[k] = zeros_theta[k] * np.exp(1j * deg * vector[i])
            zeros_theta[k + 1] = zeros_theta[k + 1] * np.exp(-1j * deg * vector[i])
            k += 2
        self.antenna.zeros = zeros_theta
        a = np.real(np.poly(zeros_theta))
        b = np.real(np.poly(zeros_theta))
        a = self.antenna.steer(a, self.theta_r) # steer towards theta
        b = self.antenna.steer(b, self.phi_r) # steer towards phi
        a.resize([len(a), 1])
        b.resize([len(b), 1]);
        excitation = np.outer(a, b)
        return excitation

    def evaluate(self, individual):
        excitation = self.vector_to_excitation(individual.vector)
        self.antenna.excitation = excitation
        af = self.antenna.calculate_matrix(excitation)
        index = self.antenna.phi_index(0)
        maxima_indices = self.antenna.detect_local_maxima(np.abs(af[:, index]))
        maxima = []
        ref = np.ones(12) * 1
        ref[1:] = ref[1:] * 0.1
        for i in range(len(maxima_indices[0])):
            k = maxima_indices[0][i]
            maxima.append(np.abs(af[k, index]))
        maxima.sort(reverse=True)
        F_1 = np.abs(maxima[0]) - np.max(np.abs(maxima[1:]))
        print(F_1)
        return [F_1]

problem = AntennaArrayProblem()
algorithm = BayesOptSerial(problem)
algorithm.options['verbose_level'] = 0
algorithm.options['n_iterations'] = 150
algorithm.run()

population = problem.last_population()
max = 10
index = 0
for i, individual in enumerate(problem.individuals):
    if individual.costs[0] < max:
        max = individual.costs[0]
        index = i
excitation = problem.vector_to_excitation(problem.individuals[index].vector)
af = problem.antenna.calculate(excitation)
problem.antenna.plot_2d_elevation(normalize=True)
problem.antenna.plot_3d()
problem.antenna.plot_zeros()
plt.show()



