import matplotlib.pylab as plt
from antenna_array import AntennaArray
from artap.problem import Problem
from artap.results import Results
from artap.datastore import SqliteDataStore
import numpy as np
from artap.algorithm_genetic import NSGAII, EpsMOEA
from artap.algorithm_swarm import SMPSO
from artap.algorithm_gradient_descent import GradientDescent
from artap.operators import GradientEvaluator
class AntennaArrayProblem(Problem):
    """ Describe simple one objective optimization problem. """

    def set(self):
        n = 11
        self.antenna = AntennaArray(n, n, assamble_L=True)
        self.theta_r =0
        self.phi_r = 0
        self.name = "LocalPythonProblem"
        self.parameters = []
        for i in range(5):
            self.parameters.append(
                {'name': 'I_{}'.format(i), 'bounds': [-30, 30], 'parameter_type': 'float', 'initial_value': 0})

        self.costs = [{'name': 'F_1', 'criteria': 'minimize'}, {'name': 'F_2', 'criteria': 'minimize'}]

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
        a = (np.poly(zeros_theta))
        b = (np.poly(zeros_theta))
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
        quantized = problem.antenna.quantize(excitation, n_bit=7)
        af_qa = problem.antenna.calculate_matrix(quantized)
        index = self.antenna.phi_index(0)
        maxima_indices = self.antenna.detect_local_maxima(np.abs(af[:, index]))
        ref = np.ones(12) * 1
        ref[1:] = 0.1
        maxima = np.abs(af[maxima_indices, index]).flatten().tolist()
        maxima_qa = np.abs(af_qa[maxima_indices, index]).flatten().tolist()
        maxima.sort(reverse=True)
        maxima_qa.sort(reverse=True)
        F_1 = 0
        F_2 = 0
        for i in range(len(maxima)):
            if np.abs(ref[i] - maxima[i]) > F_1:
                F_1 = np.abs(ref[i] - maxima[i])
        for i in range(len(maxima)):
            if np.abs(ref[i] - maxima_qa[i]) > F_2:
                F_2 = np.abs(ref[i] - maxima_qa[i])
        print(F_1, F_2)
        return [F_1, F_1]

for i in range(20):
    problem = AntennaArrayProblem()
    database_name = 'zeros_symmetry_NSGA_II_{}.sqlite'.format(i)
    problem.data_store = SqliteDataStore(problem, database_name=database_name)
    
    algorithm = SMPSO(problem)
    algorithm.options['max_population_number'] = 60
    algorithm.options['max_population_size'] = 60
    algorithm.options['max_processes'] = 1
    algorithm.run()
    
    population = problem.last_population()
    excitation = problem.vector_to_excitation(population[0].vector)
    af = problem.antenna.calculate(excitation)
    problem.antenna.plot_2d_elevation(normalize=False)
    problem.antenna.plot_3d()
    problem.antenna.plot_zeros()
    plt.show()


