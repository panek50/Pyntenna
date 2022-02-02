import matplotlib.pylab as plt
from antenna_array import AntennaArray
from artap.problem import Problem
from artap.operators import PartiallyCustomGenerator
from artap.algorithm_genetic import NSGAII
import numpy as np


class AntennaArrayProblem(Problem):
    """ Describe simple one objective optimization problem. """

    def set(self):
        n = 11
        self.antenna = AntennaArray(n, n, assamble_L=True)
        self.antenna.create_template(20, 0)
        self.antenna.crete_window(n)
        self.name = "LocalPythonProblem"
        self.parameters = []
        for i in range(self.antenna.n_x * self.antenna.n_y):
            self.parameters.append(
                {'name': 'I_{}'.format(i), 'bounds': [0, 1], 'parameter_type': 'float', 'initial_value': 0.5})
            self.parameters.append(
                {'name': 'alpha_{}'.format(i), 'bounds': [-np.pi / 2, np.pi / 2], 'parameter_type': 'float'})

        self.costs = [{'name': 'F_1', 'criteria': 'maximize'}, {'name': 'F_2', 'criteria': 'maximize'}]

    def vector_to_excitation(self, vector):
        k = 0
        n_x = self.antenna.n_x
        n_y = self.antenna.n_y
        excitation = np.zeros([n_x, n_y], dtype=complex)
        for i in range(n_x):
            for j in range(n_y):
                excitation[i, j] = vector[k] * (np.cos(vector[k + 1])
                                                + 1j * np.sin(vector[k + 1]))
                k += 2
        return excitation

    def excitation_to_vector(self, vector):
        n_x = self.antenna.n_x
        n_y = self.antenna.n_y
        vector=np.zeros(2 * n_x * n_y)
        k = 0
        for i in range(n_x):
            for j in range(n_y):
                vector[k] = np.abs(excitation[i, j])
                vector[k+1] = np.angle(excitation[i, j])
                k += 2
        return vector

    def evaluate(self, individual):
        excitation = self.vector_to_excitation(individual.vector)
        self.antenna.excitation = excitation
        array_factor = self.antenna.calculate_matrix(excitation)
        magnitude = np.abs(array_factor)

        # magnitude on required position
        theta_index = self.antenna.theta_index(20)
        phi_index = self.antenna.phi_index(0)
        magnitude_r = magnitude[theta_index, phi_index]
        print(np.max(magnitude))
        return [magnitude_r]


problem = AntennaArrayProblem()
algorithm = NSGAII(problem)
algorithm.options['max_population_number'] = 40
algorithm.options['max_population_size'] = 20
algorithm.options['max_processes'] = 1
algorithm.generator = PartiallyCustomGenerator(problem.parameters)

excitation = problem.antenna.fourier_series()
excitation = excitation / np.max(np.abs(excitation))

vector = problem.excitation_to_vector(excitation)
algorithm.generator.init([vector, vector, vector, vector, vector, vector, vector, vector, vector], number=20)
algorithm.run()


population = problem.last_population()
excitation = problem.vector_to_excitation(population[0].vector)
af = problem.antenna.calculate(excitation)
problem.antenna.plot_2d_elevation()
problem.antenna.plot_excitation(np.abs(excitation))
plt.show()