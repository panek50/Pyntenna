import matplotlib.pylab as plt
from antenna_array import AntennaArray
from artap.problem import Problem
from artap.results import Results
from artap.datastore import SqliteDataStore
from artap.algorithm_genetic import NSGAII
import numpy as np


class AntennaArrayProblem(Problem):
    """ Describe simple one objective optimization problem. """

    def set(self):
        n = 11
        self.antenna = AntennaArray(n, n, assamble_L=True)
        self.name = "AntennaArrayProblem"
        self.parameters = []
        for i in range(self.antenna.n_x * self.antenna.n_y):
            self.parameters.append(
                {'name': 'I_{}'.format(i), 'bounds': [0.0, 1], 'parameter_type': 'float', 'initial_value': 0})
            self.parameters.append(
                {'name': 'alpha_{}'.format(i), 'bounds': [-np.pi / 2, np.pi / 2], 'parameter_type': 'float'})

        self.costs = [{'name': 'F_1', 'criteria': 'maximize'}]

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
    
    def evaluate(self, individual):
        excitation = self.vector_to_excitation(individual.vector)
        self.antenna.excitation = excitation
        array_factor = self.antenna.calculate_matrix(excitation)
        magnitude = np.abs(array_factor)

        # magnitude on required position
        theta_index = self.antenna.theta_index(0)
        phi_index = self.antenna.phi_index(20)
        magnitude_r = magnitude[theta_index, phi_index]
        return [magnitude_r, magnitude_r]


problem = AntennaArrayProblem()
database_name = "antenna_nsga_II.sqlite"
algorithm = NSGAII(problem)
algorithm.options['max_population_number'] = 300
algorithm.options['max_population_size'] = 200
algorithm.options['max_processes'] = 1
algorithm.run()

population = problem.last_population()
excitation = problem.vector_to_excitation(population[0].vector)
af = problem.antenna.calculate(excitation)
problem.antenna.plot_2d_elevation()

problem.antenna.save_array_factor('./results/array_factor_NSGA_II.txt')
results = Results(problem=problem)

x = []
for i in range(algorithm.options['max_population_number']):
    table = results.goal_on_index('F_1', population_id=i)
    x.append(np.max(table[1]))

file = open('./results/convergence_NSGA_II.txt', 'w')
for i in range(len(x)):
    file.write("{} \n".format(x[i]))
file.close()
 

