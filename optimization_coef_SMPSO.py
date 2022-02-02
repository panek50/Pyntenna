import matplotlib.pylab as plt
from antenna_array import AntennaArray
from artap.problem import Problem
from artap.results import Results
from artap.algorithm_swarm import SMPSO

import numpy as np

class AntennaArrayProblem(Problem):
    """ Describe simple one objective optimization problem. """

    def set(self):
        n = 11
        self.antenna = AntennaArray(n, n, assamble_L=True)
        self.antenna.create_template(0, 0)
        self.antenna.crete_window(n)
        self.name = "LocalPythonProblem"
        self.parameters = []
        for i in range(2 * (n-1)):
            self.parameters.append(
                {'name': 'I_{}'.format(i), 'bounds': [0.1, 2], 'parameter_type': 'float', 'initial_value': 1})
            self.parameters.append(
                {'name': 'alpha_{}'.format(i), 'bounds': [-np.pi/2, np.pi/2], 'parameter_type': 'float'})

        self.costs = [{'name': 'F_1', 'criteria': 'maximize'}, {'name': 'F_2', 'criteria': 'maximize'}]

    def vector_to_excitation(self, vector):
        n_theta = self.antenna.n_x
        n_phi = self.antenna.n_y
        a = np.zeros(n_theta, dtype=complex)
        b = np.zeros(n_phi, dtype=complex)
        k = 0
        for i in range(n_theta):
            a[i] = vector[k] * (np.cos(vector[k + 1])
                                                + 1j * np.sin(vector[k + 1]))
            b[i] = vector[k + n_theta] * (np.cos(vector[k + n_theta + 1])
                                                + 1j * np.sin(vector[k + n_theta + 1]))
            k += 2
        a = self.antenna.steer(a, 0) # steer towards theta
        b = self.antenna.steer(b, 0) # steer towards phi
        a.resize([len(a), 1])
        b.resize([len(b), 1]);
        excitation = np.outer(a, b)
        if np.max(np.abs(excitation)) != 0:
            excitation = excitation / np.max(np.abs(excitation))
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
        index = self.antenna.phi_index(0)
        theta_index = self.antenna.theta_index(0)
        phi_index = self.antenna.phi_index(0)
        af = magnitude[:, phi_index]

        magnitude_r = magnitude[theta_index, phi_index]
        print(magnitude_r)
        return [magnitude_r, magnitude_r]


problem = AntennaArrayProblem()
algorithm = SMPSO(problem)
algorithm.options['max_population_number'] = 300
algorithm.options['max_population_size'] = 200
algorithm.options['max_processes'] = 1
algorithm.run()

population = problem.last_population()
excitation = problem.vector_to_excitation(population[0].vector)
print(np.abs(excitation))
af = problem.antenna.calculate(excitation)
problem.antenna.plot_2d_elevation(normalize=False)
plt.show()

results = Results(problem=problem)
problem.antenna.save_array_factor('./results/array_factor_SM_PSO_coef.txt')
x = []
for i in range(algorithm.options['max_population_number']):
    table = results.goal_on_index('F_1', population_id=i)
    x.append(np.nanmax(table[1]))

file = open('./results/convergence_SM_PSO_coef.txt', 'w')
for i in range(len(x)):
    file.write("{} \n".format(x[i]))
file.close()