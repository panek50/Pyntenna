import matplotlib.pylab as plt
from antenna_array import AntennaArray
from artap.problem import Problem
from artap.results import Results
from artap.datastore import SqliteDataStore
import numpy as np


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
        a = np.real(np.poly(zeros_theta))
        b = np.real(np.poly(zeros_theta))
        a = self.antenna.steer(a, self.theta_r) # steer towards theta
        b = self.antenna.steer(b, self.phi_r) # steer towards phi
        a.resize([len(a), 1])
        b.resize([len(b), 1])
        excitation = np.outer(a, b)
        return excitation

    def evaluate(self, individual):
        excitation = self.vector_to_excitation(individual.vector)
        self.antenna.excitation = excitation
        array_factor = self.antenna.calculate_matrix(excitation)
        magnitude = np.abs(array_factor)
        index = self.antenna.phi_index(0)
        maxima_indices = self.antenna.detect_local_maxima(np.abs(array_factor[:, index]))
        maxima = []
        ref = np.ones(12) * 3
        ref[1:] = ref[1:] * 0.05
        for i in range(len(maxima_indices[0])):
            k = maxima_indices[0][i]
            maxima.append(np.abs(array_factor[k, index]))
        maxima.sort(reverse=True)

        F = 0
        for i in range(len(maxima)):
            if np.abs(ref[i] - maxima[i]) > F:
                F = np.abs(ref[i] - maxima[i])
        F_2 = np.sum((ref[:len(maxima)] - maxima))
        theta_index = self.antenna.theta_index(0)
        phi_index = self.antenna.phi_index(0)
        magnitude_r = magnitude[theta_index, phi_index]
        print(F, F_2)
        return [F, F]

problem = AntennaArrayProblem()
n = 60
m = 19
y = np.zeros([m, n])
x = np.zeros(n)
table = None
for j in range(0, m):
    database_name = 'zeros_symmetry_NSGA_II_{}.sqlite'.format(j)
    problem.data_store = SqliteDataStore(problem, database_name=database_name, mode='read')
    results = Results(problem=problem)

    for i in range(0, n):
        table = results.goal_on_index('F_1', population_id=i)
        y[j, i] = (np.min((table[1])))
        x[i] = i


mean = np.mean(y, axis=0)
std = np.std(y, axis=0)
plt.xticks(size=14)
plt.xticks(size=14)
plt.plot(mean+std, 'k--')
plt.plot(mean, 'k')
plt.plot(mean-std, 'k--')
plt.fill_between(x, mean-std, mean+std, alpha=0.4)
plt.ylim([0, 1])
plt.grid()
plt.xlabel(r'$k$ [-]', fontsize=16)
plt.ylabel(r'$\mathcal{F}$', fontsize=16)
plt.tight_layout()
plt.savefig('mean_std_multi.pdf')
plt.show()

database_name = 'zeros_symmetry_NSGA_II_{}.sqlite'.format(0)
problem.data_store = SqliteDataStore(problem, database_name=database_name, mode='read')
problem.theta_r = 0
results = Results(problem=problem)
population = problem.last_population()
for i, individual in enumerate(population):
    if individual.features['front_number'] == 1:
        excitation = problem.vector_to_excitation(individual.vector)
        quantized = problem.antenna.quantize(excitation, n_bit=3)
        af_ex = problem.antenna.calculate_matrix(excitation)
        problem.antenna.plot_2d_elevation(normalize=True)
        #problem.antenna.plot_zeros()
        af_qa = problem.antenna.calculate(quantized)
        print(i, np.sum(np.abs(af_ex-af_qa)))
        problem.antenna.plot_2d_elevation(normalize=True, markers='k--')
        #problem.antenna.plot_zeros()
        plt.tight_layout()
        plt.savefig('comparison_quantized.pdf')
        plt.show()

table = results.pareto_front(-1)

plt.figure('pareto')
for individual in problem.individuals:
    plt.plot(individual.costs[0], individual.costs[1], 'kx')

plt.xticks(size=14)
plt.yticks(size=14)
plt.plot(table[0], table[1], 'ro')
plt.xlabel(r'$F_1$', fontsize=16)
plt.ylabel(r'$F_2$', fontsize=16)
plt.ylim([0, 0.5])
plt.xlim([0, 0.5])
plt.tight_layout()
plt.grid()
plt.savefig('pareto_mono.pdf')
plt.show()