#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import random
import numpy
import sys
import copy

from moead import MOEAD

from deap import base
from deap import creator
from deap import tools
import numpy as np

IND_INIT_SIZE = 5
MAX_ITEM = 50
MAX_WEIGHT = 50
NBR_ITEMS = 20


NGEN = 50
MU = 50
LAMBDA = 2
CXPB = 0.7
MUTPB = 0.2


class TSP:
    def __init__(self, instance):
        f = open(instance, "r")
        # load number of instances and objectives
        self.N, self.M = f.readline().split(" ")
        self.N = int(self.N)
        self.M = int(self.M)
        self.Data = np.zeros([self.M, self.N, self.N])
        for o in range(self.M):
            for i in range(self.N):
                for j in range(i):
                    self.Data[o, i, j] = f.readline()
                    self.Data[o, j, i] = self.Data[o, i, j]

    def evalTSP(self, individual):
        fitness = np.zeros(self.M)
        order = np.argsort(individual)
        #order2 = np.argsort(order)

        for o in range(self.M):
            for i in range(self.N):
                fitness[o] += self.Data[o, order[i], order[(i + 1) % self.N]]
        return tuple(fitness)

    def gausianCrossover(self, ind1, ind2):
        """Use gausean crossover"""
        c1 = copy.deepcopy(ind1)
        c2 = copy.deepcopy(ind2)

        for i in range(self.N):
            c1[i] = numpy.random.normal((ind1[i] + ind2[i])/2, abs(ind1[i] - ind2[i])/2)
            c2[i] = numpy.random.normal((ind1[i] + ind2[i])/2, abs(ind1[i] - ind2[i])/2)
        return c1, c2


    def uniformCrossover(self, ind1, ind2):
        """Apply a uniform crossover operation on input sets."""
        c1 = copy.deepcopy(ind1)
        c2 = copy.deepcopy(ind2)
        for i in range(self.N):
            if random.random() < 0.5:
                c1[i] = ind1[i]
                c2[i] = ind2[i]
            else:
                c1[i] = ind2[i]
                c2[i] = ind1[i]
        return c1, c2

    def uniformRVCrossover(self, ind1, ind2):
        """Apply a uniform crossover operation on input sets."""
        c1 = copy.deepcopy(ind1)
        c2 = copy.deepcopy(ind2)
        for i in range(self.N):
            maxv = 9.5 - max(ind1[i],ind2[i]) / 2
            minv = min(ind1[i],ind2[i]) / 2
            r = random.random()
            c1[i] = minv + (maxv-minv)*r
            c2[i] = maxv - (maxv-minv)*r
        return c1, c2

    def mutSet(self, individual):
        return individual,

    def main(self, seed=64):
        random.seed(seed)

        # Create the item dictionary: item name is an integer, and value is
        # a (weight, value) 2-uple.

        weights = []
        for i in range(self.M):
            weights.append(-1)
        creator.create("Fitness", base.Fitness, weights=weights)

        creator.create("Individual", list, fitness=creator.Fitness)

        toolbox = base.Toolbox()

        # Attribute generator
        toolbox.register("attr_item", random.randrange, NBR_ITEMS)

        # Structure initializers
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.attr_item, self.N)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evalTSP)
        toolbox.register("mate", self.gausianCrossover)
        toolbox.register("mutate", self.mutSet)
        toolbox.register("select", tools.selNSGA2)

        pop = toolbox.population(n=MU)
        hof = tools.ParetoFront()

        stats = {}
        def lambda_factory(idx):
            return lambda ind: ind.fitness.values[idx]

        fitness_tags = []
        for i in range(self.M):
            fitness_tags.append(str(i))
        for tag in fitness_tags:
            s = tools.Statistics( key=lambda_factory(
                        fitness_tags.index(tag)
                    ))
            stats[tag] = s

        mstats = tools.MultiStatistics(**stats)
        mstats.register("avg", numpy.mean, axis=0)
        mstats.register("std", numpy.std, axis=0)
        mstats.register("min", numpy.min, axis=0)
        mstats.register("max", numpy.max, axis=0)

        ea = MOEAD(pop, toolbox, MU, CXPB, MUTPB, ngen=NGEN, stats=mstats, halloffame=hof, nr=LAMBDA)
        pop = ea.execute()

        return pop, stats, hof

                 
if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder = int(sys.argv[1])

    TSP = TSP("instance_100_3")
    pop, stats, hof = TSP.main()

    pop = [str(np.argsort(p)) + " " + str(p.fitness.values) for p in pop]
    hof = [str(np.argsort(h)) + " " + str(h.fitness.values) for h in hof]
    print("POP:")
    print("\n".join(pop))

    print("PF:")
    print("\n".join(hof))
