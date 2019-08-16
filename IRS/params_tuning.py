import numpy as np
from copy import deepcopy, copy
from IRS.main import main_1
import os
import csv
from datetime import datetime
import json

NUM_GROUPS = 0
NUM_CLUSTERS = 1
DIST_FUNCTION_NAME = 2
THRESHOLD = 3


class ParticleSwarmOptimization(object):

    def __init__(self, fitness_function, varsize, swarmsize, epochs, search_space, c1, c2):
        self.fitness_function = fitness_function
        self.varsize = varsize
        self.swarmsize = swarmsize
        self.epochs = epochs
        self.range_num_groups = search_space[NUM_GROUPS]
        self.range_num_clusters = search_space[NUM_CLUSTERS]
        self.range_dist_function = search_space[DIST_FUNCTION_NAME]
        self.range_threshold = search_space[THRESHOLD]
        self.c1 = c1
        self.c2 = c2
        self.velocity = np.zeros((self.swarmsize, self.varsize))
        self.population = None
        self.fitness_population = np.zeros(swarmsize)
        self.pBest = None
        self.gBest = None
        self.fitness_gBest = 10
        self.fitness_pBest = np.zeros(swarmsize)
        self.temp = self.gBest

    def inititalize_algorithms(self):

        population = np.zeros((self.swarmsize, self.varsize))

        init_num_groups = np.random.randint(self.range_num_groups[0], self.range_num_groups[1], self.swarmsize)
        init_num_clusters = np.random.randint(self.range_num_clusters[0], self.range_num_clusters[1], self.swarmsize)
        init_dist_function = np.random.randint(self.range_dist_function[0], self.range_dist_function[1], self.swarmsize)
        init_threshold = np.random.uniform(self.range_threshold[0], self.range_threshold[1])

        population[:, NUM_GROUPS] += init_num_groups
        population[:, NUM_CLUSTERS] += init_num_clusters
        population[:, DIST_FUNCTION_NAME] += init_dist_function
        population[:, THRESHOLD] += init_threshold

        for i in range(self.swarmsize):
            population[i] = self.evaluate_population(population[i])
            fitness_i = self.get_fitness(population[i])
            self.fitness_population[i] = fitness_i
            self.fitness_pBest[i] += fitness_i
            if fitness_i < self.fitness_gBest:
                self.gBest = copy(population[i])
                self.fitness_gBest = fitness_i
        self.population = copy(population)
        self.pBest = copy(population)

    def get_fitness(self, particle):
        return self.fitness_function(particle)

    def set_gBest(self, gBest):
        self.gBest = gBest

    def evaluate_population(self, particle):
        for j in range(self.varsize):
            if j == NUM_GROUPS: # 8, 16, 32, 64

                particle[j] = np.maximum(particle[j], self.range_num_groups[0])
                particle[j] = np.minimum(particle[j], self.range_num_groups[1])
                particle[j] = int(particle[j])

                if 8 <= particle[j] <= 12: particle[j] = 8
                if 12 < particle[j] <= 24: particle[j] = 16
                if 24 < particle[j] <= 48: particle[j] = 32
                if 48 < particle[j] <= 64: particle[j] = 64

            if j == NUM_CLUSTERS: # 16-32
                particle[j] = np.maximum(particle[j], self.range_num_clusters[0])
                particle[j] = np.minimum(particle[j], self.range_num_clusters[1])
                particle[j] = int(particle[j])

            if j == DIST_FUNCTION_NAME:
                particle[j] = np.maximum(particle[j], self.range_dist_function[0])
                particle[j] = np.minimum(particle[j], self.range_dist_function[1])
                particle[j] = int(particle[j])

            if j == THRESHOLD: # 16-32
                particle[j] = np.maximum(particle[j], self.range_threshold[0])
                particle[j] = np.minimum(particle[j], self.range_threshold[1])

        return particle

    def save_gbest(self):
        results = []
        for params in self.gBest.tolist():
            results.append(params)
        results.append(self.fitness_gBest)
        with open("params_tuning_results.csv", "a+", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(results)

    def run(self):

        v_max = 10
        w_max = 0.9
        w_min = 0.4

        print("...........Initialization.........")
        self.inititalize_algorithms()
        self.save_gbest()

        gBest_collection = np.zeros(self.epochs)
        start_time = datetime.now()
        for iter in range(self.epochs):
            print(self.gBest)
            print("start iter {}".format(iter))
            w = (self.epochs - iter) / self.epochs * (w_max - w_min) + w_min
            # w = 1 - iter/(self.epochs + 1)
            new_population = np.zeros((self.swarmsize, self.varsize))
            for i in range(self.swarmsize):
                start_time = datetime.now()

                print("particle {}".format(i))
                r1 = np.random.random()
                r2 = np.random.random()
                position_i = self.population[i]
                new_velocity_i = w*self.velocity[i] \
                                 + self.c1*r1*(self.pBest[i] - position_i) \
                                 + self.c2*r2*(self.gBest - position_i)
                new_velocity_i = np.maximum(new_velocity_i, -0.1 * v_max)
                new_velocity_i = np.minimum(new_velocity_i, 0.1 * v_max)
                self.velocity[i] = new_velocity_i
                new_position_i = self.evaluate_population(position_i + new_velocity_i)

                new_population[i] = new_position_i

                fitness_new_pos_i = self.get_fitness(new_position_i)
                self.fitness_population[i] = fitness_new_pos_i

                if fitness_new_pos_i < self.fitness_pBest[i]:
                    self.pBest[i] = copy(new_position_i)
                    self.fitness_pBest[i] = fitness_new_pos_i

                search_time = datetime.now() - start_time
                search_time_in_s = (search_time.days * 24 * 60 * 60 +
                                     search_time.seconds + search_time.microseconds/10e6)

                print("time for particle {} is {}".format(i, search_time_in_s))

            current_gbest_index = np.where(self.fitness_population==np.amin(self.fitness_population))[0][0]
            current_gbest = self.population[current_gbest_index]
            self.gBest = copy(current_gbest)
            self.fitness_gBest = self.fitness_population[current_gbest_index]

            self.save_gbest()
            self.population = copy(new_population)
            print("result in iter {} is {} with fitness {}".format(iter, self.gBest, self.fitness_gBest))
            print("************************************************")
            # print(self.get_fitness(self.gBest))

            total_time = datetime.now() - start_time
            total_time_in_s = (total_time.days * 24 * 60 * 60 +
                                total_time.seconds)
            print(total_time_in_s)

        # print(total_time)
        return self.get_fitness(self.gBest), gBest_collection


class WhaleOptimizationAlgorithm(object):

    def __init__(self, fitness_function, varsize, swarmsize, epochs, search_space):
        self.fitness_function = fitness_function
        self.varsize = varsize
        self.swarmsize = swarmsize
        self.epochs = epochs
        self.range_num_groups = search_space[NUM_GROUPS]
        self.range_num_clusters = search_space[NUM_CLUSTERS]
        self.range_dist_function = search_space[DIST_FUNCTION_NAME]
        self.range_threshold = search_space[THRESHOLD]
        self.population = None
        self.fitness_population = np.zeros(swarmsize)
        self.pBest = None
        self.gBest = None
        self.fitness_gBest = 10
        self.fitness_pBest = np.zeros(swarmsize)
        self.temp = self.gBest

    def inititalize_algorithms(self):

        population = np.zeros((self.swarmsize, self.varsize))

        init_num_groups = np.random.randint(self.range_num_groups[0], self.range_num_groups[1], self.swarmsize)
        init_num_clusters = np.random.randint(self.range_num_clusters[0], self.range_num_clusters[1], self.swarmsize)
        init_dist_function = np.random.randint(self.range_dist_function[0], self.range_dist_function[1], self.swarmsize)
        init_threshold = np.random.uniform(self.range_threshold[0], self.range_threshold[1], self.swarmsize)

        population[:, NUM_GROUPS] += init_num_groups
        population[:, NUM_CLUSTERS] += init_num_clusters
        population[:, DIST_FUNCTION_NAME] += init_dist_function
        population[:, THRESHOLD] += init_threshold

        for i in range(self.swarmsize):
            population[i] = self.evaluate_population(population[i])
            fitness_i = self.get_fitness(population[i])
            self.fitness_population[i] = fitness_i
            self.fitness_pBest[i] += fitness_i
            if fitness_i < self.fitness_gBest:
                self.gBest = copy(population[i])
                self.fitness_gBest = fitness_i
        self.population = copy(population)
        self.pBest = copy(population)

    def create_solution(self):
        init_num_groups = np.random.randint(self.range_num_groups[0], self.range_num_groups[1])
        init_num_clusters = np.random.randint(self.range_num_clusters[0], self.range_num_clusters[1])
        init_dist_function = np.random.randint(self.range_dist_function[0], self.range_dist_function[1])
        init_threshold = np.random.uniform(self.range_threshold[0], self.range_threshold[1])

        new_rand_solution = np.array([init_num_groups, init_num_clusters, init_dist_function, init_threshold])

        return new_rand_solution

    def get_fitness(self, particle):
        print(particle)
        return self.fitness_function(particle)

    def evaluate_population(self, particle):
        for j in range(self.varsize):
            if j == NUM_GROUPS: # 8, 16, 32, 64

                particle[j] = np.maximum(particle[j], self.range_num_groups[0])
                particle[j] = np.minimum(particle[j], self.range_num_groups[1])
                particle[j] = int(particle[j])

                if 0 <= particle[j] <= 10: particle[j] = 8
                if 10 < particle[j] <= 20: particle[j] = 16
                if 20 < particle[j] <= 30: particle[j] = 32
                if 30 < particle[j] <= 40: particle[j] = 64
                if 40 < particle[j] <= 50: particle[j] = 128
                if 50 < particle[j] <= 60: particle[j] = 256

            if j == NUM_CLUSTERS: # 8-64
                particle[j] = np.maximum(particle[j], self.range_num_clusters[0])
                particle[j] = np.minimum(particle[j], self.range_num_clusters[1])
                particle[j] = int(particle[j])

            if j == DIST_FUNCTION_NAME:
                particle[j] = np.maximum(particle[j], self.range_dist_function[0])
                particle[j] = np.minimum(particle[j], self.range_dist_function[1])
                particle[j] = int(particle[j])

            if j == THRESHOLD:
                particle[j] = np.maximum(particle[j], self.range_threshold[0])
                particle[j] = np.minimum(particle[j], self.range_threshold[1])

        return particle

    def save_gbest(self):
        results = []
        for params in self.gBest.tolist():
            results.append(params)
        results.append(self.fitness_gBest)
        with open("params_tuning_results.csv", "a+", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(results)

    def run(self):

        print("...........Initialization.........")
        self.inititalize_algorithms()
        self.save_gbest()

        gBest_collection = np.zeros(self.epochs)
        start_time = datetime.now()
        for iter in range(self.epochs):
            print("start iter {}: gBest {} with fitness {}".format(iter, self.gBest, self.fitness_gBest))
            a = 2 - 2 * iter / (self.epochs - 1)
            # w = 1 - iter/(self.epochs + 1)
            new_population = np.zeros((self.swarmsize, self.varsize))
            new_population_fitness = np.zeros(self.swarmsize)
            for i in range(self.swarmsize):
                start_time = datetime.now()

                current_agent = self.population[i]

                r = np.random.rand()
                A = 2 * a * r - a
                C = 2 * r
                l = np.random.uniform(-1, 1)
                p = 0.5
                b = 1

                if np.random.uniform() < p:
                    if np.abs(A) < 1:
                        D = np.abs(C * self.gBest - current_agent)
                        new_position = self.gBest - A * D
                    else:
                        # x_rand = pop[np.random.randint(self.pop_size)] # chon ra 1 thang random
                        x_rand = self.create_solution()
                        D = np.abs(C * x_rand - current_agent)
                        new_position = (x_rand - A * D)
                else:
                    D1 = np.abs(self.gBest - current_agent)
                    new_position = D1 * np.exp(b * l) * np.cos(2 * np.pi * l) + self.gBest

                new_position = self.evaluate_population(new_position)
                new_fitness = self.get_fitness(new_position)
                new_population[i] += new_position
                new_population_fitness[i] += new_fitness

                search_time = datetime.now() - start_time
                search_time_in_s = (search_time.days * 24 * 60 * 60 +
                                    search_time.seconds + search_time.microseconds / 10e6)

                print("time for particle {} is {}".format(i, search_time_in_s))

            self.population = copy(new_population)
            self.fitness_population = copy(new_population_fitness)

            if np.amin(self.fitness_population) < self.fitness_gBest:
                current_gbest_index = np.where(self.fitness_population == np.amin(self.fitness_population))[0][0]
                current_gbest = self.population[current_gbest_index]
                self.gBest = copy(current_gbest)
                self.fitness_gBest = self.fitness_population[current_gbest_index]
            self.save_gbest()

            print("result in iter {} is {} with fitness {}".format(iter, self.gBest, self.fitness_gBest))
            print("************************************************")
            # print(self.get_fitness(self.gBest))

            total_time = datetime.now() - start_time
            total_time_in_s = (total_time.days * 24 * 60 * 60 +
                                total_time.seconds)
            print(total_time_in_s)

        # print(total_time)
        return self.get_fitness(self.gBest), gBest_collection


fitness_function = main_1
varsize = 4
swarmsize = 20
epochs = 20
search_space = [[0, 40], [8, 32], [0, 100], [0.3, 0.6]]
# print("*********************STARTING PSO************************")
# pso = ParticleSwarmOptimization(fitness_function, varsize, swarmsize, epochs, search_space, c1, c2)
# pso.run()
# print("*********************FINISH PSO************************")

print("*********************STARTING WOA************************")
pso = WhaleOptimizationAlgorithm(fitness_function, varsize, swarmsize, epochs, search_space)
pso.run()
print("*********************FINISH WOA************************")