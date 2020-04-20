import random
import itertools
import pandas as pd
import math
import numpy as np
from copy import deepcopy
from sys import float_info
from joblib import Parallel, delayed, dump, load
from statistics import median, mean
from decimal import *
from tqdm import tqdm

data_list = {}
data = pd.read_csv("Dane_S2_50_10.csv", sep=";", header=None)
data = [[r[0], list(r[1:])] for r in data.values]
data_list["Dane_S2_50_10"] = data
data = pd.read_csv("Dane_S2_100_20.csv", sep=",", header=None)
data = [[r[0], list(r[1:])] for r in data.values]
data_list["Dane_S2_100_20"] = data
data = pd.read_csv("Dane_S2_200_20.csv", sep=",", header=None)
data = [[r[0], list(r[1:])] for r in data.values]
data_list["Dane_S2_200_20"] = data

def swapPositions(lst, pos1, pos2):
    lst[pos1], lst[pos2] = lst[pos2], lst[pos1]
    return lst

def getRandomElementAndIndex(lst):
    rand = random.randrange(0, len(lst))
    return (lst[rand], rand)

def getRandomElement(lst):
    rand = random.randrange(0, len(lst))
    return lst[rand]

def getRandomIndex(lst):
    rand = random.randrange(0, len(lst))
    return rand

def calcMultipleMachinesFitness(lst, task_values):
    last_row = lst[0]
    for idx, _ in enumerate(lst):
        if idx == 0:
            last_row = (lst[idx], addTimes(task_values[lst[idx]], None))
        else:
            last_row = (lst[idx], addTimes(task_values[lst[idx]], last_row[1]))
    return last_row[1][-1]


def addTimes(lst, previous_lst):
    lst_cpy = lst.copy()
    for idx, _ in enumerate(lst_cpy):
        if idx > 0:
            prev = -1
            if previous_lst:
                prev = lst_cpy[idx]+previous_lst[idx]
            current = lst_cpy[idx-1]+lst_cpy[idx]
            if current > prev:
                lst_cpy[idx] = current
            else:
                lst_cpy[idx] = prev
        elif previous_lst:
            lst_cpy[idx] = lst_cpy[idx]+previous_lst[idx]
    return lst_cpy

def mutateSolution(solution):
    x0_idx = random.randint(0, len(data)-1)
    x1_idx = x0_idx
    while x1_idx == x0_idx:
        x1_idx = random.randint(0, len(data)-1)
    swapPositions(solution, x0_idx, x1_idx)

def generateFirstSolution(data, shuffle = False):
    data_copy = data.copy()
    if shuffle:
        random.shuffle(data_copy)
    return data_copy

def tournamentSelection(population, fitnesses, n = 2, k = 2):
    parents = []
    available_indexes = range(len(population))
    for _ in range(n):
        contestants = random.sample(available_indexes, k)
        best_contestant = contestants[0]
        for c in contestants:
            if fitnesses[c] < fitnesses[best_contestant]:
                best_contestant = c
        parents.append(population[best_contestant])
    return parents

def rouletteSelection(population, fitnesses, n = 2):
    max_fitness = max(fitnesses)
    fitnesses = [max_fitness-x for x in fitnesses]
    population = [population for _, population in sorted(zip(fitnesses,population), reverse=True)]
    sum_of_fitnesses = sum(fitnesses)
    parents = []
    for _ in range(n):
        u = random.random() * sum_of_fitnesses
        sum_ = 0
        for i, p in enumerate(population):
            sum_ += fitnesses[i]
            if sum_ >= u:
                parents.append(p)
                break
    return parents

def randomSelection(population, fitnesses, n = 2):
    return [random.choice(population) for i in range(n)]

def bestSelection(population, fitnesses, n = 2):
    return [population for _, population in sorted(zip(fitnesses,population))][0:n]

def worstSelection(population, fitnesses, n = 2):
    return [population for _, population in sorted(zip(fitnesses,population), reverse=True)][0:n]

# z https://github.com/DEAP/deap/blob/master/deap/tools/crossover.py
def PMXCrossover(parent1, parent2):
    size = len(parent1)
    parent1 = [x-1 for x in parent1]
    parent2 = [x-1 for x in parent2]
    p1, p2 = [0] * size, [0] * size

    for i in range(size):
        p1[parent1[i]] = i
        p2[parent2[i]] = i
    a, b = random.sample(range(size), 2)
    if a > b:
        a, b = b, a
    for i in range(a, b):
        temp1 = parent1[i]
        temp2 = parent2[i]
        parent1[i], parent1[p1[temp2]] = temp2, temp1
        parent2[i], parent2[p2[temp1]] = temp1, temp2
        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    return [x+1 for x in parent1], [x+1 for x in parent2]

def OXCrossover(ind1, ind2):
    size = len(ind1)
    ind1 = [x-1 for x in ind1]
    ind2 = [x-1 for x in ind2]
    a, b = random.sample(range(size), 2)
    if a > b:
        a, b = b, a

    holes1, holes2 = [True] * size, [True] * size
    for i in range(size):
        if i < a or i > b:
            holes1[ind2[i]] = False
            holes2[ind1[i]] = False

    temp1, temp2 = ind1, ind2
    k1, k2 = b + 1, b + 1
    for i in range(size):
        if not holes1[temp1[(i + b + 1) % size]]:
            ind1[k1 % size] = temp1[(i + b + 1) % size]
            k1 += 1

        if not holes2[temp2[(i + b + 1) % size]]:
            ind2[k2 % size] = temp2[(i + b + 1) % size]
            k2 += 1

    for i in range(a, b + 1):
        ind1[i], ind2[i] = ind2[i], ind1[i]
    return [x+1 for x in ind1], [x+1 for x in ind2]
    '''
    a, b = random.sample(range(len(parent1)), 2)
    if a > b:
        a, b = b, a
    return(
        OXCrossoverGenerateChild(parent1, parent2, a, b),
        OXCrossoverGenerateChild(parent2, parent1, a, b)
    )
    '''
    
def OXCrossoverGenerateChild(base_parent, other_parent, a, b):
    child = base_parent.copy()
    genes = [x for x in other_parent[b:len(other_parent)] if not x in child[a:b]] + [x for x in other_parent[0:b] if not x in child[a:b]]
    for i in range(b, len(child)):
        child[i] = genes.pop(0)
    for i in range(a):
        child[i] = genes.pop(0)
    return child

def genetic(data, n, max_same_iters, evaluate, selection, crossover, mutation, mutation_probability):
    iters = 0
    counter = 0
    data_dict = dict(data)
    genes, _ = zip(*data)
    genes = list(genes)
    population = []
    for _ in range(n):
        x = genes.copy()
        random.shuffle(x)
        population.append(x)
    last_best_score = sorted([evaluate(p, data_dict) for p in population])[0]
    while(True):
        iters += 1
        offspring = []
        fitnesses = [evaluate(p, data_dict) for p in population]
        best_fitness = sorted(fitnesses)[0]
        print("%s %s %s %s:" % (str(n), str(mutation_probability), crossover.__name__, selection.__name__), 'Best score so far', best_fitness, 'Iters', iters)
        for _ in range(n//2):
            parent1, parent2 = selection(population, fitnesses)
            offspring.extend(crossover(parent1, parent2))
        for x in offspring:
            if random.random() < mutation_probability:
                mutateSolution(x)
        population = offspring + population
        fitnesses = [evaluate(p, data_dict) for p in population]
        population = bestSelection(population, fitnesses, n)
        counter = counter + 1 if last_best_score == best_fitness else 1
        last_best_score = best_fitness
        if counter >= max_same_iters or not x:
            break
    fitnesses = [evaluate(p, data_dict) for p in population]
    best_fitness = sorted(fitnesses)[0]
    return (best_fitness, [x for y, x in sorted(zip(fitnesses, population))][0], iters)

mutate_prob = [0.1, 0.2, 0.4, 0.6, 0.8]
cross = [OXCrossover, PMXCrossover]
selection = [rouletteSelection, tournamentSelection]
population = [20, 40, 80, 100]

def genetic_helper(data, n, max_same_iters, evaluate, selection, crossover, mutation, mutation_probability):
    print(mutation_probability, crossover.__name__, selection.__name__, n)
    return ((mutation_probability, crossover.__name__, selection.__name__, n), genetic(data, n, max_same_iters, evaluate, selection, crossover, mutation, mutation_probability))

combs = list(itertools.product(mutate_prob, cross, selection, population))
for name, data in data_list.items():
    print(name)
    tmp = Parallel(n_jobs=-1)(delayed(genetic_helper)(data, n, 200, calcMultipleMachinesFitness, s, c, mutateSolution, m) for m, c, s, n in combs)
    print(tmp)
    print(tmp[0])
    dump(tmp, name + "_genetic.pickle")
'''
p1 = [1,4,2,8,5,7,3,6,9]
p2 = [7,5,3,1,9,8,6,4,2]
child_mine = cxPartialyMatched(p1, p2)
print(child_mine)
if len(child_mine[0]) == len(set(child_mine[0])):
    print("C1 pass")
else:
    print("C1 fail")
if len(child_mine[1]) == len(set(child_mine[1])):
    print("C2 pass")
else:
    print("C2 fail")

p1 = [1,2,3 , 4,5,6 , 7,8,9]
p2 = [2,4,1 , 7,8,3 , 9,5,6]
child_mine = cxPartialyMatched(p1, p2)
print(child_mine)
if len(child_mine[0]) == len(set(child_mine[0])):
    print("C1 pass")
else:
    print("C1 fail")
if len(child_mine[1]) == len(set(child_mine[1])):
    print("C2 pass")
else:
    print("C2 fail")

p1 = [9,2,7,5,4,3,6,1,8]
p2 = [2,8,3,6,9,5,7,4,1]
child_mine = cxPartialyMatched(p1, p2)
print(child_mine)
if len(child_mine[0]) == len(set(child_mine[0])):
    print("C1 pass")
else:
    print("C1 fail")
if len(child_mine[1]) == len(set(child_mine[1])):
    print("C2 pass")
else:
    print("C2 fail")

p1 = list(range(1, 10))
random.shuffle(p1)
p2 = list(range(1, 10))
random.shuffle(p2)
child_mine = cxPartialyMatched(p1, p2)
print(child_mine)
if len(child_mine[0]) == len(set(child_mine[0])):
    print("C1 pass")
else:
    print("C1 fail")
if len(child_mine[1]) == len(set(child_mine[1])):
    print("C2 pass")
else:
    print("C2 fail")

for _ in range(20):
    p1 = list(range(1, 51))
    random.shuffle(p1)
    p2 = list(range(1, 51))
    random.shuffle(p2)
    child_mine = OXCrossover(p1, p2)
    if len(child_mine[0]) == len(set(child_mine[0])):
        print("C1 pass")
    else:
        print("C1 fail")
    if len(child_mine[1]) == len(set(child_mine[1])):
        print("C2 pass")
    else:
        print("C2 fail")
'''