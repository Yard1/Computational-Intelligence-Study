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
#data = pd.read_csv("Zeszyt2.csv", sep=";", header=None)
#data = [tuple(r) for r in data.values]
data = pd.read_csv("Dane_S2_50_10.csv", sep=";", header=None)
data = [[r[0], list(r[1:])] for r in data.values]


def swapPositions(lst, pos1, pos2):
    lst[pos1], lst[pos2] = lst[pos2], lst[pos1]
    return lst


def calcSumOfDeviationSquares(lst):
    val = 0
    val_sq = 0
    for x in lst:
        val += x[1]
        val_sq += (x[2]-val)**2
    return val_sq


def calcMultipleMachinesFitness(lst):
    lst_cpy = deepcopy(lst)
    last_row = lst_cpy[0]
    for idx, x in enumerate(lst_cpy):
        if idx == 0:
            lst_cpy[idx][1] = addTimes(lst_cpy[idx][1], None)
        else:
            lst_cpy[idx][1] = addTimes(lst_cpy[idx][1], last_row[1])
        last_row = lst_cpy[idx]
    return last_row[1][-1]


def addTimes(lst, previous_lst):
    lst_cpy = deepcopy(lst)
    for idx, x in enumerate(lst_cpy):
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


def neh(data, evaluate):
    data_sorted = sorted(data, key=lambda x: sum(x[1]), reverse=True)
    solutions = [data_sorted[0]]
    for row in data_sorted[1:]:
        checked_row = row
        # print(checked_row)
        best_solutions = None
        best_fitness = -1
        for idx in range(len(solutions)+1):
            temp_solutions = solutions.copy()
            temp_solutions.insert(idx, checked_row)
            fitness = evaluate(temp_solutions)
            if best_fitness < 0 or fitness <= best_fitness:
                best_solutions = temp_solutions.copy()
                best_fitness = fitness
        solutions = best_solutions
    return (best_fitness, solutions)


def generateFirstSolution(data):
    data_copy = data.copy()
    # random.shuffle(data_copy)
    return data_copy


def neighborsGenerator(solution):
    for i in range(len(solution)):
        for j in range(len(solution)):
            if i != j:
                new_solution = solution.copy()
                swapPositions(new_solution, i, j)
                yield (new_solution, solution[i][0], solution[j][0])


def mutateSolution(solution):
    x0_idx = random.randint(0, len(data)-1)
    x1_idx = x0_idx
    while x1_idx == x0_idx:
        x1_idx = random.randint(0, len(data)-1)
    swapPositions(solution, x0_idx, x1_idx)


def reduceTemperature(temp, multiplier):
    return temp*multiplier


def probability(temperature, old_score, new_score):
    if new_score < old_score:
        return 1.0
    else:
        return Decimal(math.e)**Decimal(((old_score - new_score) / temperature))


def climb(data, evaluate, max_same_iters):
    best = generateFirstSolution(data)
    best_score = evaluate(best)
    last_best_score = best_score
    counter = 0

    while True:
        #print('Best score so far', best_score, 'Solution', best, 'Counter', counter)
        new_solution = best.copy()
        mutateSolution(new_solution)

        score = evaluate(new_solution)
        if score < best_score:
            best = new_solution
            best_score = score
        counter = counter + 1 if last_best_score == best_score else 1
        last_best_score = best_score
        if counter >= max_same_iters:
            break
    # print()
    return(best_score, best)


def simulated_annealing(data, evaluate, multiplier, max_same_iters, temperature_min=float_info.min):
    temperature = 1
    best = generateFirstSolution(data)
    s_best = best.copy()
    best_score = evaluate(best)
    last_best_score = best_score
    counter = 0

    while temperature > temperature_min:
        #print('Best score so far', best_score, '\tTemperature', temperature, '\tSolution', s_best)
        new_solution = best.copy()
        mutateSolution(new_solution)
        score = evaluate(new_solution)
        prob = probability(temperature, best_score, score)
        if prob > Decimal(random.uniform(0, 1)):
            best = new_solution
            if score < best_score:
                s_best = best.copy()
                best_score = score
        temperature = reduceTemperature(temperature, multiplier)
        counter = counter + 1 if last_best_score == best_score else 1
        last_best_score = best_score
        if counter >= max_same_iters:
            break
    # print()
    return(best_score, s_best)


def tabu(data, evaluate, cadence, max_same_iters):
    best = generateFirstSolution(data)
    s_best = best.copy()
    best_score = evaluate(best)
    tabu_list = dict()
    last_best_score = best_score
    counter = 0
    while True:
        #print('Best score so far', best_score, 'Solution', best)
        #print('Best score so far', best_score, '\tTabu list', tabu_list.keys())
        x = None
        y = None
        t_best_score = -1
        for t in neighborsGenerator(best):
            new_solution = t[0]
            t_score = evaluate(new_solution)
            if t_best_score == -1:
                t_best_score = t_score
            if (t_score < t_best_score and (t[1], t[2]) not in tabu_list and (t[2], t[1]) not in tabu_list):
                best = new_solution.copy()
                t_best_score = t_score
                x = t[1]
                y = t[2]
        if t_best_score < best_score:
            s_best = best.copy()
            best_score = t_best_score
        tabu_list[x, y] = None
        while len(tabu_list) > cadence:
            for k, _ in tabu_list.items():
                tabu_list.pop(k)
                break
        counter = counter + 1 if last_best_score == best_score else 1
        last_best_score = best_score
        if counter >= max_same_iters or not x:
            break
    print("DONE %d" % cadence)
    return(best_score, s_best)


def save_csv(data, fname, sep=";"):
    with open(fname, "w") as f:
        for x in data:
            f.write("%d;%d;%d\n" % (x[0], x[1], x[2]))
# print(len(data))

#s = neh(data, calcMultipleMachinesFitness)
#print(s[0], s[1])

#s = climb(data, calcMultipleMachinesFitness, 5000)
#print(s[0], s[1])

#s = tabu(data, calcMultipleMachinesFitness, 20, 50)
#print(s[0], s[1])

s = simulated_annealing(data, calcMultipleMachinesFitness, 0.99, 1000)
print(s[0], s[1])


'''
sa = {}

def simulated_annealing_parallel_helper(x, data, evaluate, multiplier):
    return (x, simulated_annealing(data, evaluate, multiplier, 500))


def tabu_parallel_helper(x, data, evaluate, cadance):
    print(x)
    return (x, tabu(data, evaluate, cadance, 20))

#tmp = Parallel(n_jobs=-1)(delayed(simulated_annealing_parallel_helper)(i, data, calcSumOfDeviationSquares, Decimal(i)) for i in tqdm([round(x,4) for x in np.linspace(0.9,0.99,10)]*30))
#dump(tmp, "sa.pickle")


tmp = Parallel(n_jobs=-1)(delayed(tabu_parallel_helper)(i, data,
                                                        calcSumOfDeviationSquares, i) for i in range(5, len(data)//10))
dump(tmp, "tabu.pickle")

tmp = load("sa.pickle")
tmp2 = {}
for k,v  in tmp:
	if k not in tmp2:
		tmp2[k] = [v]
	else:
		tmp2[k].append(v)

for k, v in tmp2.items():
	scores, solutions = zip(*v)
	min_sc = min(scores)
	#sa[k] = ((mean(scores), median(scores), min_sc))
	sa[k] = median(scores)
print(sa)

import matplotlib.pylab as plt
lists = sorted(sa.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
plt.plot(x, y)
plt.savefig("sa.png")
'''
