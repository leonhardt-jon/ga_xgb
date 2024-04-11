import numpy as np 
import xgboost as xgb

def train_surrogate(population, fitness_values):
    x = np.array([individual for individual in population])
    y = np.array(fitness_values)
    model = xgb.XGBRegressor()
    model.fit(x, y) # no train/test split needed
    return model

class GA():
    def __init__(self, pop_size, generations, function, lb=-2, ub=2, n_vars = 2, oversize_mult = 3, n_best = 15):
        self.pop_size = pop_size 
        self.generations = generations 
        self.function = function
        self.lb = lb 
        self.ub = ub 
        self.n_vars = n_vars
        self.oversize_mult = oversize_mult
        self.n_best = n_best

    def run(self):
        population = self.initialize_population(self.pop_size)
        for gen in range(self.generations):
            scores = self.evaluate(population)
            sorted_pop =[x for _, x in sorted(zip(scores, population), key=lambda x: x[0]) ]
            if not gen == self.generations - 1:
                model = train_surrogate(population, scores)
                new_population = self.selection(sorted_pop, scores, len(population) * self.oversize_mult)
                surrogate_scores = model.predict(np.array(new_population))
                oversized_population = [x for _, x in sorted(zip(surrogate_scores, new_population), key=lambda x: x[0])]
                population = np.array(oversized_population)[:self.pop_size]
            print(sorted(scores)[0])
        res = sorted(scores)[0]
        res_individual = sorted_pop[0]
        return res, res_individual

    def initialize_population(self, pop_size):
        return [np.random.uniform(self.lb, self.ub, self.n_vars) for _ in range(pop_size)] 

    def evaluate(self, population):
        return [self.function(ind) for ind in population] 

    def mutate(self, parent):
        child = []
        for x in parent:
            if np.random.random() < 0.2:
                mut = x + np.random.normal(0, 1)
            else:
                mut = x
            child.append(mut)
        return np.array(child)

    def crossover(self, parent_1, parent_2):
        child = []
        for i, _ in enumerate(parent_1):
            if np.random.random() < 0.7:
                cx = np.random.random() * (parent_1[i] - parent_2[i])
            else:
                idx = np.random.randint(2)
                cx = parent_1[i] if idx == 0 else parent_2[i]
            child.append(cx)
        return np.array(child)

    def selection_wheel(self, parents):
        parent_indexes = [i for i in range(len(parents))]
        choices = np.random.choice(parent_indexes, size=2, replace=False)
        child = self.crossover(parents[choices[0]], parents[choices[1]]) 
        child = self.mutate(child)
        return child

    def selection(self, population, fitness_scores, pop_size):
        top_k = population[:self.n_best]
        new_pop = []
        while len(new_pop) < pop_size:
            child = self.selection_wheel(top_k)
            new_pop.append(child)
        return new_pop
