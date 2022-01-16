from .population import *
from ..problems.mpi import *
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
import logging

ALGOS = {}

# register class as an algorithm
def register_algo(name):
    def wrapped(cls):
        ALGOS[name] = cls
        return cls
    return wrapped

@register_algo("ea")
class EA:
    def __init__(self, args, server):
        self.args = args
        self.server = server
        self.pop = Population(args).random()
        self.total_evals = 0
        self.gen=0
        self.best_ind = None
        self.best_fit = -np.Inf
        
    def __len__(self):
        return len(self.pop)
    
    def __getitem__(self, key):
        return self.pop[key]
        
    def __repr__(self):
        s = f"EA: {self.total_evals}\n"
        for i in self.pop:
            s += f" - {i}\n"
        return s
    
    def get_elites(self):
        self.pop.agents = self.pop.split()[1]
        return self
    
    def get_parent(self):
        candidates = np.random.choice(self.pop.agents, 3)
        best = np.argmax([i.fitness for i in candidates])
        return candidates[best]

    def update(self):
        children = []
        for _ in range(self.args["n_pop"]-len(self)):
            parent = self.get_parent()
            children.append(parent.mutate())
        self.get_elites()
        self.pop.add(children)
        self.pop.sorted=False
        return self

    def evaluate_children(self):
        children = [i for i in self.pop if len(i.fitnesses) == 0]
        self.server.batch_evaluate(children, seed_min=0, seed_max=self.args["evo_seed_max"])
        self.total_evals += len(children)
        for i in self.pop.agents:
            i.lifetime += len(children)
        self.pop.sorted=False
        self.pop.update()
        self.pop.sort()
        return len(children)

    def update_best(self):
        self.pop.sort()
        best = self.pop[0]
        if best.fitness > self.best_fit:
            self.best_ind = deepcopy(best)
            self.best_fit = best.fitness
        return self

    def evaluate_elite(self):
        eval_inds = [deepcopy(self.best_ind) for i in range(self.args["n_test_evals"])]
        for i in eval_inds:
            i.reset_fitness()
        self.server.batch_evaluate(eval_inds, seed_min=self.args["evo_seed_max"], seed_max=100000)
        mean_fit = np.mean([i.fitness for i in eval_inds])
        self.logger.info(f"{self.gen},{self.total_evals},{mean_fit}")

    def step(self):
        if self.gen > 0:
            self.update()
        # n_children = self.evaluate_children()
        for i in self.pop:
            i.reset_fitness()
        self.server.batch_evaluate(self.pop.agents)
        self.total_evals += len(self.pop.agents)
        self.pop.sorted=False
        self.pop.sort()
        self.gen += 1
        return self
    
    def run(self, name=""):
        pbar = tqdm(total=self.args["total_evals"])
        pbar.set_description(name)
        X, Y = [], []
        eval_interval = self.args["total_evals"] / self.args["test_eval_interval"]
        eval_check = eval_interval
        while self.total_evals < self.args["total_evals"]:
            self.step()
            f = np.max([np.mean(i.true_fitnesses) for i in self])
            pbar.update(self.total_evals - pbar.n)
            X.append(self.total_evals)
            Y.append(f)
            if self.total_evals > eval_check:
                self.update_best()
                self.evaluate_elite()
                eval_check += eval_interval
        pbar.close()
        return X, Y

@register_algo("rs")
class MultiEA(EA):
    def step(self):
        if self.gen > 0:
            self.update()
        
        eval_agents = [i for i in self.pop.agents if i.n_evals == 0] * self.args["n_resampling"]
        self.server.batch_evaluate(eval_agents)
        self.total_evals += len(eval_agents)
        self.pop.sorted=False
        self.pop.sort()
        self.gen += 1
        return self
