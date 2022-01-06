from .population import *
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from ..problems.mpi import *

class EA:
    def __init__(self, args, pb):
        self.args = args
        self.server = Server(pb)
        self.pop = Population(args).random()
        self.total_evals = 0
        self.gen=0
        
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
        self.get_elites()
        children = []
        for _ in range(self.args["n_pop"]-len(self)):
            parent = self.get_parent()
            children.append(parent.mutate())
        self.pop.add(children)
        self.pop.sorted=False
        return self
    
    def evaluate_children(self):
        children = [i for i in self.pop if len(i.fitnesses) == 0]
        self.server.batch_evaluate(children)
        self.total_evals += len(children)
        for i in self.pop.agents:
            i.lifetime += len(children)
        self.pop.sorted=False
        self.pop.update()
        self.pop.sort()
        return len(children)
    
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
    
    def run(self, gens, name=""):
        max_evals = 20000
        pbar = tqdm(total=max_evals)
        pbar.set_description(name)
        X, Y = [], []
        while self.total_evals < max_evals:
            self.step()
            f = np.max([np.mean(i.true_fitnesses) for i in self])
            pbar.update(self.total_evals - pbar.n)
            X.append(self.total_evals)
            Y.append(f)
        pbar.close()
        return X, Y

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