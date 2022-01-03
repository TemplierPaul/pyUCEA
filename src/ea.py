from .population import *
import numpy as np
import matplotlib.pyplot as plt

class EA:
    def __init__(self, args, pb):
        self.args = args
        self.pb = pb
        self.pop = Population(args).random()
        self.gen_evals = 0
        self.total_evals = 0
        
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
        self.gen_evals = 0
        self.sorted=False
        return self
    
    def evaluate_children(self):
        for i in self.pop:
            if len(i.fitnesses) == 0:
                f, noise = self.evaluate(i)
                self.pop.add_eval(i, f, noise)
                self.gen_evals += 1
        self.sorted=False
        self.pop.sort()
        return self
    
    def evaluate(self, agent):
        f, noise = self.pb.evaluate(agent.genome)
        self.total_evals += 1
        return f, noise
    
    def step(self):
        self.update()
        self.evaluate_children()
        N = 1
        for _ in range(N):
            for i in self.pop:
                f, noise = self.evaluate(i)
                self.pop.add_eval(i, f, noise)
                self.gen_evals += 1
        self.sorted=False
        self.pop.sort()
        return self


class MultiEA(EA):
    def step(self):
        self.update()
        self.evaluate_children()
        N = self.args["max_eval"] // self.args["n_pop"] # Distribute max evals between agents
        for _ in range(N):
            for i in self.pop:
                f, noise = self.evaluate(i)
                self.pop.add_eval(i, f, noise)
                self.gen_evals += 1
        self.sorted=False
        self.pop.sort()
        return self