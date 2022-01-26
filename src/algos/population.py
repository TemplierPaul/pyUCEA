from .ind import *
import matplotlib.pyplot as plt
import numpy as np


class Population:
    def __init__(self, args):
        self.args = args
        self.agents = np.array([])
        self.sorted = False
        self.ind_type = BoolInd if args["bool_ind"] else Ind
        self.scaling_factor = args["scaling_factor"] or 1.0
        self.epsilon = args["epsilon"] or 0.1
        if not args["epsilon_fixed"]:
            self.epsilon = self.epsilon * self.scaling_factor
        
    def __repr__(self):
        s = "Population:\n"
        for i in self.agents:
            s += f" - {i}\n"
        return s
    
    def __len__(self):
        return len(self.agents)
        
    def random(self, pb):
        self.agents = np.array([self.ind_type(self.args).random(pb) for _ in range(self.args["n_pop"])])
        self.sorted = False
        return self
        
    def sort(self):
        if not self.sorted:
            f = np.argsort([-i.fitness for i in self.agents]) # highest fitness first
            self.agents = self.agents[f] # highest fitness first
            self.sorted = True
        return self
        
    def __getitem__(self, key):
        return self.agents[key]
    
    def set_bounds(self, ind):
        u = ind.n_evals
        u_0 = ind.u_g_0
        t = ind.lifetime / 2 
        d = self.args["delta"]
        n = len(self.agents)
        k = 1.2
        a = 2
        # ind.beta = self.args["scaling_factor"] * np.sqrt(
        #     np.log(1.25 * n * (t ** 4) / d) / (2.0 * u)
        # )
        ind.beta = self.scaling_factor * np.sqrt(
            np.log(
                n * k * (t**a) * (u_0 + t) / d
            ) / (2.0 * u)
        )
        return ind

    def new_gen(self):
        if self.args["scaling_type"] == "last": # Reset scaling factor to forget previous generations
            self.scaling_factor = 0.0
        max_fit = -np.Inf
        min_fit = np.Inf
        for i in self:
            i.lifetime = 0
            i.u_g_0 = i.n_evals
            if len(i.fitnesses) > 0:
                max_fit = max(max_fit, i.fitness)
                min_fit = min(min_fit, i.fitness)
        if self.args["scaling_type"] == "scale":
            self.scaling_factor = max_fit - min_fit
        elif self.args["scaling_type"] != "constant":
            self.scaling_factor = max(self.scaling_factor, max_fit)
        if not self.args["epsilon_fixed"]:
            self.epsilon = self.args["epsilon"] * self.scaling_factor
        return self

    
    def update(self):
        for i in self.agents:
            self.set_bounds(i)
        self.sorted = False
        return self
    
    def split(self):
        self.sort()
        high = self.agents[:self.args["n_elites"]]
        low = self.agents[self.args["n_elites"]:]
        return low, high
    
    def get_limits(self):
        l, h = self.split()
        h_limit = np.argsort([i.fitness - i.beta for i in h]) # lowest of the best first
        l_limit = np.argsort([-1*(i.fitness + i.beta) for i in l]) + self.args["n_elites"] # highest of the others first
        return l_limit, h_limit
    
    def plot(self, real_fit=None):
        plt.figure(figsize=(16, 8))
        plt.subplot(121) 
        self.sort()
        for k in range(len(self)):
            i = self[k]
            plt.scatter(k, i.fitness, c="b")
            if real_fit is not None:
                plt.scatter(k, real_fit[i], c="r")
            plt.errorbar(k, i.fitness, i.beta, c="b")
        plt.title("Population")
        
        plt.subplot(122, frameon=False)
        evals = [i.n_evals for i in self]
        plt.bar(range(len(self)), evals)
        plt.title(f"Evaluations - {np.sum(evals)}")
        plt.show()
        
    def dist(self, low, high):
        return (low.fitness + low.beta) - (high.fitness - high.beta)
    
    def add(self, children):
        assert isinstance(children, (list, np.array))
        children = np.array(children)
        self.agents = np.concatenate([self.agents, children])
        self.sorted = False

    def save(self, path):
        self.sort()
        for i in range(len(self)):
            self[i].save(f"{path}/{i}.npz")
        return self

    def load(self, path):
        self.agents = np.array(
            [self.ind_type(self.args).load(f"{path}/{i}.npz") for i in range(self.args["n_pop"])]
            )
        self.sorted = False
        return self
