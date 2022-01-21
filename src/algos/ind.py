import numpy as np
from copy import deepcopy
import os
import errno

def create_path(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

class Ind:
    def __init__(self, args, genome=None, path=None):
        if path is not None:
            self.load(path)
            return
        self.args = args
        self.genome = genome
        self.fitnesses = []
        self.true_fitnesses = []
        self.lifetime = 0
        self.u_g_0 = 0
        self.beta = np.inf
        
    @property
    def fitness(self):
        return np.mean(self.fitnesses) if len(self.fitnesses) > 0 else - np.inf
    
    @property
    def n_evals(self):
        return len(self.fitnesses)
        
    def __repr__(self):
        return f'Ind ({self.args["n_genes"]}) | {self.fitness:.2} / {self.n_evals}'
    
    def __str__(self):
        return self.__repr__()   

    def reset_fitness(self):
        self.fitnesses = []
        self.true_fitnesses = []
        self.lifetime = 0
        return self
        
    def random(self, pb=None):
        if pb is None:
            self.genome=np.random.randn(self.args["n_genes"])
        else:
            self.genome=pb.random_genome()
            assert len(self.genome) == self.args["n_genes"]
        return self
    
    def mutate(self):
        noise = np.random.randn(self.args["n_genes"]) * self.args["mut_size"]
        new_g = self.genome + noise
        new_i = Ind(self.args, genome=new_g)
        return new_i

    def save(self, path):
        d = {
            "genome": self.genome,
            "fitnesses": self.fitnesses,
            "true_fitnesses": self.true_fitnesses,
            "lifetime": self.lifetime,
            "beta": self.beta
        }
        # save compressed
        if not path.endswith(".npz"):
            path += ".npz"
        if "/" in path:
            create_path(path[:path.rfind("/")])
        np.savez_compressed(path, **d)
        return self

    def load(self, path):
        d = np.load(path)
        self.genome = d["genome"]
        self.fitnesses = d["fitnesses"]
        self.true_fitnesses = d["true_fitnesses"]
        self.lifetime = d["lifetime"]
        self.beta = d["beta"]
        return self


class BoolInd(Ind):

    def random(self, pb=None):
        self.genome = np.random.randint(2, size=10) * 1.0
        return self

    def mutate(self):
        new_g = deepcopy(self.genome)
        switch = np.random.rand(len(self.genome)) < 0.1
        new_g[switch] = 1.0 - new_g[switch]
        new_i = BoolInd(self.args, genome=new_g)
        return new_i
