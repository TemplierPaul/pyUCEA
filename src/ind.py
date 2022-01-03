import numpy as np

class Ind:
    def __init__(self, args, genome=None):
        self.args = args
        self.genome = genome
        self.fitnesses = []
        self.true_fitnesses = []
        self.lifetime = 0
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
        
    def random(self):
        self.genome=np.random.randn(self.args["n_genes"])
        return self
    
    def mutate(self):
        noise = np.random.randn(self.args["n_genes"]) * self.args["mut_size"]
        new_g = self.genome + noise
        new_i = Ind(self.args, genome=new_g)
        return new_i

