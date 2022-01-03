from .ea import *

class UCEA(EA):
    def __init__(self, args, pb):
        super().__init__(args, pb)
        
    def __repr__(self):
        s = f"UCEA: {self.total_evals}\n"
        for i in self.pop:
            s += f" - {i}\n"
        return s      
    
    def step(self):
        self.update()
        self.evaluate_children()
        while self.gen_evals < self.args["max_eval"]:
            l, h = self.pop.get_limits()
            A, B = self.pop[l[0]], self.pop[h[0]]
            d = self.pop.dist(A, B)
            if d <= self.args["epsilon"]:
                break
            fA, noise_A = self.evaluate(A)
            self.pop.add_eval(A, fA, noise_A)
            fB, noise_B = self.evaluate(B)
            self.pop.add_eval(B, fB, noise_B)
            self.gen_evals += 2
            self.sorted=False
        self.pop.sort()
        return self