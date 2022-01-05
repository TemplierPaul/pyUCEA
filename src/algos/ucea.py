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
        gen_evals = 0
        self.update()
        self.evaluate_children()
        while gen_evals < self.args["max_eval"]:
            l, h = self.pop.get_limits()
            A, B = self.pop[l[0]], self.pop[h[0]]
            d = self.pop.dist(A, B)
            if d <= self.args["epsilon"]:
                break

            A, B = self.server.batch_evaluate([A, B])
            gen_evals += 2
            A.lifetime += 2
            B.lifetime += 2
            self.pop.sorted=False
        self.pop.sort()
        self.total_evals += gen_evals
        return self