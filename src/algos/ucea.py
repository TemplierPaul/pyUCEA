from .ea import *

@register_algo("ucea")
class UCEA(EA):
    def __init__(self, args, pb):
        super().__init__(args, pb)
        
    def __repr__(self):
        s = f"UCEA: {self.total_evals}\n"
        for i in self.pop:
            s += f" - {i}\n"
        return s      
    
    def step(self):
        if self.gen > 0:
            self.pop.new_gen()
            self.update()
        gen_evals = self.evaluate_children()
        while gen_evals < self.args["max_eval"]:
            self.pop.update()
            l, h = self.pop.get_limits()
            A, B = self.pop[l], self.pop[h]
            d = self.pop.dist(A, B)
            if d <= self.pop.epsilon or (gen_evals + 2) >= self.args["max_eval"]:
                self.logger("elite_lower_fitness", B.fitness)
                self.logger("elite_lower_beta", B.beta)
                self.logger("rest_upper_fitness", A.fitness)
                self.logger("rest_upper_beta", A.beta)
                self.logger("distance", d)
                break

            A, B = self.server.batch_evaluate([A, B])
            gen_evals += 2
            for i in self.pop:
                i.lifetime += 2
        self.pop.sort()
        self.total_evals += gen_evals
        self.gen += 1
        return self
