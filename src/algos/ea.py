from .population import *
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from ..problems.mpi import *
from ..utils.logger import Logger
import warnings

try: 
    import wandb
    use_wandb = True
except:
    use_wandb = False
    print("No WANDB")

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
        self.logger = Logger()
        self.logger.add_list(
            ["fitness", "evaluations", "generation"]
        )
        self.logger.add_list(
            ["validation fitness", "validation evaluations", "validation generation"]
        )
        self.logger.add_list(
            ["final fitness", "final validation fitness"]
        )
        self.wandb_run = None
        self.set_wandb(args["wandb"])

    def __len__(self):
        return len(self.pop)
    
    def __getitem__(self, key):
        return self.pop[key]
        
    def __repr__(self):
        s = f"EA: {self.total_evals}\n"
        for i in self.pop:
            s += f" - {i}\n"
        return s

    # Logging
    def set_wandb(self, project):
        if project =="":
            return
        if use_wandb:
            d = {
                "algo": self.__class__.__name__,
                "noise_name": f"""{self.args["noise_type"]} - {self.args["noise"]}""",
            }
            d = {**d, **(self.args)}
            self.wandb_run = wandb.init(
                project=project,
                config=d
            )
            print("wandb run:", wandb.run.name)
        else:
            warnings.warn("WANDB not installed")


    def close_wandb(self): 
        if self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None
    
    def log(self):
        # d = {
        #         "generations": self.gen,
        #     }
        d = self.logger.export()
        # d = {**d, **l}
        if self.wandb_run is not None: # pragma: no cover
            wandb.log(d)
        return d

    # Specific agents
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
    
    # Evaluation
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
    
    # Algorithm steps
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

    def get_validation(self, log=True):
        # Get elite
        self.pop.sort()
        elite = self.pop[0]
        # for i in self:
        #     assert elite.fitness >= i.fitness
        # Get validation fitness
        fit = np.mean(self.server.validation(elite, self.args["val_size"]))
        if log:
            self.logger("validation evaluations", self.total_evals)
            self.logger("validation fitness", fit)
            self.logger("validation generation", self.gen)
        return fit
    
    def run(self, name=""):
        pbar = tqdm(total=self.args["total_evals"])
        pbar.set_description(name)
        while self.total_evals < self.args["total_evals"]:
            self.step()
            f = np.max([np.mean(i.true_fitnesses) for i in self])
            pbar.update(self.total_evals - pbar.n)
            self.logger("evaluations", self.total_evals)
            self.logger("fitness", f)
            self.logger("generation", self.gen)
            # Add validation step
            if self.args["val_freq"]> 0 and self.gen % self.args["val_freq"] == 0:
                self.get_validation(log=True)
            # val_f = self.logger.last(key="validation fitness") or 0
            # pbar.set_description(name + f" -> {val_f:.2f}")
            if self.gen % self.args["log_freq"] == 0:
                self.log()
        f = np.max([np.mean(i.true_fitnesses) for i in self])
        self.logger("final fitness", f)
        fit = self.get_validation(log=(self.args["val_freq"]> 0))
        self.logger("final validation fitness", fit)
        self.log()
        pbar.close()
        self.close_wandb()
        return self.logger["evaluations"], self.logger["fitness"]

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