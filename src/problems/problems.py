import berl
import numpy as np

PROBLEMS={}

def register_pb(name):
    def wrapped(pb):
        PROBLEMS[name]=pb
        return pb
    return wrapped


class Problem:
    def __init__(self):
        pass
    
    def evaluate(self, genome):
        pass

@register_pb("all_ones")
class AllOnes(Problem):
    def __init__(self):
        self.name = "All_Ones"

    def evaluate(self, genome):
        n = len(genome)
        ones = np.ones(n)
        # mse 
        f = 10 - np.mean(np.square(genome - ones))
        # f = np.sum(- np.abs(genome - ones)) + 10
        return f, 0

class RL(Problem):
    def __init__(self, env):
        self.cfg = {
            "env":env,
            "noise_size":0,
            "reward_clip":False,
            "episode_frames":200,
            "c51":False,
            "stack_frames": 1
        }
        self.Net = berl.gym_flat_net(env, 10)
        self.n_genes = berl.get_genome_size(self.Net, c51=False)
        self.node = berl.Primary(self.Net, self.cfg)
        self.name = env
        
    def evaluate(self, genome, seed=-1):
        return self.node.evaluate(genome, seed=seed), 0

@register_pb("cartpole")
def f():
    return RL("CartPole-v0")


## Wrapper for noisy env

class Noisy:
    def __init__(self, pb, noise=0.5, max_fit=1, normal=True):
        self.pb = pb
        self.noise = noise
        self.max_fit=max_fit
        self.normal=normal
        
    def get_noise(self):
        if self.normal:
            return np.random.randn()
        else:
            return np.random.random() * 2 - 1
        
    def evaluate(self, genome, noisy=True):
        real_fit, _ = self.pb.evaluate(genome)
        if not noisy:
            return real_fit, 0
        noise = self.get_noise() * self.max_fit * self.noise 
        return real_fit, noise