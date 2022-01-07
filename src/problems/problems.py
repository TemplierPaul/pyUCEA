import numpy as np
from ..utils.models import get_genome_size, gym_flat_net
from ..algos.rl_agent import Agent
from .env.env import make_env

PROBLEMS={}

def register_pb(name):
    def wrapped(pb):
        PROBLEMS[name]=pb
        return pb
    return wrapped


class Problem:
    def __init__(self):
        pass

    def __repr__(self):
        return "Base problem"

    def __str__(self):
        return self.__repr__()
    
    def evaluate(self, genome, **kwargs):
        pass


@register_pb("all_ones")
class AllOnes(Problem):
    def __init__(self):
        self.name = "All_Ones"
        self.n_genes = 10
        self.max_fit = 10
        self.bool_ind = True

    def __repr__(self):
        return f"All Ones - {self.n_genes} genes"

    def evaluate(self, genome, **kwargs):
        genome = np.round(np.clip(genome, 0.0, 1.0))
        f = np.sum(genome)
        return f, 0

@register_pb("float_all_ones")
class AllOnes(Problem):
    def __init__(self):
        self.name = "Float_All_Ones"
        self.n_genes = 10
        self.max_fit = 10
        self.bool_ind = False

    def __repr__(self):
        return f"Float All Ones - {self.n_genes} genes"

    def evaluate(self, genome, **kwargs):
        f = 10-np.mean(np.square(genome-np.ones(self.n_genes)))
        return f, 0

@register_pb("leading_ones")
class LeadingOnes(Problem):
    def __init__(self):
        self.name = "Leading_Ones"
        self.n_genes = 10
        self.max_fit = 10
        self.bool_ind = True

    def __repr__(self):
        return f"Leading Ones - {self.n_genes} genes"

    def evaluate(self, genome, **kwargs):
        genome = np.round(genome)
        if np.sum(genome) == len(genome):
            f = len(genome)
        else:
            f = np.argmin(genome)
        return f, 0


class RL(Problem):
    def __init__(self, cfg):
        self.config = cfg
        self.Net = cfg["net"]
        self.n_genes = get_genome_size(self.Net, c51=False)
        self.name = env
        self.max_fit = 200
        self.bool_ind = False

        env = make_env(self.config["env"], seed=seed, render=render)
        self.n_actions = env.action_space.n
        env.close()

    def __repr__(self):
        return f"RL - {self.name}"

    def get_action(self, agent, obs):
        return agent.act(obs)

    def evaluate(self, genome, seed=0, render=False, test=False):
        if seed < 0:
            seed = np.random.randint(0, 1000000000)
        seed=0
        env = make_env(self.config["env"], seed=seed, render=render)
        agent = self.make_agent(genome)

        agent.state.reset()

        try:
            obs = env.reset()
            n_frames = 0
            total_r = 0
            done = False

            while not done and n_frames < self.config["episode_frames"]:
                action = self.get_action(agent, obs)
                obs, r, done, _ = env.step(action)

                if self.config["reward_clip"]>0:
                    r = max(min(r, self.config["reward_clip"]), -self.config["reward_clip"])

                if render:
                    env.render()

                total_r += r
                n_frames += 1

        finally:
            env.close()
        # self.frames += n_frames
        return total_r, 0

    def make_agent(self, genome=None):
        i = Agent(self.Net, self.config)
        if genome is not None:
            i.genes = genome
        return i

@register_pb("cartpole")
def f():
    game = "CartPole-v1"
    cfg = {
        "env":game,
        "episode_frames":200,
        "reward_clip":False,
        "c51":False,
        "stack_frames": 1,
        "net":gym_flat_net(game, 10)
    }
    pb = RL(cfg)
    return pb


## Wrapper for noisy env: additional noise on the fitness
@register_pb("noise_fitness")
class NoisyFit:
    def __init__(self, pb, noise=0.5, normal=True):
        self.pb = pb
        self.noise = noise
        self.normal=normal

    def __repr__(self):
        return f"Noisy ({int(self.noise*100)}%) {self.pb}"

    def __str__(self):
        return self.__repr__()

    def __getattr__(self, key):
        return self.pb.__getattribute__(key)
        
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

# Wrapper for RL with noisy actions, but no additional noise on the fitness
@register_pb("noise_action")
class NoisyAction:
    def __init__(self, pb, noise=0.5, normal=True):
        self.pb = pb
        self.noise = noise
        self.normal=normal
        # Noisy action (replace function)
        self.pb.get_action = self.get_action

    def __repr__(self):
        return f"Noisy ({int(self.noise*100)}%) {self.pb}"

    def __str__(self):
        return self.__repr__()

    def __getattr__(self, key):
        return self.pb.__getattribute__(key)
        
    def get_action(self, agent, obs):
        if np.random.random() < self.noise:
            return np.random.randint(0, self.pb.n_actions)
        else:
            return agent.act(obs)

    def evaluate(self, genome, noisy=True):
        return self.pb.evaluate(genome)