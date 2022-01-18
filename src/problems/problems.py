import numpy as np
from ..utils.models import get_genome_size, gym_flat_net, init_weights, min_conv, impala
from ..algos.rl_agent import Agent
from .env.env import make_env
from .env.minatar import MINATAR_ENVS
import torch

PROBLEMS={}
MINATAR_FRAMES = 2000
PROCGEN_FRAMES = 100000

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

    def random_genome(self):
        return np.random.randn(self.args["n_genes"])
    
    def evaluate(self, genome, **kwargs):
        pass

    def train(self):
        pass

    def eval(self):
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
        self.name = cfg["env"]
        self.max_fit = cfg["max_fit"]
        self.bool_ind = False

        env = make_env(self.config["env"])
        self.n_actions = env.action_space.n
        env.close()

    def __repr__(self):
        return f"RL - {self.name}"

    def random_genome(self):
        net = self.Net(c51=False).double()
        init_weights(net)
        with torch.no_grad():
            params = net.parameters()
            vec = torch.nn.utils.parameters_to_vector(params)
        return vec.cpu().double().numpy()

    def get_action(self, agent, obs):
        return agent.act(obs)

    def evaluate(self, genome, seed=0, render=False):
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

                if render:
                    env.render()

                total_r += r
                n_frames += 1

        finally:
            env.close()
        # self.frames += n_frames
        return total_r, 0

    def render(self, genome, seed=0, save_path=None):
        env = make_env(self.config["env"], seed=seed, render=True)
        if save_path is not None:
            env = gym.wrappers.Monitor(env, "./videos/cartpole", force=True)

        agent = self.make_agent(genome)

        agent.state.reset()

        try:
            obs = env.reset()
            n_frames = 0
            done = False

            while not done and n_frames < self.config["episode_frames"]:
                action = self.get_action(agent, obs)
                obs, r, done, _ = env.step(action)
                env.render()
                n_frames += 1

        finally:
            env.close()
        # self.frames += n_frames
        return 0

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
        "max_fit":200,
        # "c51":False,
        "stack_frames": 1,
        "net":gym_flat_net(game, 10)
    }
    pb = RL(cfg)
    return pb

@register_pb("min-breakout")
def f():
    game = "min-breakout"
    cfg = {
        "env":game,
        "episode_frames":MINATAR_FRAMES,
        "max_fit":None,
        # "c51":False,
        "stack_frames": 1,
        "net":min_conv(game)
    }
    pb = RL(cfg)
    return pb

@register_pb("min-si")
def f():
    game = "min-space_invaders"
    cfg = {
        "env":game,
        "episode_frames":MINATAR_FRAMES,
        "max_fit":None,
        # "c51":False,
        "stack_frames": 1,
        "net":min_conv(game)
    }
    pb = RL(cfg)
    return pb

@register_pb("min-asterix")
def f():
    game = "min-asterix"
    cfg = {
        "env":game,
        "episode_frames":MINATAR_FRAMES,
        "max_fit":None,
        # "c51":False,
        "stack_frames": 1,
        "net":min_conv(game)
    }
    pb = RL(cfg)
    return pb

@register_pb("min-freeway")
def f():
    game = "min-freeway"
    cfg = {
        "env":game,
        "episode_frames":MINATAR_FRAMES,
        "max_fit":None,
        # "c51":False,
        "stack_frames": 1,
        "net":min_conv(game)
    }
    pb = RL(cfg)
    return pb

@register_pb("min-seaquest")
def f():
    game = "min-seaquest"
    cfg = {
        "env":game,
        "episode_frames":MINATAR_FRAMES,
        "max_fit":None,
        # "c51":False,
        "stack_frames": 1,
        "net":min_conv(game)
    }
    pb = RL(cfg)
    return pb

@register_pb("bigfish")
def f():
    game = "bigfish"
    cfg = {
        "env": game,
        "episode_frames": PROCGEN_FRAMES,
        "max_fit": None,
        "stack_frames": 1,
        "net": impala(game)
    }
    pb = RL(cfg)
    return pb

@register_pb("bossfight")
def f():
    game = "bossfight"
    cfg = {
        "env": game,
        "episode_frames": PROCGEN_FRAMES,
        "max_fit": None,
        "stack_frames": 1,
        "net": impala(game)
    }
    pb = RL(cfg)
    return pb

@register_pb("coinrun")
def f():
    game = "coinrun"
    cfg = {
        "env": game,
        "episode_frames": PROCGEN_FRAMES,
        "max_fit": None,
        "stack_frames": 1,
        "net": impala(game)
    }
    pb = RL(cfg)
    return pb
