import numpy as np
from ..utils.models import get_genome_size, gym_flat_net, init_weights, impala, gym_conv, gym_conv_efficient, cont_conv #, min_conv
from ..algos.rl_agent import Agent
from .env.env import make_env
from .env.minatar import MINATAR_ENVS
import torch

PROBLEMS={}
MINATAR_FRAMES = 2000
PROCGEN_FRAMES = 100000
PROCGEN_NETS = {
    "impala": impala,
    "conv": gym_conv,
    "efficient": gym_conv_efficient,
}

def register_pb(name):
    def wrapped(pb):
        PROBLEMS[name]=pb
        return pb
    return wrapped


class Problem:
    def __init__(self, cfg):
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
    def __init__(self, name, cfg):
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
    def __init__(self, name, cfg):
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
    def __init__(self, name, cfg):
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
        self.n_genes = get_genome_size(self.Net)
        self.name = cfg["env"]
        self.max_fit = cfg["max_fit"]
        self.bool_ind = False
        self.discrete = cfg["discrete"]

        env = make_env(self.config["env"])
        self.action_space = env.action_space.n if self.discrete else env.action_space
        env.close()

    def __repr__(self):
        return f"RL - {self.name}"

    def random_genome(self):
        net = self.Net().double()
        init_weights(net)
        with torch.no_grad():
            params = net.parameters()
            vec = torch.nn.utils.parameters_to_vector(params)
        return vec.cpu().double().numpy()

    def get_action(self, agent, obs):
        return agent.act(obs) if self.discrete else agent.continuous_act(obs)

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
            env = gym.wrappers.Monitor(env, save_path, force=True)

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
def f_cartpole(name, args):
    game = "CartPole-v1"
    cfg = {
        "env":game,
        "episode_frames":200,
        "max_fit":200,
        "stack_frames": 1,
        "discrete":True,
        "net":gym_flat_net(game, 10)
    }
    pb = RL(cfg)
    return pb

@register_pb("carracing")
def f_carracing(name, args):
    game = "CarRacing-v0"
    cfg = {
        "env":game,
        "episode_frames":200,
        "max_fit":1000,
        "stack_frames": 1,
        "discrete":False,
        "net":cont_conv(game, h_size=64, norm=False)
    }
    pb = RL(cfg)
    return pb

@register_pb("ant")
def f_carracing(name, args):
    game = "Ant-v2"
    cfg = {
        "env":game,
        "episode_frames":1000,
        "max_fit":1000,
        "stack_frames": 1,
        "discrete":False,
        "net":gym_flat_net(game, 128)
    }
    pb = RL(cfg)
    return pb

@register_pb("humanoid")
def f_carracing(name, args):
    game = "Humanoid-v2"
    cfg = {
        "env":game,
        "episode_frames":1000,
        "max_fit":1000,
        "stack_frames": 1,
        "discrete":False,
        "net":gym_flat_net(game, 128)
    }
    pb = RL(cfg)
    return pb

@register_pb("standup")
def f_carracing(name, args):
    game = "HumanoidStandup-v2"
    cfg = {
        "env":game,
        "episode_frames":1000,
        "max_fit":1000,
        "stack_frames": 1,
        "discrete":False,
        "net":gym_flat_net(game, 128)
    }
    pb = RL(cfg)
    return pb

def procgen_pb(g, args):
    cfg = {
        "env": g,
        "episode_frames": PROCGEN_FRAMES,
        "max_fit": None,
        "stack_frames": 1,
        "discrete":True,
        "net": PROCGEN_NETS[args.net](g, norm=args.net_norm)
    }
    pb = RL(cfg)
    return pb


PROCGEN_NAMES = [
    "bigfish",
    "bossfight",
    "caveflyer",
    "chaser",
    "climber",
    "coinrun",
    "dodgeball",
    "fruitbot",
    "heist",
    "jumper",
    "leaper",
    "maze",
    "miner",
    "ninja",
    "plunder",
    "starpilot"
]

for game in PROCGEN_NAMES:
    PROBLEMS[game] = procgen_pb
