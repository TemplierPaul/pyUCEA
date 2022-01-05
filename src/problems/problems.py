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

    def __repr__(self):
        return f"All Ones - {self.n_genes} genes"

    def evaluate(self, genome, **kwargs):
        n = len(genome)
        ones = np.ones(n)
        # mse 
        f = 10 - np.mean(np.square(genome - ones))
        # f = np.sum(- np.abs(genome - ones)) + 10
        return f, 0

class RL(Problem):
    def __init__(self, env):
        self.config = {
            "env":env,
            # "noise_size":0,
            "reward_clip":False,
            "episode_frames":200,
            "c51":False,
            "stack_frames": 1
        }
        self.Net = berl.gym_flat_net(env, 10)
        self.n_genes = berl.get_genome_size(self.Net, c51=False)
        self.name = env
        self.max_fit = 200

    def __repr__(self):
        return f"RL - {self.name}"

    def evaluate(self, genome, seed=0, render=False, test=False):
        if seed < 0:
            seed = np.random.randint(0, 1000000000)
        seed=0
        env = berl.make_env(self.config["env"], seed=seed, render=render)
        agent = self.make_agent(genome)

        # Virtual batch normalization 
        # agent.model(self.vb)

        agent.state.reset()

        try:
            obs = env.reset()
            n_frames = 0
            total_r = 0
            done = False

            while not done and n_frames < self.config["episode_frames"]:
                action = agent.act(obs)
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
        i = berl.Agent(self.Net, self.config)
        if genome is not None:
            i.genes = genome
        return i

@register_pb("cartpole")
def f():
    pb = RL("CartPole-v1")
    return pb


## Wrapper for noisy env

class Noisy:
    def __init__(self, pb, noise=0.5, normal=True):
        self.pb = pb
        self.noise = noise
        self.max_fit=self.pb.max_fit
        self.normal=normal

    def __repr__(self):
        return f"Noisy ({int(self.noise*100)}%) {self.pb}"

    def __str__(self):
        return self.__repr__()
        
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