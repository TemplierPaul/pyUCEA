from .problems import *

## Wrapper for noisy env: additional noise on the fitness
@register_pb("noise_fitness")
class NoisyFit:
    def __init__(self, pb, noise=0.5, normal=True, **kwargs):
        self.pb = pb
        self.noise = noise
        self.normal=normal
        self.pb.name = f"F_{self.pb.name}"

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
        real_fit, _ = self.pb.evaluate(genome, seed=0)
        if not noisy:
            return real_fit, 0
        noise = self.get_noise() * self.max_fit * self.noise 
        return real_fit, noise

# Wrapper for RL with noisy actions, but no additional noise on the fitness
@register_pb("noise_sticky")
class NoisySticky:
    def __init__(self, pb, noise=0.5, normal=True, **kwargs):
        self.pb = pb
        self.noise = noise
        self.normal=normal
        # Noisy action (replace function)
        self.pb.get_action = self.get_action
        self.pb.name = f"Y_{self.pb.name}"
        self.previous_action = None

    def __repr__(self):
        return f"Sticky ({int(self.noise*100)}%) {self.pb}"

    def __str__(self):
        return self.__repr__()

    def __getattr__(self, key):
        return self.pb.__getattribute__(key)

    def get_action(self, agent, obs):
        if self.previous_action != None and np.random.random() < self.noise:
            return self.previous_action
        if self.discrete:
            act = agent.act(obs)
        else:
            act = agent.continuous_act(obs)
        self.previous_action = act
        return act

    def evaluate(self, genome, noisy=True):
        return self.pb.evaluate(genome)

# Wrapper for RL with noisy actions, but no additional noise on the fitness
@register_pb("noise_action")
class NoisyAction:
    def __init__(self, pb, noise=0.5, normal=True, **kwargs):
        self.pb = pb
        self.noise = noise
        self.normal=normal
        # Noisy action (replace function)
        self.pb.get_action = self.get_action
        self.pb.name = f"A_{self.pb.name}"


    def __repr__(self):
        return f"Noisy ({int(self.noise*100)}%) {self.pb}"

    def __str__(self):
        return self.__repr__()

    def __getattr__(self, key):
        return self.pb.__getattribute__(key)

    def get_action(self, agent, obs):
        if self.discrete:
            return (
                np.random.randint(0, self.pb.action_space)
                if np.random.random() < self.noise
                else agent.act(obs)
            )

        action = agent.continuous_act(obs)
        h, l = self.action_space.high, self.action_space.low
        r = np.random.randn(self.action_space.shape[0]) * self.noise
        noise = r * (h-l)
        return action + noise

    def evaluate(self, genome, noisy=True):
        return self.pb.evaluate(genome)

# Wrapper for RL with noisy seed, but no additional noise on the fitness
@register_pb("noise_seed")
class NoisySeed:
    def __init__(self, pb, noise=0.5, normal=True, train_seeds=200, val_seeds=100000, **kwargs):
        self.pb = pb
        self.noise = noise
        self.normal=normal
        self.pb.name = f"S_{self.pb.name}"
        self.train_seeds = train_seeds
        self.val_seeds = val_seeds
        self.status = None

    def __repr__(self):
        return f"Noisy ({int(self.noise*100)}%) {self.pb}"

    def __str__(self):
        return self.__repr__()

    def __getattr__(self, key):
        return self.pb.__getattribute__(key)

    def train(self):
        self.status = "train"

    def eval(self):
        self.status = "eval"

    def evaluate(self, genome, noisy=True):
        if self.status == "train":
            seed = np.random.randint(0, self.train_seeds)
        else:
            # seed = np.random.randint(self.train_seeds, self.train_seeds + self.eval_seeds)
            seed = np.random.randint(0, self.val_seeds)

        return self.pb.evaluate(genome, seed=seed)
