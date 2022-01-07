from .problems import *

## Wrapper for noisy env: additional noise on the fitness
@register_pb("noise_fitness")
class NoisyFit:
    def __init__(self, pb, noise=0.5, normal=True):
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
@register_pb("noise_action")
class NoisyAction:
    def __init__(self, pb, noise=0.5, normal=True):
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
        if np.random.random() < self.noise:
            return np.random.randint(0, self.pb.n_actions)
        else:
            return agent.act(obs)

    def evaluate(self, genome, noisy=True):
        return self.pb.evaluate(genome)

# Wrapper for RL with noisy actions, but no additional noise on the fitness
@register_pb("noise_seed")
class NoisySeed:
    def __init__(self, pb, noise=0.5, normal=True):
        self.pb = pb
        self.noise = noise
        self.normal=normal
        self.pb.name = f"S_{self.pb.name}"

    def __repr__(self):
        return f"Noisy ({int(self.noise*100)}%) {self.pb}"

    def __str__(self):
        return self.__repr__()

    def __getattr__(self, key):
        return self.pb.__getattribute__(key)

    def evaluate(self, genome, noisy=True):
        seed = np.random.randint(0, 100000000)
        return self.pb.evaluate(genome, seed=seed)