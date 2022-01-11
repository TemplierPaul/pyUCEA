from ..utils.state import * 
import torch.nn.functional as F


class Agent:
    def __init__(self, Net, config):
        self.config = config
        self.Net = Net 

        self.device = torch.device("cpu")

        self.model = None
        self.target = None

        self.fitness = None

        # State
        if self.config["stack_frames"] > 1:
            self.state = FrameStackState(self.config["obs_shape"], self.config["stack_frames"])
        else:
            self.state = State()

        self.criterion = torch.nn.MSELoss()
        

    def __repr__(self): # pragma: no cover
        return f"Agent {self.model} > fitness={self.fitness}" 
        
    def __str__(self): # pragma: no cover
        return self.__repr__()

    def make_network(self):
        self.model = self.Net(c51=False).to(self.device).double()
        return self

    @property
    def genes(self):
        if self.model is None:
            return None
        with torch.no_grad():
            params = self.model.parameters()
            vec = torch.nn.utils.parameters_to_vector(params)
        return vec.cpu().double().numpy()

    @genes.setter
    def genes(self, params):
        if self.model is None:
            self.make_network()
        assert len(params) == len(self.genes), "Genome size does not fit the network size"
        if np.isnan(params).any():
            raise
        a = torch.tensor(params, device=self.device)
        torch.nn.utils.vector_to_parameters(a, self.model.parameters())
        self.model = self.model.to(self.device).double()
        self.fitness = None
        return self

    def act(self, obs):
        self.state.update(obs)
        with torch.no_grad():
            x = self.state.get().to(self.device).double()
            actions = self.model(x).cpu().detach().numpy()
        return int(np.argmax(actions))

    def __eq__(self, other): 
        if type(other) != type(self): # pragma: no cover
            return False
        return (self.genes == other.genes).all()
