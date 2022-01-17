from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
import numpy as np
from collections import OrderedDict
import sys
import time

def flush(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()

SERVER_NODE = 0

class Client:
    def __init__(self, pb):
        self.pb = pb
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank() # The process ID (integer 0-3 for 4-process run)
        self.size = self.comm.Get_size() # The number of processes

        # flush(f"Creating node {self.rank}\n")

    def __repr__(self):
        return f"Secondary {self.rank}"

    def __str__(self):
        return self.__repr__()

    def run(self):
        output = {
            "data": None,
            "rank": self.rank
        }
        # Send a message to the master
        self.comm.send(output, dest=0)
        while True:
            # Wait for a message from the master
            incoming = self.comm.recv(source=0)
            if incoming["stop"]:
                break

            # Do some work
            fit, noise = self.evaluate(
                incoming["data"]["genome"], 
                seed=incoming["data"]["seed"],
                eval=incoming["data"]["eval"]
                )
            output["data"] = {
                "fitness": fit,
                "noise": noise,
                "index": incoming["data"]["index"]
            }
            # Send the result back to the master
            self.comm.send(output, dest=0)


    def evaluate(self, genome, seed=-1, eval=False):
        # time.sleep(np.random.random())
        # flush(msg=f"{self.rank} evaluating {genome}\n")
        if eval:
            self.pb.eval()
        else:
            self.pb.train()
        f, noise = self.pb.evaluate(genome)
        return f, noise


class Server(Client):
    def __init__(self, pb):
        super().__init__(pb)
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank() # The process ID (integer 0-3 for 4-process run)
        assert self.rank == SERVER_NODE, f"Server must be rank {SERVER_NODE}"
        self.waitings = []

    def batch_evaluate(self, agents, **kwargs):
        if not isinstance(agents, (list, np.ndarray)):
            agents = [agents]
        seed = kwargs.get("seed", -1) # -1 means random seed
        if self.size == 1:
            for agent in agents:
                fit, noise = self.evaluate(agent.genome, seed=seed, eval=False)
                # print(fit, noise)
                agent.fitnesses.append(fit + noise)
                agent.true_fitnesses.append(fit)
            return agents

        to_complete = len(agents)

        # Send to waiting clients
        index = 0
        for i in self.waitings:
            # Send new agent to evaluate
            # unique ID
            
            d = {
                "data": {
                    "genome": agents[index].genome,
                    "index": index,
                    "seed":seed,
                    "eval": False
                },
                "stop": False,
            }
            self.comm.send(d, dest=i)
            index += 1
            if index == len(agents):
                break
        
        self.waitings=self.waitings[index:]

        while to_complete > 0:
            msg = self.comm.recv(source=ANY_SOURCE)
            if msg == "stop":
                break
            
            if msg["data"] is not None:
                agent_index = msg["data"]["index"]
                self.update_agent(agents[agent_index], msg["data"])
                to_complete -= 1

            if index < len(agents):
                # Send new agent to evaluate
                d = {
                    "data": {
                        "genome": agents[index].genome,
                        "index": index,
                        "seed":seed,
                        "eval": False
                    },
                    "stop": False,
                }
                self.comm.send(d, dest=msg["rank"])
                index += 1
            else:
                self.waitings.append(msg["rank"])
        
        return agents

    def validation(self, agent, n_evals=None,**kwargs):
        if n_evals is None:
            n_evals = self.size - 1
        fitnesses = np.zeros(shape=n_evals)
        seed = kwargs.get("seed", -1) # -1 means random seed
        if self.size == 1:
            for i in range(n_evals):
                fit, noise = self.evaluate(agent.genome, seed=seed, eval=True)
                fitnesses[i] = fit# + noise
            return fitnesses

        to_complete = n_evals

        # Send to waiting clients
        index = 0
        for i in self.waitings:
            # Send new agent to evaluate
            # unique ID
            
            d = {
                "data": {
                    "genome": agent.genome,
                    "index": index,
                    "seed":seed,
                    "eval": True
                },
                "stop": False,
            }
            self.comm.send(d, dest=i)
            index += 1
            if index == n_evals:
                break
        
        self.waitings=self.waitings[index:]

        while to_complete > 0:
            msg = self.comm.recv(source=ANY_SOURCE)
            if msg == "stop":
                break
            
            if msg["data"] is not None:
                agent_index = msg["data"]["index"]
                fitnesses[agent_index] = msg["data"]["fitness"]# + msg["data"]["noise"]
                to_complete -= 1

            if index < n_evals:
                # Send new agent to evaluate
                d = {
                    "data": {
                        "genome": agent.genome,
                        "index": index,
                        "seed":seed,
                        "eval": True
                    },
                    "stop": False,
                }
                self.comm.send(d, dest=msg["rank"])
                index += 1
            else:
                self.waitings.append(msg["rank"])

        return fitnesses

    def update_agent(self, agent, data):
        agent.fitnesses.append(data["fitness"] + data["noise"])
        agent.true_fitnesses.append(data["fitness"])
        return agent

    def stop(self):
        d = {
            "data": None,
            "stop": True,
        }
        for i in range(self.comm.Get_size()):
            if i != SERVER_NODE:
                self.comm.send(d, dest=i)
