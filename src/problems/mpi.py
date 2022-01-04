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
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank() # The process ID (integer 0-3 for 4-process run)
        self.size = self.comm.Get_size() # The number of processes

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
            fit, noise = self.evaluate(incoming["data"]["genome"], seed=incoming["data"]["seed"])
            output["data"] = {
                "fitness": fit,
                "noise": noise,
                "index": incoming["data"]["index"]
            }
            # Send the result back to the master
            self.comm.send(output, dest=0)


    def evaluate(self, genome, seed=-1):
        time.sleep(np.random.random())
        return genome[0], 0

class Server(Client):
    def __init__(self):
        super().__init__()
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank() # The process ID (integer 0-3 for 4-process run)
        assert self.rank == SERVER_NODE, f"Server must be rank {SERVER_NODE}"
        self.waitings = []

    def batch_evaluate(self, agents, seed=-1):
        if not isinstance(agents, list):
            agents = [agents]

        if self.size == 1:
            for agent in agents:
                fit, noise = self.evaluate(agent.genome, seed=seed)
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
                    "seed":seed
                },
                "stop": False,
            }
            self.comm.send(d, dest=i)
            index += 1
        
        self.waitings=[]

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
                        "seed":seed
                    },
                    "stop": False,
                }
                self.comm.send(d, dest=msg["rank"])
                index += 1
            else:
                self.waitings.append(msg["rank"])
        
        return agents


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
