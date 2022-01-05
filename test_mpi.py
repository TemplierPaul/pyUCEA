from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
import numpy as np
import time
from src.problems.mpi import Client, Server, flush
from src.algos.ind import Ind
from src.run import *

print("Hello")
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

pb = AllOnes()
pb = RL("CartPole-v1")

if rank == 0:
    # server
    server = Server(pb)

    args ={"n_genes":pb.n_genes}
    genomes = [np.random.random(pb.n_genes) * i for i in range(10)]
    pop = [Ind(args, genome=genome) for genome in genomes]
    pop = server.batch_evaluate(pop)
    time.sleep(1)
    for a in pop:
        flush(f"{a} -> {a.fitness}\n")

    server.stop()
else:
    # client
    client = Client(pb)
    client.run()