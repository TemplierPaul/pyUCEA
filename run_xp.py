from src.run import *

from mpi4py import MPI
import argparse

# Create args parser
parser = argparse.ArgumentParser(description='Run EA on a given problem')
parser.add_argument('--problem', type=str, default='all_ones', help='Problem to run')
parser.add_argument('--gens', type=int, default=1000, help='Number of generations')
parser.add_argument('--noise', type=float, default=0, help='Noise level')
parser.add_argument('--normal_noise', default=False, help='Normal noise', action='store_true')
parser.add_argument('--n', type=int, default=1, help='Number of evaluations')

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    args = parser.parse_args()
    basepb = PROBLEMS[args.problem]()

    print(args)

    if rank == 0:
        flush("Main node - creating EA\n")
        run_xp(basepb, args.gens, args.normal_noise, args.noise, args.n)
    else:
        flush(f"Creating node {rank}\n")
        pb = Noisy(basepb, noise=args.noise, normal=args.normal_noise)
        client = Client(pb)
        client.run()