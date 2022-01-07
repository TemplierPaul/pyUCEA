from src.run import *

from mpi4py import MPI
import argparse

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

# Create args parser
parser = argparse.ArgumentParser(description='Run EA on a given problem')
parser.add_argument('--problem', type=str, default='all_ones', help='Problem to run')
parser.add_argument('--n_pop', type=int, default=12, help='Number of agents')
parser.add_argument('--n_elites', type=int, default=6, help='Number of elites')
parser.add_argument('--mut_size', type=float, default=0.3, help='Mutation size')
parser.add_argument('--max_eval', type=int, default=120, help='Max evaluations')
parser.add_argument('--n_resampling', type=int, default=10, help='Number of resampling')

# Noise type
noise_types = [k.replace("noise_", "") for k in PROBLEMS.keys() if "noise_" in k]
parser.add_argument('--noise_type', type=str, default='fitness', choices=noise_types, help='Noise type: fitness / action / seed')
parser.add_argument('--noise', type=float, default=0, help='Noise level')
parser.add_argument('--normal_noise', default=False, help='Normal noise', action='store_true')

# UCEA
parser.add_argument('--delta', type=float, default=0.1, help='Delta')
parser.add_argument('--scaling_factor', type=float, default=1, help='Scaling factor')
parser.add_argument('--epsilon', type=float, default=1, help='Epsilon')

# Runs
parser.add_argument('--evals', type=int, dest="total_evals" ,default=1000, help='Number of generations')
parser.add_argument('--n', dest="n_evals", type=int, default=1, help='Number of evaluations')

# Add argument "algos" as list of values
parser.add_argument('--algos', type=str, nargs='+', default=['ea', "rs", "ucea"], help='Algorithm to run') 



if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    args = parser.parse_args()
    # flush(args)
    basepb = PROBLEMS[args.problem]()
    
    noise_wrapper = PROBLEMS[f"noise_{args.noise_type}"]
    pb = noise_wrapper(basepb, noise=args.noise, normal=args.normal_noise)

    args.n_genes = pb.n_genes
    args.max_fit = pb.max_fit
    args.bool_ind = pb.bool_ind

    if rank == 0:
        # flush("Main node - creating EA\n")
        try:
            server = Server(pb)
            run_xp(server, args)
        finally:
            server.stop()
    else:
        # flush(f"Creating node {rank}\n")
        client = Client(pb)
        client.run()