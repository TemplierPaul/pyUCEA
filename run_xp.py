from src.run import *
from src.problems.problems import PROCGEN_NETS

from mpi4py import MPI
import argparse
import sys
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

# Create args parser
parser = argparse.ArgumentParser(description='Run EA on a given problem')
parser.add_argument('--problem', type=str, default='all_ones', help='Problem to run')
parser.add_argument('--n_pop', type=int, default=12, help='Number of agents')
parser.add_argument('--n_elites', type=int, default=6, help='Number of elites')
parser.add_argument('--mut_size', type=float, default=0.01, help='Mutation size')
parser.add_argument('--n_resampling', type=int, default=10, help='Number of resampling')
parser.add_argument('--tournament_fitness', type=str, default='mean', choices=['mean', 'median'], help='Function to use for determining tournmanet winner')

# Noise type
noise_types = [k.replace("noise_", "") for k in PROBLEMS.keys() if "noise_" in k]
parser.add_argument('--noise_type', type=str, default='fitness', choices=noise_types, help='Noise type: fitness / action / seed')
parser.add_argument('--noise', type=float, default=0, help='Noise level')
parser.add_argument('--normal_noise', default=False, help='Normal noise', action='store_true')
# Arguments for train / validation seed ranges
parser.add_argument('--train_seeds', type=int, default=200, help='Train max seed')
parser.add_argument('--val_seeds', type=int, default=1000, help='Validation max seed')
# Validation eval frequency
parser.add_argument('--val_freq', type=int, default=5, help='Validation eval frequency')
# Validation size
parser.add_argument('--val_size', type=int, default=10, help='Validation size')

# UCEA
parser.add_argument('--delta', type=float, default=0.1, help='Delta')
parser.add_argument('--scaling_factor', type=float, default=1.0, help='Scaling factor')
# Scaling type:
#   - constant: constant scaling factor (default)
#   - best: scaling factor is the best fitness overall
#   - last: scaling factor is the best fitness of the last generation
parser.add_argument('--scaling_type', type=str, default="constant", choices=["constant", "best", "last"], help='Scaling type')
parser.add_argument('--epsilon', type=float, default=1, help='Epsilon')
parser.add_argument('--max_eval', type=int, default=32, help='Max evaluations per generation in UCEA')

# Runs
parser.add_argument('--evals', type=int, dest="total_evals" ,default=1000, help='Number of evaluations')
parser.add_argument('--n', dest="n_evals", type=int, default=1, help='Number of evaluations')
parser.add_argument('--no_plot', default=False, help='Stop plot', action='store_true')
# Wandb project name
parser.add_argument('--wandb', type=str, default="", help='Wandb project name')
parser.add_argument('--job', type=str, default="", help='Job name')
parser.add_argument('--entity', default=None, help='Wandb entity name')
# Save freq and path
parser.add_argument('--save_freq', type=int, default=50, help='Save frequency')
parser.add_argument('--save_path', type=str, default="./genomes", help='Save path')
# Log frequency
parser.add_argument('--log_freq', type=int, default=1, help='Log frequency')

# Add argument "algos" as list of values
parser.add_argument('--algos', type=str, nargs='+', default=['ea', "rs", "ucea"], help='Algorithm to run') 
# Net from the NETWORKS dict
parser.add_argument('--net', type=str, default='impala', choices=PROCGEN_NETS.keys(), help='Network to use for Procgen')
# normalize
parser.add_argument('--net_norm', default=False, help='Use LayerNorm', action='store_true')

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    args = parser.parse_args()
    # flush(str(PROBLEMS.keys()) + "\n")
    basepb = PROBLEMS[args.problem](args)
    
    noise_wrapper = PROBLEMS[f"noise_{args.noise_type}"]
    pb = noise_wrapper(
        basepb, 
        noise=args.noise, 
        normal=args.normal_noise,
        train_seeds=args.train_seeds,
        val_seeds=args.val_seeds,
        )

    args.n_genes = pb.n_genes
    args.max_fit = pb.max_fit
    args.bool_ind = pb.bool_ind

    if rank == 0:
        argstring = ("Arguments " +
                     ' '.join([str(k) + "=" + str(getattr(args, k)) for k in vars(args)]) +
                     " \n")
        print(argstring)
        sys.stdout.flush()
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
