from src.run import *

# Create args parser
parser = argparse.ArgumentParser(description='Run EA on a given problem')
parser.add_argument('--problem', type=str, default='all_ones', help='Problem to run')
parser.add_argument('--max_fit', type=float, default=10, help='Max fitness')
parser.add_argument('--n_genes', type=int, default=10, help='Number of genes')
parser.add_argument('--gens', type=int, default=1000, help='Number of generations')
parser.add_argument('--noise', type=float, default=0, help='Noise level')
parser.add_argument('--normal_noise', default=False, help='Normal noise', action='store_true')
parser.add_argument('--n', type=int, default=1, help='Number of evaluations')

if __name__ == "__main__":
    args = parser.parse_args()
    basepb = PROBLEMS[args.problem]()
    print(args)
    run_xp(basepb, args.max_fit, args.n_genes, args.gens, args.normal_noise, args.noise, args.n)