from src.run import *

noise_levels = [0, 0.25, 0.5, 0.75, 1]
pb = RL("CartPole-v1")

for noise in noise_levels:
    run_xp(
        basepb=pb,
        max_fit=200,
        n_genes=pb.n_genes,
        gens=100,
        normal_noise=False,
        noise=noise,
        n_evals=2
    )

    run_xp(
        basepb=pb,
        max_fit=200,
        n_genes=pb.n_genes,
        gens=100,
        normal_noise=True,
        noise=noise,
        n_evals=2
    )
