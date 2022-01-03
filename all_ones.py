from src.run import *

noise_levels = [0, 0.25, 0.5, 0.75, 1]

for noise in noise_levels:
    run_xp(
        basepb=AllOnes(),
        max_fit=10,
        n_genes=10,
        gens=500,
        normal_noise=False,
        noise=noise,
        n_evals=2
    )

    # run_xp(
    #     basepb=AllOnes(),
    #     max_fit=10,
    #     n_genes=10,
    #     gens=1000,
    #     normal_noise=True,
    #     noise=noise,
    #     n_evals=1
    # )

