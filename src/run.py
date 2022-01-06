from .algos.ea import *
from .algos.ucea import *
from .problems.problems import *
import sys

from .postproc.graphs import *
from .postproc.postprocessing import *
import pandas as pd
import numpy as np
import datetime

def multi_run(algo, cfg, pb, gens, n, name=""):
    X = [[] for _ in range(n)] # Evaluations
    Y = [[] for _ in range(n)] # Fitness
    for i in range(n):
        ea = algo(cfg, pb)
        # print(ea)
        X[i], Y[i] = ea.run(gens, name=name)
        ea.pop.plot()
    return X, Y


def run_xp(basepb, 
    gens=1000, 
    normal_noise=False,
    noise=0,
    n_evals=1
    ):
    max_fit = basepb.max_fit
    cfg = {
        "n_pop":12,
        "n_elites":6,
        "n_genes":basepb.n_genes, 
        "delta":0.1,
        "scaling_factor":max_fit,
        "epsilon":1,
        "mut_size":0.3,
        "max_eval":120,
        "n_resampling":10,
        "max_fit":max_fit,
        "noise_level": noise
    }

    algos = [EA, MultiEA, UCEA]
    noise_type='Normal' if normal_noise else 'Uniform'


    pb = Noisy(basepb, noise=noise, normal=normal_noise)

    gen_fitness = [[] for _ in algos]
    eval_fitness = [[] for _ in algos]
    evals = [[] for _ in algos]

    results = {}

    for algo in algos:
        a = algo.__name__
        X, Y = multi_run(algo, cfg, pb, gens, n_evals, name=f'{algo.__name__} | {noise}')
        path = f'saves/{noise_type}/Data_{a}_{basepb.name}_{int(noise*100)}'
        # add timestamp to path
        path += f'_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        save(X, Y, path)
        X, Y = load(path)
        results[a] = postprocessing(X, Y)
        
    title = f"{basepb.name} - {int(noise*100)}% noise ({noise_type})"
    # Fitness = f(gen)
    path = f"plots/{noise_type}/Gen_{basepb.name}_{int(noise*100)}.png"
    # gen_graph(results, title=title, save=path, max_val=max_fit)

    # Fitness = f(eval)
    path = f"plots/{noise_type}/Eval_{basepb.name}_{int(noise*100)}.png"
    eval_graph(results, title=title, save=path, max_val=max_fit)

    # Eval = f(gen)
    path = f"plots/{noise_type}/Cost_{basepb.name}_{int(noise*100)}.png"
    # cost_graph(results, title=title, save=path)

