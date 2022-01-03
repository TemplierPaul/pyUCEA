from .ea import *
from .ucea import *
from .env import *
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

from scipy import interpolate

def aggregate(X, Y):
    X = np.array(X)
    Y = np.array(Y)
    F = [interpolate.interp1d(X[i], Y[i]) for i in range(len(X))] # Compute interpolation function
    
    # Compute bounds
    max_evals = min([x[-1] for x in X])
    min_evals = max([x[0] for x in X])
    
    # Create as many points
    n = len(X[0])
    step = (max_evals-min_evals)/n
    new_X = np.arange(min_evals, max_evals+1, step)
    new_X[-1]=int(new_X[-1])

    # Interpolate new X
    try:
        new_Y = [f(new_X) for f in F]
    except:
        for k in range(len(F)):
            f = F[k]
            for x in new_X:
                try:
                    f(x)
                except:
                    print("Failing:", k, x)
    
    # Aggregate
    # mean_Y = np.mean(new_Y, axis=0)
    
    return new_X, new_Y


def multi_run(algo, cfg, pb, gens, n, name=""):
    X = [[] for _ in range(n)]
    Y = [[] for _ in range(n)]
    for i in range(n):
        ea = algo(cfg, pb)
        pbar = tqdm(range(gens))
        pbar.set_description(name)
        for gen in pbar:
            ea.step()
            f = np.mean([i.true_fitnesses[-1] for i in ea])
            X[i].append(ea.total_evals)
            Y[i].append(f)
    
    new_X, new_Y = aggregate(X, Y)
    return {
        "fitness": Y,
        "gens":np.arange(gens)+1,
        "fitness_evals": new_Y,
        "evals":new_X
    }

def gen_graph(results, title="", save=None):
    plt.figure(figsize=(16, 8))
    for algo, d in results.items():
        mean_Y = np.mean(d["fitness"], axis=0)
        std = np.std(d["fitness"], axis=0)
        plt.plot(mean_Y, label=algo)
#        plt.fill_between(d["gens"], mean_Y-std, mean_Y+std)
        
    plt.legend()
    plt.xlabel("Generations")
    plt.ylabel("True fitness")
    plt.title(title)
    if save:
        plt.savefig(save)

def eval_graph(results, title="", save=None):
    plt.figure(figsize=(16, 8))
    for algo, d in results.items():
        mean_Y = np.mean(d["fitness_evals"], axis=0)
        std = np.std(d["fitness_evals"], axis=0)
        plt.plot(d["evals"], mean_Y, label=algo)
#        plt.fill_between(d["evals"], mean_Y-std, mean_Y+std)
        
    plt.legend()
    plt.xlabel("Evaluations")
    plt.ylabel("True fitness")
    plt.title(title)
    if save:
        plt.savefig(save)


def run_xp(basepb, 
    max_fit, 
    n_genes, 
    gens=1000, 
    normal_noise=False,
    noise=0,
    n_evals=1
    ):
    cfg = {
        "n_pop":12,
        "n_elites":6,
        "n_genes":n_genes, 
        "delta":0.1,
        "epsilon":0.1,
        "mut_size":0.3,
        "max_eval":120,
        "max_fit":10,
        "noise_level": noise
    }

    algos = [EA, MultiEA, UCEA]
    noise_type='Normal' if normal_noise else 'Uniform'


    pb = Noisy(basepb, noise=noise, max_fit=max_fit, normal=normal_noise)

    gen_fitness = [[] for _ in algos]
    eval_fitness = [[] for _ in algos]
    evals = [[] for _ in algos]

    results = {}

    for algo in algos:
        a = algo.__name__
        results[a] = multi_run(algo, cfg, pb, gens, n_evals, name=f'{algo.__name__} | {noise}')

    title = f"{basepb.name} - {int(noise*100)}% noise ({noise_type})"
    path = f"plots/{noise_type}/Gen_{basepb.name}_{int(noise*100)}.png"

    gen_graph(results, title=title, save=path)

    title = f"{basepb.name} - {int(noise*100)}% noise ({noise_type})"
    path = f"plots/{noise_type}/Eval_{basepb.name}_{int(noise*100)}.png"

    eval_graph(results, title=title, save=path)

    # TODO: add graph "evals = f(gen)"