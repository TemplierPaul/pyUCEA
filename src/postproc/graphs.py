import matplotlib.pyplot as plt
import numpy as np
import matplotlib
font = {'size'   : 22}

matplotlib.rc('font', **font)

def gen_graph(results, title="", save=None, max_val=None):
    plt.figure(figsize=(16, 8))
    for algo, d in results.items():
        label = f"{algo} - {len(d['fitness'])} runs"
        mean_Y = np.mean(d["fitness"], axis=0)
        std = np.std(d["fitness"], axis=0)
        # print shapes
        plt.plot(mean_Y, label=label)
        # plt.fill_between(d["gens"], mean_Y-std, mean_Y+std, alpha=0.2)
    if max_val:
        # add horizontal line
        plt.axhline(y=max_val, color='r', linestyle='-', label='Max value')
        
    plt.legend()
    plt.xlabel("Generations")
    plt.ylabel("Mean true fitness")
    plt.title(title)
    if save:
        plt.savefig(save)

def eval_graph(results, title="", save=None, max_val=None):
    plt.figure(figsize=(16, 8))
    for algo, d in results.items():
        label = f"{algo} - {len(d['fitness_evals'])} runs"
        mean_Y = np.mean(d["fitness_evals"], axis=0)
        std = np.std(d["fitness_evals"], axis=0)
        plt.plot(d["evals"], mean_Y, label=label)
        plt.fill_between(d["evals"], mean_Y-std, mean_Y+std, alpha=0.2)

    if max_val:
        # add horizontal line
        plt.axhline(y=max_val, color='r', linestyle='-', label='Max value')

    plt.legend()
    plt.xlabel("Evaluations")
    plt.ylabel("Mean true fitness")
    # plt.xlim(0, 20000)
    plt.title(title)
    if save:
        plt.savefig(save)

def cost_graph(results, title="", save=None):
    plt.figure(figsize=(16, 8))
    for algo, d in results.items():
        label = f"{algo} - {len(d['cost'])} runs"
        diff = [i[1:] - i[:-1] for i in d["cost"]]
        diff = np.array([np.insert(diff[i], 0, d["cost"][i][0]) for i in range(len(diff))])
        
        mean_Y = np.mean(diff, axis=0)    
        std = np.std(diff, axis=0)
        plt.plot(d["cost_gens"], mean_Y, label=label)
        plt.fill_between(d["cost_gens"], mean_Y-std, mean_Y+std, alpha=0.2)

    plt.legend()
    plt.xlabel("Generations")
    plt.ylabel("Evaluations")
    plt.title(title)
    if save:
        plt.savefig(save)


