from .algos.ea import *
from .algos.ucea import *
from .problems.noise import *
import sys

from .postproc.graphs import *
from .postproc.postprocessing import *
import pandas as pd
import numpy as np
import datetime
import logging
import os

def multi_run(algo, cfg, server, n, name=""):
    X = [[] for _ in range(n)] # Evaluations
    Y = [[] for _ in range(n)] # Fitness
    for i in range(n):
        ea = algo(cfg, server)
        # print(ea)
        X[i], Y[i] = ea.run(name=name)
        # if isinstance(ea, UCEA):
        #     ea.pop.plot()
    return X, Y


def run_xp(server, args):
    max_fit = server.pb.max_fit
    # args to dict
    cfg = args.__dict__

    algos = [ALGOS[i.lower()] for i in args.algos]
    noise_type='Normal' if args.normal_noise else 'Uniform'

    gen_fitness = [[] for _ in algos]
    eval_fitness = [[] for _ in algos]
    evals = [[] for _ in algos]

    results = {}

    for algo in algos:
        n = args.n_evals
        X = [[] for _ in range(n)] # Evaluations
        Y = [[] for _ in range(n)] # Fitness

        a = algo.__name__
        path = f'saves/{noise_type}/Data_{a}_{server.pb.name}_{int(args.noise*100)}'
        path += f'_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/'
        if not os.path.exists(path):
            os.makedirs(path)
        name=f'{algo.__name__} | {args.problem} | {args.noise}'
        for i in range(n):
            ea = algo(cfg, server)
            ea.logger = logging.getLogger(f'{algo.__name__}_{args.problem}_{args.noise}_{i}')
            ea.logger.setLevel(logging.INFO)
            handler = logging.FileHandler(path+f'/eval_{i}.csv', mode='w')
            handler.setLevel(logging.INFO)
            ea.logger.addHandler(handler)
            X[i], Y[i] = ea.run(name=name)

        # X, Y = multi_run(algo, cfg, server, args.n_evals,
        # path = f'saves/{noise_type}/Data_{a}_{server.pb.name}_{int(args.noise*100)}'
        # add timestamp to path
        # path += f'_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        save(X, Y, path)
        print("\nSaved to " + path + "\n")
        sys.stdout.flush()
        X, Y = load(path)
        results[a] = postprocessing(X, Y)

    if args.no_plot:
        return
        
    if server.pb.name[0]=="F":
        title = f"{server.pb.name[2:]} - {int(args.noise*100)}% noise ({noise_type} fitness noise)"
    elif server.pb.name[0]=="A":
        title = f"{server.pb.name[2:]} - {int(args.noise*100)}% noise (Action noise)"
    elif server.pb.name[0]=="S":
        title = f"{server.pb.name[2:]} (Seed noise)"

    # Fitness = f(gen)
    path = f"plots/{noise_type}/Gen_{server.pb.name}_{int(args.noise*100)}.png"
    # gen_graph(results, title=title, save=path, max_val=args.max_fit)

    # Fitness = f(eval)
    path = f"plots/{noise_type}/Eval_{server.pb.name}_{int(args.noise*100)}.png"
    eval_graph(results, title=title, save=path, max_val=args.max_fit)

    # Eval = f(gen)
    path = f"plots/{noise_type}/Cost_{server.pb.name}_{int(args.noise*100)}.png"
    # cost_graph(results, title=title, save=path)

