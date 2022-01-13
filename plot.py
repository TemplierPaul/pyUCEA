#!/usr/bin/env python3

from src.run import *
from src.algos.ea import *
from src.algos.ucea import *
from src.problems.noise import *
from src.postproc.graphs import *
from src.postproc.postprocessing import *

import numpy as np
import os
import fnmatch
import argparse

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# Create args parser
parser = argparse.ArgumentParser(description='Plot results from a save file')
parser.add_argument('--problem', type=str, default='all_ones', help='Problem to run')
noise_types = [k.replace("noise_", "") for k in PROBLEMS.keys() if "noise_" in k]
parser.add_argument('--noise_type', type=str, default='fitness', choices=noise_types, help='Noise type: fitness / action / seed')
parser.add_argument('--noise', type=float, default=0, help='Noise level')
parser.add_argument('--normal_noise', default=False, help='Normal noise', action='store_true')
parser.add_argument('--algos', type=str, nargs='+', default=['ea', "rs", "ucea"], help='Algorithm to run')

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in dirs:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

if __name__ == "__main__":

    args = parser.parse_args()

    basepb = PROBLEMS[args.problem]()
    noise_type='Normal' if args.normal_noise else 'Uniform'
    algos = [ALGOS[i.lower()] for i in args.algos]
    noise_wrapper = PROBLEMS[f"noise_{args.noise_type}"]
    pb = noise_wrapper(basepb, noise=args.noise, normal=args.normal_noise)
    args.max_fit = pb.max_fit

    results = {}

    for algo in algos:
        a = algo.__name__
        dirs = find(f'Data_{a}_{pb.name}_{int(args.noise*100)}*', f'saves/{noise_type}/')
        print(dirs)
        path = dirs[-1] # TODO : if there are multiple dates, take most recent?
        print(path)
        if os.path.isdir(path):
            X, Y = load(path)
        else:
            raise Exception("No file " + path)
        results[a] = postprocessing(X, Y)

    if pb.name[0]=="F":
        title = f"{pb.name[2:]} - {int(args.noise*100)}% noise ({noise_type} fitness noise)"
    elif pb.name[0]=="A":
        title = f"{pb.name[2:]} - {int(args.noise*100)}% noise (Action noise)"
    elif pb.name[0]=="S":
        title = f"{pb.name[2:]} (Seed noise)"

    path = f"plots/{noise_type}/Eval_{pb.name}_{int(args.noise*100)}.png"
    eval_graph(results, title=title, save=path, max_val=args.max_fit)
