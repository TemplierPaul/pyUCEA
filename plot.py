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
parser.add_argument('--dirs', type=str, nargs='+', help='Save directories to plot')
parser.add_argument('--eval', default=False, help='plot eval graphs', action='store_true')
parser.add_argument('--out', type=str, default='png', help='Output type')

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in dirs:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

if __name__ == "__main__":

    args = parser.parse_args()

    results = {}

    for d in args.dirs:
        parts = d.split("/")
        assert parts[0] == "saves"
        noise_type = parts[1]
        _, a, noise_wrapper, pb_name, noise, day, time = parts[2].split("_")
        # TODO: doesn't work for problems with "_" in name (Leading_Ones, space_invaders)
        if os.path.isdir(d):
            root = "eval" if args.eval else "run"
            X, Y = load(d, root)
        else:
            raise Exception("No file " + d)
        results[a] = postprocessing(X, Y)

    if noise_wrapper == "F":
        title = f"{pb_name} - {noise}% noise ({noise_type} fitness noise)"
    elif noise_wrapper == "A":
        title = f"{pb_name} - {noise}% noise (Action noise)"
    elif noise_wrapper == "S":
        title = f"{pb_name} (Seed noise)"
    if args.eval:
        title += " Eval"

    path = f"plots/{noise_type}/Eval_{noise_wrapper}_{pb_name}_{noise}"
    if args.eval:
        path += "_eval"
    path += "." + args.out
    print(path)
    eval_graph(results, title=title, save=path)
