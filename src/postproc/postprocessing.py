import pandas as pd
import numpy as np
from scipy import interpolate

def aggregate(X, Y):
    X = np.array(X)
    Y = np.array(Y)
    F = [interpolate.interp1d(X[i], Y[i]) for i in range(len(X))] # Compute interpolation function

    # Compute bounds
    max_evals = min(x[-1] for x in X)
    min_evals = max(x[0] for x in X)

    # Create as many points
    n = len(X[0])
    step = (max_evals-min_evals)/n
    assert step > 0, f"No points to interpolate between {min_evals} and {max_evals}"
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

def postprocessing(X, Y):
    new_X, new_Y = aggregate(X, Y)
    generations = [np.arange(len(x)) for x in X]
    cost_gens, cost = aggregate(generations, X)
    return {
        "fitness": Y,
        "gens":np.arange(len(X))+1,
        "fitness_evals": new_Y,
        "evals":new_X,
        "cost":cost,
        "cost_gens":cost_gens
    }

def save(X, Y, path):
    # to numpy
    X = np.array(X)
    Y = np.array(Y)
    # print("Save", X.shape, Y.shape)
    # make new folder 
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    # make each run a separate file
    for i in range(len(X)):
        df = pd.DataFrame({"evals":X[i], "fitness":Y[i]})
        df.to_csv(f"{path}/run_{i}.csv")

def load(path, t="run"):
    X = []
    Y = []
    i = 0
    while True:
        try:
            df = pd.read_csv(f"{path}/{t}_{i}.csv")
            X.append(df["evals"].values)
            Y.append(df["fitness"].values)
            i += 1
        except:
            break
    # to numpy
    X = np.array(X)
    Y = np.array(Y)
    # print("Load", X.shape, Y.shape)
    return X, Y
