from scipy.stats import poisson
from typing import Tuple, Mapping, Sequence
import numpy as np
from copy import copy
import time

def repeat(x: np.ndarray, n: int) -> np.ndarray:
    axis_extend = len(x.shape)
    return np.repeat(np.expand_dims(x, axis=axis_extend), n,axis=axis_extend)

def to_dict(
    vf: np.ndarray, 
    pol: np.ndarray, 
    el: Sequence[Tuple[float, float]]
) -> Mapping[Tuple[int, float], Tuple[float, float]]:
    return {(i, el[j][0]): (vf[i,j], el[int(pol[i,j])][0]) for i in range(vf.shape[0]) for j in range(vf.shape[1])}

def backwardDP_vectorized(
    time_steps: int,
    init_inv: int,
    el: Sequence[Tuple[float, float]] # price from low to high
) -> Sequence[Mapping[Tuple[int, float], Tuple[float, float]]]:
    tr_inv = np.array([[j-i for i in range(init_inv+1)] for j in range(init_inv+1)])
    tr_re = np.array([[max(j-i,0) for i in range(init_inv+1)] for j in range(init_inv+1)])
    indic = np.zeros((init_inv+1, init_inv+1))
    indic[:,0] = 1
    ret = []
    vf = np.zeros((init_inv+1, len(el)))
    for t in reversed(range(time_steps)):
        pol = np.zeros((init_inv+1, len(el))) - 1
        vf_new = copy(vf)
        rvs = poisson(np.array([item[1] for item in el]))
        prob1 = rvs.pmf(repeat(tr_inv, len(el)))
        prob2 = 1 - rvs.cdf(repeat(tr_inv, len(el))-1)
        ind = repeat(indic,len(el))
        prob = prob1 * (1 - ind) + ind * prob2
        reward = repeat(tr_re, len(el)) * np.array([item[0] for item in el])
        q = np.sum(prob*(reward + repeat(vf, init_inv+1).transpose(2,0,1)), axis=1)
        for i in range(len(el)):
            vf_new[:, i] =  np.max(q, axis=1)
            pol[:, i] = np.argmax(q, axis=1)
        vf = vf_new
        ret.append(to_dict(vf, pol, el))
    return ret[::-1]

if __name__ == '__main__':
    alpha, beta = 1.0, 5
    el_func = lambda x, alpha, beta: alpha * np.e ** (-beta*x)
    ts = 20
    ii = 18
    n_prices = 50
    start = time.time()
    vf_pol = backwardDP_vectorized(ts, ii, [(p, el_func(p, alpha, beta)) for p in np.linspace(0, 1, n_prices+1)])
    end = time.time()
    print("Testing vectoried backward DP in python: ")
    print("Number of time steps: ", ts)
    print("Initial inventory: ", ii)
    print("Number of actions: ", n_prices)
    print("Optimal value function at initial state: ", vf_pol[0][(ii, 1)][0])
    print("Optimal policy at initial state: ", vf_pol[0][(ii, 1)][1])
    print("Time: ", end - start)