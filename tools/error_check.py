#!/usr/bin/env python3
"""
error_check.py — Compute L2 error vs Black-Scholes closed-form
Usage:
    python3 error_check.py bs_restart.h5 params.json
Prints:
    <L2 error as a plain floating-point number>
"""
import sys, json, math
import h5py
import numpy as np

def closed_form(S, K, r, sigma, T):
    """欧式看涨期权解析解 C(S,0)"""
    if S <= 0 or T <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    Φ  = lambda x: 0.5*(1 + math.erf(x/math.sqrt(2)))
    return S*Φ(d1) - K*math.exp(-r*T)*Φ(d2)

if len(sys.argv) != 3:
    print("Usage: error_check.py bs_restart.h5 params.json")
    sys.exit(1)

h5_file   = sys.argv[1]
json_file = sys.argv[2]

# 1) load parameters
prm = json.load(open(json_file, 'r'))
S_min, S_max = prm["S_min"], prm["S_max"]
N_S          = prm["N_S"]
T            = prm["T"]
K, r, sigma  = prm["K"], prm["r"], prm["sigma"]

# 2) open HDF5 & read last frame
with h5py.File(h5_file, 'r') as f:
    Vnum = f["/V"][-1][:]  # shape (N_S+1,)

# 3) build price grid
S = np.linspace(S_min, S_max, N_S+1)

# 4) compute analytic reference at t=0
Vref = np.array([closed_form(s, K, r, sigma, T) for s in S])

# 5) L2 error
err = np.linalg.norm(Vnum - Vref) / math.sqrt(len(S))
print(f"{err:.6e}")

