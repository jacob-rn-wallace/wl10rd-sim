"""Tune a single-support pattern (kinematic ZMP margin + Fig. 12 hand-over states). Fast (~1-2 min).

    python scripts/tune_patterns.py --kind steady --design wl10rd_sim/designs/tuned.json --out work/p.json
    kinds: steady | start | stop   (tune steady first; start and stop reuse its hand-over states)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from wl10rd_sim.design import load, save
from wl10rd_sim.tune import tune_pattern

ap = argparse.ArgumentParser()
ap.add_argument('--design', required=True)
ap.add_argument('--kind', required=True, choices=['steady', 'start', 'stop'])
ap.add_argument('--out', required=True)
ap.add_argument('--iters', type=int, default=150)
ap.add_argument('--seed', type=int, default=1)
a = ap.parse_args()
best, det = tune_pattern(load(a.design), a.kind, iters=a.iters, sigma=0.25, seed=a.seed, popsize=14)
save(best, a.out)
print('saved', a.out)
