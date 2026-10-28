"""Tune the stop step (stop.* pattern keys, stop_S2_*, t_settle) by CMA-ES, from exact snapshots taken at
the start of the stop step after --steps change-overs (nominal + perturbed runs that get that far).

    python scripts/tune_stop.py --design work/c.json --steps 4 --out work/s.json
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from wl10rd_sim.design import load, save
from wl10rd_sim.tune import tune_stop

ap = argparse.ArgumentParser()
ap.add_argument('--design', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--steps', type=int, default=4)
ap.add_argument('--iters', type=int, default=40)
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--nominal-only', action='store_true')
a = ap.parse_args()
best, c = tune_stop(load(a.design), a.steps, iters=a.iters, seed=a.seed, perturb=not a.nominal_only)
save(best, a.out)
print('saved', a.out, 'cost', round(c, 3))
