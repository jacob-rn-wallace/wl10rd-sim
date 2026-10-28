"""Tune the change-over table (Table-3 entries, D_end_angle, A_descent) by CMA-ES.

Every candidate is run from an exact snapshot taken at the first change-over, through --steps change-overs
and the single-support steps after them. With --full the last step is the stop step + settling, so a
success is a complete walk. With --robust the cost is summed over 4 snapshots with small extra pelvis
velocity (0, +-3 cm/s sideways, +3 cm/s forward).

    python scripts/tune_changeover.py --design wl10rd_sim/designs/tuned.json --steps 4 --full --out work/c.json
Slow: about 0.4 s per change-over per candidate; --steps 4 --full with 14 x 40 candidates ~ 20-30 min.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from wl10rd_sim.design import load, save
from wl10rd_sim.sim import Walker
from wl10rd_sim.tune import tune_changeover, tune_changeover_robust, perturbed_snapshots, changeover_cost

ap = argparse.ArgumentParser()
ap.add_argument('--design', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--steps', type=int, default=4)
ap.add_argument('--full', action='store_true', help='end with the stop step and settling')
ap.add_argument('--robust', action='store_true', help='sum the cost over 4 perturbed starting snapshots')
ap.add_argument('--iters', type=int, default=40)
ap.add_argument('--sigma', type=float, default=0.05, help='CMA step in normalised [0,1] units (0.05 = local)')
ap.add_argument('--seed', type=int, default=1)
a = ap.parse_args()
d = load(a.design)
w = Walker(des=d, n_steps=1)
snap = w.prepare_first_changeover()
if a.robust:
    snaps = perturbed_snapshots(w, snap)
    print('start', [round(changeover_cost(w, s, d, n_steps=a.steps, full=a.full), 2) for s in snaps], flush=True)
    best, c = tune_changeover_robust(d, snaps, w, iters=a.iters, sigma=a.sigma, seed=a.seed, n_steps=a.steps, full=a.full)
else:
    c, det = changeover_cost(w, snap, d, detail=True, n_steps=a.steps, full=a.full)
    print('start cost', round(c, 3), det.get('fail'), flush=True)
    best, det = tune_changeover(d, iters=a.iters, sigma=a.sigma, snap=snap, w=w, n_steps=a.steps, seed=a.seed, full=a.full)
save(best, a.out)
print('saved', a.out)
