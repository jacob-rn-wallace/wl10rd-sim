"""Run complete walks from standing and report how far each gets. The robustness check used in the README.

    python scripts/check_walk.py --design wl10rd_sim/designs/tuned.json            # default battery
    python scripts/check_walk.py --design d.json --case 4,0,0 --case 4,0.02,0     # steps, vy, vx at start
    python scripts/check_walk.py --design d.json --case 4,0,0,1.0                  # 4th field: kg added to upper body
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from wl10rd_sim.design import load
from wl10rd_sim.sim import Walker

ap = argparse.ArgumentParser()
ap.add_argument('--design', required=True)
ap.add_argument('--case', action='append', help='steps,vy,vx[,dm]  (initial pelvis velocity m/s, upper-body mass kg)')
a = ap.parse_args()
cases = a.case or ['4,0,0', '5,0,0', '4,0.02,0', '4,-0.02,0', '4,0,0,1', '4,0,0,-1']
d = load(a.design)
for c in cases:
    v = [float(x) for x in c.split(',')] + [0.0] * 4
    n, vy, vx, dm = int(v[0]), v[1], v[2], v[3]
    w = Walker(des=d, n_steps=n, log_hz=20)
    if dm:
        w.m.body_mass[w.m.body('torso').id] += dm
    if vy or vx:
        orig = w.reset_standing
        def rs(orig=orig, vy=vy, vx=vx):
            orig(); w.d.qvel[1] = vy; w.d.qvel[0] = vx
        w.reset_standing = rs
    w.run()
    n_co = sum(1 for _, e in w.events if e.startswith('A_'))
    print(f'steps={n} vy={vy:+.2f} vx={vx:+.2f} dm={dm:+.1f}:',
          'COMPLETE' if w.fail is None else f'FAIL ({w.fail})', f'| change-overs done {n_co} | t={w.t:.2f} s', flush=True)
