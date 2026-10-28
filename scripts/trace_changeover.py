"""Print a text trace of the first change-over and the step after it (for debugging the table).

    python scripts/trace_changeover.py --design wl10rd_sim/designs/tuned.json --every 20
Columns: time, phase, pelvis xyz, CoM xyz, measured ZMP, switches (L toe_o toe_i heel_o heel_i, then R),
ankle pitch/roll angles L,R [deg], ankle pitch/roll torques L,R [N m].
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
from wl10rd_sim.design import load
from wl10rd_sim.sim import Walker, _Fail

ap = argparse.ArgumentParser()
ap.add_argument('--design', required=True)
ap.add_argument('--every', type=int, default=20, help='print every Nth 5 ms sample')
a = ap.parse_args()
w = Walker(des=load(a.design), n_steps=1, log_hz=200)
try:
    snap = w.prepare_first_changeover(); w.restore(snap)
    w._change_over('L'); w._single_support('R', 'steady', track=True)
except _Fail as e:
    print('FAIL', e)
print([(round(t, 3), e) for t, e in w.events])
idx = w.idx
keys = ['L_toe_o', 'L_toe_i', 'L_heel_o', 'L_heel_i', 'R_toe_o', 'R_toe_i', 'R_heel_o', 'R_heel_i']
for r in w.log[::a.every]:
    q = r['qpos']; sw = ''.join('1' if r['sw'][k] else '0' for k in keys)
    ap_ = [np.degrees(q[idx.leg_q[s][4]]) for s in 'LR']; ar = [np.degrees(q[idx.leg_q[s][5]]) for s in 'LR']
    tp = [r['ctrl_force'][idx.leg_a[s][4]] for s in 'LR']; tr = [r['ctrl_force'][idx.leg_a[s][5]] for s in 'LR']
    print(f"{r['t']:.3f} {r['phase']:2s} pel={q[:3].round(3)} com={r['com'].round(3)} zmp={np.round(r['zmp'], 3)} "
          f"sw={sw} ap={ap_[0]:.1f},{ap_[1]:.1f} ar={ar[0]:.1f},{ar[1]:.1f} tP={tp[0]:.0f},{tp[1]:.0f} tR={tr[0]:.0f},{tr[1]:.0f}")
