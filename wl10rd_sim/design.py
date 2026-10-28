"""Gait design: the numbers that define the walking patterns and the change-over sequence.

The change-over table mirrors the paper's Table 3. Conventions used here (chosen so the paper's
signs and neutral positions make geometric sense, see README):
  ankle pitch angle  q_p : the joint angle; q_p = -(forward lean of the shank over the foot)
                           e.g. the rear leg in double support leans forward ~15 deg -> q_p ~ -15 deg (paper NP = -15)
  ankle pitch torque      : + rotates the shank backward relative to a fixed foot (brakes forward motion)
  ankle roll  angle  q~_r : mirrored per side so that + = hip outboard of the foot
                           (the state when the body is over that foot; paper NP = +12 deg)
Impedance law (Eq.33 rearranged): tau = TRQ - VIS * dq - ELA * (q - NP)
'POS' = position control (paper Table 1 'POS'; shown as '---' / 'not used' in Tables 2-3).
"""
import copy
import math

D = math.pi / 180

# Paper Table 3 (experiment). Units: VIS N m s/rad, ELA N m/rad, NP deg, TRQ N m
TABLE3 = {
    'HP': {'A': (200, 0, 0, -3), 'B': (90, 0, 0, -28), 'C': (55, 300, -15, 19), 'D': 'POS'},
    'HR': {'A': 'POS', 'B': 'POS', 'C': (80, 0, 0, 0), 'D': 'POS'},
    'FP': {'A': 'POS', 'B': (55, 280, 0, 0), 'C': (50, 300, 0, 0), 'D': (400, 800, 1, 28)},
    'FR': {'A': 'POS', 'B': (80, 0, 0, 0), 'C': (250, 300, 12, -10), 'D': (70, 800, 12, 38)},
}

# Deviation used in this simulation (see README): the hind-ankle roll is compliant in A and B so the body
# can move diagonally (paper Fig.5); Table 3 marks it '---'.
TABLE_SIM = copy.deepcopy(TABLE3)
TABLE_SIM['HR']['A'] = (40, 0, 0, 0)
TABLE_SIM['HR']['B'] = (40, 0, 0, 0)

DEFAULT = dict(
    # ---- posture
    pelvis_height=0.91,        # [est] standing pelvis (hip axis) height; legs nearly straight as in Fig. 18 and WL-10R Figs. 7, 10
    S0_z=0.86, S1_z=0.90, S2_z=0.88,   # [tuned] pelvis height at the three key states (reach limits the 40 cm stance)
    # ---- start (complete walking)
    t_stand=0.5, t_shift=1.5,
    shift_x=0.0, shift_y=0.145,  # pelvis position over the first support foot (lateral from midline)
    # ---- steady single-support pattern (relative to the support ankle; lateral 'y' measured from the
    #      midline towards the support side)
    T=1.0,                     # [paper] single support 1.0 s
    Tm=0.5,                    # middle (knee-bent) state time
    B_track=0.30,              # tracking-mode duration (paper: 'B', value not given)
    S0_x=-0.02, S0_y=0.150, S0_vx=0.15, S0_vy=0.0,     # pattern start (nominal end of change-over)
    S0_toe_pitch=0.25, S0_toe_clear=0.005,              # rear foot just lifted, toe down
    S1_x=0.04, S1_y=0.160, S1_sw_x=0.0, S1_sw_h=0.08,   # middle state: swing ankle lift
    S2_x=0.03, S2_y=0.150, S2_vx=0.40, S2_vy=0.0,       # pattern end = start of phase A
    S2_heel_clear=0.020, S2_land_pitch=-0.12,            # front foot: heel above floor, toe up
    # ---- CoM hand-over states relative to the support ankle: (x forward, u lateral, + = outboard)
    # [paper, Fig. 12 digitized; paper_data/fig12_digitized.json] start: CoM about over the ankle, nearly at rest;
    # end: 167 mm ahead, 86 mm inboard, moving ~0.45 m/s at 45 deg towards the next foot (speed assumes 25 ms markers)
    tgt_S2=(0.167, -0.086), tgt_S2v=(0.32, -0.32),
    tgt_S0=(-0.0085, 0.0176), tgt_S0v=(0.03, 0.02),
    # ---- change-over
    table=copy.deepcopy(TABLE_SIM),
    A_descent=0.57,            # front-foot descent per metre of pelvis advance in phase A (= locked legs pivoting on the rear ankle)
    D_end_angle=0.0,           # [paper: 'ankle angle'] D ends when the front ankle pitch q_p <= this (rad)
    D_end_roll=0.03,           # [est] D also waits until the front ankle roll is within this of the next pattern's start (rad)
    D_max=0.35,                # [est] safety: D hands over to single support after this time
    t_limit=1.2,               # safety: max change-over time
    # ---- stop
    stop_mode='single',        # 'single': a stop step (preset pattern) that sets the swing foot down beside the support
                               # foot, then double-support settling; 'double': a slow static step after the last change-over
    stop_step_z=0.84,          # [est] pelvis height during the closing static step (reach)
    stop_S2_x=0.0, stop_S2_y=0.150,
    t_settle=1.5,
)


def design(**over):
    d = copy.deepcopy(DEFAULT)
    for k, v in over.items():
        d[k] = v
    return d


def load(path):
    """Load a design saved as JSON (designs/tuned.json is the walk shown in the viewer)."""
    import json
    with open(path) as f:
        over = json.load(f)
    if 'table' in over:
        over['table'] = {a: {p: (v if v == 'POS' else tuple(v)) for p, v in vv.items()} for a, vv in over['table'].items()}
    return design(**over)


def save(des, path):
    """Save a design as JSON at full float precision (rounding breaks tuned walks; see CLAUDE.md)."""
    import json
    out = {}
    for k, v in des.items():
        if k == 'table':
            out[k] = {a: {p: (e if e == 'POS' else [float(x) for x in e]) for p, e in vv.items()} for a, vv in v.items()}
        elif isinstance(v, (tuple, list)):
            out[k] = [float(x) for x in v]
        elif isinstance(v, float):
            out[k] = float(v)
        else:
            out[k] = v
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
