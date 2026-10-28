"""
WL-10RD simulation parameters.

Every number is tagged:
  [paper]  taken from Takanishi et al., JRSJ 3(4) 1985 (Fig. 14, Fig. 16, Sec. 4, Tables 1-3)
  [est]    an estimate made for this simulation (not in any source we have)
  [tuned]  found by running the simulation (see tune.py); starts from a paper value when one exists

Units: SI (m, kg, s, rad, N m) unless the name says otherwise.
Coordinate frame: x forward, y left, z up.
"""
import math

DEG = math.pi / 180.0

# ---------------------------------------------------------------- geometry
GEOM = dict(
    hip_height_axis=0.938,     # [paper] Fig.16: 138 + 400 + 400 mm from floor to hip pitch axis
    hip_spacing=0.260,         # [paper] Fig.16: 260 mm between leg axes
    thigh=0.400,               # [paper] Fig.16: hip pitch axis -> knee
    shank=0.400,               # [paper] Fig.16: knee -> ankle
    ankle_height=0.138,        # [paper] Fig.16: ankle axis -> sole
    foot_length=0.280,         # [paper] Fig.16 side view
    foot_width=0.140,          # [paper] Fig.16 front view
    ankle_x_in_foot=0.0,       # [est] ankle assumed over the middle of the sole
    spike_inset_x=0.015,       # [est] spikes 15 mm in from toe / heel edge
    spike_inset_y=0.015,       # [est] spikes 15 mm in from the sole sides
    spike_radius=0.008,        # [est]
    yaw_joint_below_hip=0.15,  # [est] position of the yaw (RA) joint along the thigh
    # upper body (Fig.16: 500/388/440 mm widths, 110+65+210+120 mm heights above the hip axis)
    pelvis_size=(0.20, 0.50, 0.164),   # [paper dims 500 x 164; depth est] x,y,z full sizes
    pelvis_center_z=0.028,             # [est from Fig.16] relative to hip axis
    box_size=(0.44, 0.388, 0.21),      # [paper] control box
    box_center_z=0.280,                # [paper] 110+65+105 mm above the hip axis
    top_size=(0.30, 0.30, 0.12),       # [paper height 120; plan size est] valves/connectors on top
    top_center_z=0.445,                # [paper]
)

# ---------------------------------------------------------------- masses
# Total 83 kg is [paper]. The split is [est]: the paper gives no link masses.
MASS = dict(
    # Calibrated on the paper's Figs. 8 and 12 (see README): the hand-over states imply a CoM about
    # 0.75-0.80 m above the ankle axis with nearly straight legs, which needs most of the mass above the hips.
    # The WL-10R paper (RoManSy '84) gives ~80 kg for the predecessor with its onboard computer, CFRP frame.
    pelvis=17.0,        # [est] pelvis block incl. 4 hip roll/pitch actuators
    box=29.0,           # [est] control box (Z8002 + 6 Z8 boards, power electronics)
    top=9.0,            # [est] items on top of the box
    hip_roll_link=1.0,  # [est] per leg
    thigh_upper=2.0,    # [est] per leg
    thigh_lower=1.5,    # [est] per leg, incl. hip yaw RA
    shank=4.0,          # [est] per leg, incl. knee RSA + DC motor near the knee
    ankle_link=1.0,     # [est] per leg
    foot=4.5,           # [est] per leg, incl. ankle RSAs + plate
)
TOTAL_MASS = MASS['pelvis'] + MASS['box'] + MASS['top'] + 2 * sum(
    MASS[k] for k in ('hip_roll_link', 'thigh_upper', 'thigh_lower', 'shank', 'ankle_link', 'foot'))

# centres of mass relative to each body's joint origin (z down the leg)  [est]
COM = dict(
    hip_roll_link=(0.0, 0.0, 0.0),
    thigh_upper=(0.0, 0.0, -0.08),
    thigh_lower=(0.0, 0.0, -0.13),   # relative to the yaw joint
    shank=(0.0, 0.0, -0.12),
    ankle_link=(0.0, 0.0, 0.0),
    foot=(0.0, 0.0, -0.075),
)

# ---------------------------------------------------------------- actuators
# Electro-hydraulic RSA (DC servo motor + rotary servo actuator) on 10 DOF, servo valve + RA on hip yaw [paper].
# Their stiffness, damping and torque limits are not given anywhere [est].
SERVO = dict(
    kp=60000.0,    # [est] position-servo stiffness, N m/rad (electro-hydraulic RSA: very stiff)
    kv=600.0,      # [est] position-servo damping, N m s/rad
    kp_ankle=40000.0,  # [est] ankle position-servo stiffness (used when the ankle is in POS mode)
    kv_ankle=800.0,    # [est] best of a small sweep (ZMP tracking error in single support; README)
)
TORQUE_LIMIT = dict(          # [est]
    hip_roll=300.0, hip_pitch=300.0, hip_yaw=100.0, knee=300.0,
    ankle_pitch=100.0, ankle_roll=100.0,   # [paper] Sec.3.1.1: ankle torque control range about +-100 N m
)
# Achievable ankle impedance quoted in the paper (performance test on WL-10R):
ANKLE_RANGE_PAPER = dict(vis=(40, 130), ela=(0, 1700), trq=(-100, 100))   # [paper] Sec.3.1.1

# ---------------------------------------------------------------- simulation
SIM = dict(
    timestep=0.001,
    friction=1.0,          # [est] spikes on floor; the paper assumes friction is large enough
    contact_solref=(0.002, 3.0),
    switch_force=20.0,     # [est] N, sole micro-switch closing force
    debounce=0.010,        # [est] s, switch input debounce for the phase triggers
)

# ---------------------------------------------------------------- gait
GAIT = dict(
    step_length=0.40,      # [paper] 40 cm
    t_single=1.0,          # [paper] single support 1.0 s
    t_change_nominal=0.5,  # [paper] change-over 0.5 s (result, not a set value)
    foot_y=0.130,          # [paper] parallel gait, feet under the leg axes (hip spacing / 2)
)
