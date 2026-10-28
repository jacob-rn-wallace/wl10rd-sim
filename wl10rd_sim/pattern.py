"""Single-support preset walking pattern (paper Sec.3.2):
three states (start, knee-bent middle, end), joint-space interpolation by a 6th-order polynomial,
ZMP checked on the whole trajectory, and the tracking mode of Eq.(45) at the start."""
import numpy as np
from . import params as P
from . import kin
from .model import SIDES

HA = P.GEOM['ankle_height']


# ---------------------------------------------------------------- 6th-order polynomial
def poly6_coeffs(T, Tm, q0, v0, a0, q1, q2, v2, a2):
    """Coefficients c[0..6] of q(t)=sum c_k t^k satisfying start pos/vel/acc, middle pos at Tm,
    end pos/vel/acc at T. Vectorised over joints (q's are arrays)."""
    A = np.zeros((7, 7))
    A[0, 0] = 1
    A[1, 1] = 1
    A[2, 2] = 2
    A[3] = [Tm**k for k in range(7)]
    A[4] = [T**k for k in range(7)]
    A[5] = [k * T**(k - 1) if k > 0 else 0 for k in range(7)]
    A[6] = [k * (k - 1) * T**(k - 2) if k > 1 else 0 for k in range(7)]
    rhs = np.vstack([q0, v0, a0, q1, q2, v2, a2])
    return np.linalg.solve(A, rhs)


def poly_eval(c, t, deriv=0):
    t = np.asarray(t, float)
    out = 0
    for k in range(deriv, 7):
        f = 1.0
        for j in range(deriv):
            f *= (k - j)
        out = out + f * np.multiply.outer(t**(k - deriv), c[k]) if np.ndim(t) else out + f * t**(k - deriv) * c[k]
    return out


def tracking_mode(theta_p_t, theta_p0, theta_e, t, B):
    """Eq.(45) as the method intends: start at the measured angle theta_e and converge
    to the preset pattern by t = B. (The printed sign of the last factor is reversed.)"""
    if t >= B:
        return theta_p_t
    return theta_p_t + ((B - t) ** 3 / B ** 3) * (theta_e - theta_p0)


# ---------------------------------------------------------------- key states
def _ankle_z_for_spike(spike, clear, pitch):
    o = kin.spike_offsets('L')[spike]
    r = P.GEOM['spike_radius']
    c = kin.foot_R(pitch) @ np.array([o[0], 0.0, -HA + r])
    return clear + r - c[2]


def swing_ankle_for_heel(x, y, heel_clear, pitch):
    """Ankle position of a foot pitched by `pitch` (toe up < 0) whose heel spikes are heel_clear above the floor."""
    return np.array([x, y, _ankle_z_for_spike('heel_o', heel_clear, pitch)])


def swing_ankle_for_toe(x, y, toe_clear, pitch):
    """Ankle position of a foot pitched by `pitch` (toe down > 0) whose toe spikes are toe_clear above the floor."""
    return np.array([x, y, _ankle_z_for_spike('toe_o', toe_clear, pitch)])


class KeyState:
    """Cartesian description of one key state of the single-support pattern."""
    def __init__(self, pelvis_p, pelvis_v, sup_ankle, swing_ankle, swing_pitch, swing_v=None, pelvis_a=None):
        self.pelvis_p = np.asarray(pelvis_p, float)
        self.pelvis_v = np.asarray(pelvis_v, float)
        self.pelvis_a = np.zeros(3) if pelvis_a is None else np.asarray(pelvis_a, float)
        self.sup_ankle = np.asarray(sup_ankle, float)
        self.swing_ankle = np.asarray(swing_ankle, float)
        self.swing_pitch = swing_pitch
        self.swing_v = np.zeros(3) if swing_v is None else np.asarray(swing_v, float)


def joints_of(body, sup, ks):
    """Leg joint vectors (qL, qR) for a key state (pelvis upright)."""
    sw = 'R' if sup == 'L' else 'L'
    I = np.eye(3)
    qs = kin.leg_ik_full(body, sup, ks.pelvis_p, I, ks.sup_ankle, I)
    body.d.qpos[body.idx.leg_q[sup]] = qs
    qw = kin.leg_ik_full(body, sw, ks.pelvis_p, I, ks.swing_ankle, kin.foot_R(ks.swing_pitch))
    return (qs, qw) if sup == 'L' else (qw, qs)


def joint_velocities_of(body, sup, ks, h=1e-4):
    """Joint velocities for pelvis velocity ks.pelvis_v with the support foot fixed and the swing foot moving at swing_v."""
    sw = 'R' if sup == 'L' else 'L'
    q0 = np.concatenate(joints_of(body, sup, ks))
    ks2 = KeyState(ks.pelvis_p + h * ks.pelvis_v, ks.pelvis_v, ks.sup_ankle, ks.swing_ankle + h * ks.swing_v, ks.swing_pitch)
    q1 = np.concatenate(joints_of(body, sup, ks2))
    v = (q1 - q0) / h
    return v[:6], v[6:]


def joint_accelerations_of(body, sup, ks, h=2e-3):
    """Joint accelerations for pelvis acceleration ks.pelvis_a (support foot fixed, swing foot at constant velocity)."""
    def q_at(t):
        k = KeyState(ks.pelvis_p + t * ks.pelvis_v + 0.5 * t * t * ks.pelvis_a, ks.pelvis_v, ks.sup_ankle,
                     ks.swing_ankle + t * ks.swing_v, ks.swing_pitch)
        return np.concatenate(joints_of(body, sup, k))
    if not np.any(ks.pelvis_a):
        return np.zeros(12)
    return (q_at(h) - 2 * q_at(0.0) + q_at(-h)) / (h * h)


class SSPattern:
    """A single-support pattern for support leg `sup`, duration T, middle time Tm."""
    def __init__(self, body, sup, S0, S1, S2, T, Tm, v0=None, a0=None):
        self.sup, self.T, self.Tm = sup, T, Tm
        self.sw = 'R' if sup == 'L' else 'L'
        self.sup_ankle = S0.sup_ankle.copy()
        q0 = np.concatenate(joints_of(body, sup, S0))
        q1 = np.concatenate(joints_of(body, sup, S1))
        q2 = np.concatenate(joints_of(body, sup, S2))
        if v0 is None:
            v0 = np.concatenate(joint_velocities_of(body, sup, S0))
        v2 = np.concatenate(joint_velocities_of(body, sup, S2))
        # paper Sec.3.2: start and end angle, velocity AND acceleration are specified for continuity with
        # the change-over phase
        a0 = joint_accelerations_of(body, sup, S0) if a0 is None else a0
        a2 = joint_accelerations_of(body, sup, S2)
        self.c = poly6_coeffs(T, Tm, q0, v0, a0, q1, q2, v2, a2)
        self.S0, self.S1, self.S2 = S0, S1, S2

    def q(self, t):
        t = np.clip(t, 0, self.T)
        qq = sum(self.c[k] * t**k for k in range(7))
        return qq[:6], qq[6:]

    def qd(self, t):
        t = np.clip(t, 0, self.T)
        qq = sum(k * self.c[k] * t**(k - 1) for k in range(1, 7))
        return qq[:6], qq[6:]


def evaluate(body, pat, n=201, margin_poly=True):
    """Kinematic evaluation of a pattern: pelvis pose (support foot fixed flat), CoM, ZMP,
    swing-spike clearance. Returns a dict of arrays."""
    m, idx = body.m, body.idx
    ts = np.linspace(0, pat.T, n)
    sup = pat.sup
    sup_p = pat.sup_ankle
    I = np.eye(3)
    com = np.zeros((n, 3)); L = np.zeros((n, 3)); pel = np.zeros((n, 3)); pitch = np.zeros(n)
    clear = np.zeros(n)
    qpos_prev = None
    qposs = []
    for i, t in enumerate(ts):
        qL, qR = pat.q(t)
        pw, Rw = kin.pelvis_from_support(body, sup, sup_p, I, qL, qR)
        body.set(pw, Rw, qL, qR)
        qposs.append(body.d.qpos.copy())
        pel[i] = pw
        pitch[i] = np.arctan2(Rw[0, 2], Rw[2, 2])
        # swing spike heights
        sw = pat.sw
        zmin = min(body.d.geom_xpos[idx.spikes[(sw, k)]][2] for k in ('toe_o', 'toe_i', 'heel_o', 'heel_i'))
        clear[i] = zmin - P.GEOM['spike_radius']
    # velocities by finite difference of qpos
    qposs = np.array(qposs)
    dt = ts[1] - ts[0]
    import mujoco
    for i in range(n):
        j0, j1 = max(i - 1, 0), min(i + 1, n - 1)
        v = np.zeros(m.nv)
        mujoco.mj_differentiatePos(m, v, (j1 - j0) * dt, qposs[j0], qposs[j1])
        body.d.qpos[:] = qposs[i]
        body.d.qvel[:] = v
        mujoco.mj_kinematics(m, body.d); mujoco.mj_comPos(m, body.d); mujoco.mj_comVel(m, body.d)
        mujoco.mj_subtreeVel(m, body.d)
        com[i] = body.d.subtree_com[idx.torso]
        L[i] = body.d.subtree_angmom[idx.torso]
    mass = float(sum(m.body_mass))
    zmp, comv = kin.zmp_from_trajectory(ts, com, L, mass)
    poly = kin.sole_polygon_xy(sup, sup_p[:2])
    marg = np.array([kin.point_in_poly_margin(z, poly) for z in zmp])
    return dict(t=ts, com=com, comv=comv, zmp=zmp, margin=marg, pelvis=pel, pelvis_pitch=pitch,
                clearance=clear, poly=poly, qpos=qposs)
