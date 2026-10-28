"""Kinematics: analytic leg IK (hip roll -> hip pitch -> yaw -> knee -> ankle pitch -> ankle roll, Fig.14),
foot/sole geometry, whole-body CoM, angular momentum and ZMP."""
import numpy as np
import mujoco
from . import params as P
from .model import SIDES, LEG_JOINTS

G = P.GEOM
L1, L2, HA = G['thigh'], G['shank'], G['ankle_height']
HIP_Y = G['hip_spacing'] / 2
GRAV = 9.81


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def foot_R(pitch=0.0, roll=0.0):
    """Foot orientation in world: pitch>0 = toe down, roll>0 = right-hand about +x."""
    return Ry(pitch) @ Rx(roll)


def leg_ik(side, pelvis_p, pelvis_R, ankle_p, foot_Rw=None):
    """Return the 6 joint angles [hip_roll, hip_pitch, hip_yaw, knee, ankle_pitch, ankle_roll].
    If foot_Rw is None the ankle angles are returned as nan."""
    s = SIDES[side]
    hip = pelvis_p + pelvis_R @ np.array([0.0, s * HIP_Y, 0.0])
    d = pelvis_R.T @ (ankle_p - hip)
    phi = np.arctan2(d[1], -d[2])
    vx = d[0]
    vz = -np.hypot(d[1], d[2])
    r2 = vx * vx + vz * vz
    c2 = (r2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    c2 = np.clip(c2, -1.0, 1.0)
    th2 = np.arccos(c2)                       # knee >= 0 (shank swings back)
    wx = -L2 * np.sin(th2)
    wz = -L1 - L2 * np.cos(th2)
    th1 = np.arctan2(vx, vz) - np.arctan2(wx, wz)
    th1 = (th1 + np.pi) % (2 * np.pi) - np.pi
    if foot_Rw is None:
        return np.array([phi, th1, 0.0, th2, np.nan, np.nan])
    Rsh = pelvis_R @ Rx(phi) @ Ry(th1 + th2)
    Mm = Rsh.T @ foot_Rw
    b = np.arctan2(-Mm[1, 2], Mm[1, 1])
    a = np.arctan2(-Mm[2, 0], Mm[0, 0])
    return np.array([phi, th1, 0.0, th2, a, b])


def reach_ok(side, pelvis_p, pelvis_R, ankle_p, margin=0.002):
    s = SIDES[side]
    hip = pelvis_p + pelvis_R @ np.array([0.0, s * HIP_Y, 0.0])
    return np.linalg.norm(ankle_p - hip) < L1 + L2 - margin


# ---------------------------------------------------------------- sole geometry
def spike_offsets(side):
    s = SIDES[side]
    sx = G['foot_length'] / 2 - G['spike_inset_x']
    sy = G['foot_width'] / 2 - G['spike_inset_y']
    return {'toe_o': np.array([sx, s * sy, -HA]), 'toe_i': np.array([sx, -s * sy, -HA]),
            'heel_o': np.array([-sx, s * sy, -HA]), 'heel_i': np.array([-sx, -s * sy, -HA])}


def sole_polygon_xy(side, ankle_xy):
    """Support polygon of one flat foot (spike rectangle)."""
    o = spike_offsets(side)
    pts = [o['toe_o'], o['toe_i'], o['heel_i'], o['heel_o']]
    return np.array([[ankle_xy[0] + p[0], ankle_xy[1] + p[1]] for p in pts])


def ankle_from_sole(sole_xy, pitch=0.0):
    """Ankle position of a flat foot whose sole centre is at sole_xy."""
    return np.array([sole_xy[0], sole_xy[1], HA])


# ---------------------------------------------------------------- whole-body quantities
class Body:
    """Scratch MjData for kinematic evaluation."""
    def __init__(self, m, idx):
        self.m, self.idx = m, idx
        self.d = mujoco.MjData(m)

    def set(self, pelvis_p, pelvis_R, qL, qR, qvel=None):
        d = self.d
        d.qpos[:3] = pelvis_p
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.asarray(pelvis_R, float).reshape(-1))
        d.qpos[3:7] = quat
        for side, q in (('L', qL), ('R', qR)):
            d.qpos[self.idx.leg_q[side]] = q
        if qvel is not None:
            d.qvel[:] = qvel
        else:
            d.qvel[:] = 0
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        if qvel is not None:
            mujoco.mj_comVel(self.m, d)
            mujoco.mj_subtreeVel(self.m, d)
        return d

    def com(self):
        return self.d.subtree_com[self.idx.torso].copy()

    def foot_pose(self, side):
        b = self.idx.foot[side]
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()


def pelvis_from_support(body, side_sup, foot_p, foot_Rw, qL, qR):
    """Pelvis pose such that the support foot is at (foot_p, foot_Rw) given leg joint angles."""
    body.set(np.zeros(3), np.eye(3), qL, qR)
    fp, fR = body.foot_pose(side_sup)
    # world = T * local, want T * (fp, fR) = (foot_p, foot_Rw)
    Rw = foot_Rw @ fR.T
    pw = foot_p - Rw @ fp
    return pw, Rw


def zmp_from_trajectory(t, com, angmom, mass):
    """ZMP from CoM trajectory and centroidal angular momentum (finite differences).
    Equivalent to Eqs.(31)-(32) of the paper for point masses, plus the rotational terms."""
    dt = np.gradient(t)
    cd = np.gradient(com, t, axis=0)
    cdd = np.gradient(cd, t, axis=0)
    Ld = np.gradient(angmom, t, axis=0)
    F = mass * (cdd + np.array([0, 0, GRAV]))
    px = com[:, 0] - (com[:, 2] * F[:, 0] + Ld[:, 1]) / F[:, 2]
    py = com[:, 1] - (com[:, 2] * F[:, 1] - Ld[:, 0]) / F[:, 2]
    return np.stack([px, py], axis=1), cd


def point_in_poly_margin(pt, poly):
    """Signed distance from pt to the edges of a convex polygon (positive inside)."""
    n = len(poly)
    best = np.inf
    # orientation
    area = 0.5 * sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1] for i in range(n))
    sgn = 1.0 if area > 0 else -1.0
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        e = b - a
        nrm = np.array([-e[1], e[0]]) * sgn
        nrm /= np.linalg.norm(nrm)
        best = min(best, np.dot(pt - a, nrm))
    return best


def leg_ik_full(body, side, pelvis_p, pelvis_R, ankle_p, foot_Rw, iters=6):
    """Analytic IK followed by a few damped-least-squares iterations using all 6 joints
    (the hip yaw corrects the small foot-yaw coupling)."""
    q = leg_ik(side, pelvis_p, pelvis_R, ankle_p, foot_Rw)
    if not reach_ok(side, pelvis_p, pelvis_R, ankle_p, margin=1e-4):
        raise ValueError(f'{side} ankle target out of reach')
    m, d, idx = body.m, body.d, body.idx
    other = 'R' if side == 'L' else 'L'
    q_other = d.qpos[idx.leg_q[other]].copy()
    jac_p = np.zeros((3, m.nv)); jac_r = np.zeros((3, m.nv))
    cols = idx.leg_d[side]
    b = idx.foot[side]
    for _ in range(iters):
        qL, qR = (q, q_other) if side == 'L' else (q_other, q)
        body.set(pelvis_p, pelvis_R, qL, qR)
        p = d.xpos[b]; R = d.xmat[b].reshape(3, 3)
        ep = ankle_p - p
        Re = foot_Rw @ R.T
        er = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
        e = np.concatenate([ep, er])
        if np.abs(e).max() < 1e-9:
            break
        mujoco.mj_jacBody(m, d, jac_p, jac_r, b)
        J = np.vstack([jac_p[:, cols], jac_r[:, cols]])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), e)
        q = q + np.clip(dq, -0.2, 0.2)
    return q
