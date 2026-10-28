"""Build the WL-10RD MuJoCo model (MJCF) from params.py."""
import numpy as np
import mujoco
from . import params as P

G, M, C = P.GEOM, P.MASS, P.COM

LEG_JOINTS = ['hip_roll', 'hip_pitch', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll']
SIDES = {'L': +1, 'R': -1}
# joint numbers used in the paper (Fig.14); even = left side in the figure
PAPER_JOINT_NO = {('L', 'hip_roll'): 8, ('R', 'hip_roll'): 9, ('L', 'hip_pitch'): 6, ('R', 'hip_pitch'): 7,
                  ('L', 'hip_yaw'): 10, ('R', 'hip_yaw'): 11, ('L', 'knee'): 4, ('R', 'knee'): 5,
                  ('L', 'ankle_pitch'): 0, ('R', 'ankle_pitch'): 1, ('L', 'ankle_roll'): 2, ('R', 'ankle_roll'): 3}


def _box_inertia(m, sx, sy, sz):
    return (m / 12 * (sy**2 + sz**2), m / 12 * (sx**2 + sz**2), m / 12 * (sx**2 + sy**2))


def _f(v):
    return ' '.join(f'{x:.6g}' for x in v)


def _inertial(m, pos, size):
    ix, iy, iz = _box_inertia(m, *size)
    return f'<inertial mass="{m}" pos="{_f(pos)}" diaginertia="{ix:.6g} {iy:.6g} {iz:.6g}"/>'


def leg_xml(side):
    s = SIDES[side]
    hy = s * G['hip_spacing'] / 2
    l1, l2, ha = G['thigh'], G['shank'], G['ankle_height']
    yj = G['yaw_joint_below_hip']
    fl, fw = G['foot_length'], G['foot_width']
    sx = fl / 2 - G['spike_inset_x']
    sy = fw / 2 - G['spike_inset_y']
    r = G['spike_radius']
    lim = P.TORQUE_LIMIT
    vis = 'contype="0" conaffinity="0" group="1"'
    n = lambda j: f'{side}_{j}'
    spikes = ''.join(
        f'<geom name="{side}_spike_{nm}" type="sphere" size="{r}" pos="{_f((px, py, -ha + r))}" '
        f'rgba="0.15 0.15 0.15 1" condim="3" contype="1" conaffinity="0" group="2"/>'
        for nm, px, py in [('toe_o', sx, s * sy), ('toe_i', sx, -s * sy), ('heel_o', -sx, s * sy), ('heel_i', -sx, -s * sy)])
    return f'''
    <body name="{side}_hip_roll_link" pos="0 {hy} 0">
      <joint name="{n('hip_roll')}" axis="1 0 0" range="-35 35"/>
      {_inertial(M['hip_roll_link'], C['hip_roll_link'], (0.08, 0.08, 0.08))}
      <geom {vis} type="cylinder" size="0.055 0.05" euler="0 90 0" rgba="0.75 0.75 0.78 1"/>
      <body name="{side}_thigh_upper" pos="0 0 0">
        <joint name="{n('hip_pitch')}" axis="0 1 0" range="-100 60"/>
        {_inertial(M['thigh_upper'], C['thigh_upper'], (0.10, 0.10, 0.16))}
        <geom {vis} type="cylinder" size="0.07 0.045" euler="90 0 0" rgba="0.85 0.85 0.88 1"/>
        <geom {vis} type="box" size="0.045 0.04 {yj/2:.4f}" pos="0 0 {-yj/2:.4f}" rgba="0.2 0.2 0.22 1"/>
        <body name="{side}_thigh_lower" pos="0 0 {-yj}">
          <joint name="{n('hip_yaw')}" axis="0 0 1" range="-30 30"/>
          {_inertial(M['thigh_lower'], C['thigh_lower'], (0.10, 0.10, 0.25))}
          <geom {vis} type="cylinder" size="0.045 0.06" pos="0 0 -0.07" rgba="0.8 0.8 0.82 1"/>
          <geom {vis} type="box" size="0.04 0.035 {(l1-yj)/2:.4f}" pos="0 0 {-(l1-yj)/2:.4f}" rgba="0.2 0.2 0.22 1"/>
          <body name="{side}_shank" pos="0 0 {-(l1-yj):.4f}">
            <joint name="{n('knee')}" axis="0 1 0" range="-5 130"/>
            {_inertial(M['shank'], C['shank'], (0.12, 0.12, 0.30))}
            <geom {vis} type="cylinder" size="0.07 0.05" euler="90 0 0" rgba="0.85 0.85 0.88 1"/>
            <geom {vis} type="box" size="0.04 0.035 {l2/2:.4f}" pos="0 0 {-l2/2:.4f}" rgba="0.2 0.2 0.22 1"/>
            <body name="{side}_ankle_link" pos="0 0 {-l2}">
              <joint name="{n('ankle_pitch')}" axis="0 1 0" range="-50 50"/>
              {_inertial(M['ankle_link'], C['ankle_link'], (0.08, 0.08, 0.08))}
              <geom {vis} type="cylinder" size="0.06 0.05" euler="90 0 0" rgba="0.85 0.85 0.88 1"/>
              <body name="{side}_foot" pos="0 0 0">
                <joint name="{n('ankle_roll')}" axis="1 0 0" range="-35 35"/>
                {_inertial(M['foot'], C['foot'], (fl, fw, 0.10))}
                <geom {vis} type="cylinder" size="0.05 0.06" euler="0 90 0" pos="0 0 -0.03" rgba="0.75 0.75 0.78 1"/>
                <geom {vis} type="box" size="0.03 0.03 0.04" pos="0 0 -0.08" rgba="0.2 0.2 0.22 1"/>
                <geom {vis} type="box" size="{fl/2} {fw/2} 0.008" pos="0 0 {-ha + 2*r + 0.008:.4f}" rgba="0.1 0.1 0.1 1"/>
                <site name="{side}_sole" pos="0 0 {-ha:.4f}"/>
                {spikes}
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>'''


def actuators_xml():
    out = []
    for side in SIDES:
        for j in LEG_JOINTS:
            lim = P.TORQUE_LIMIT[j]
            # generic actuator: tau = ctrl + b1*q + b2*qdot  (b1 = -K, b2 = -D set at run time)
            out.append(f'<general name="{side}_{j}" joint="{side}_{j}" gainprm="1" biastype="affine" '
                       f'biasprm="0 0 0" ctrllimited="false" forcelimited="true" forcerange="{-lim} {lim}"/>')
    return '\n    '.join(out)


def build_xml():
    g = G
    S = P.SIM
    hz = g['hip_height_axis']
    vis = 'contype="0" conaffinity="0" group="1"'
    torso_inertials = []
    # combine three boxes into one inertial (parallel-axis theorem)
    parts = [(M['pelvis'], (0, 0, g['pelvis_center_z']), g['pelvis_size']),
             (M['box'], (0, 0, g['box_center_z']), g['box_size']),
             (M['top'], (0, 0, g['top_center_z']), g['top_size'])]
    mt = sum(p[0] for p in parts)
    com = sum(np.array(p[1]) * p[0] for p in parts) / mt
    I = np.zeros(3)
    for m, pos, size in parts:
        ii = np.array(_box_inertia(m, *size))
        d = np.array(pos) - com
        I += ii + m * np.array([d[1]**2 + d[2]**2, d[0]**2 + d[2]**2, d[0]**2 + d[1]**2])
    ps, bs, ts = g['pelvis_size'], g['box_size'], g['top_size']
    xml = f'''<mujoco model="WL-10RD">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="{S['timestep']}" integrator="implicitfast" gravity="0 0 -9.81"/>
  <default>
    <joint damping="0.5" armature="0.02" limited="true"/>
    <geom solref="{_f(S['contact_solref'])}" friction="{S['friction']} 0.01 0.001"/>
  </default>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="20 5 0.1" contype="0" conaffinity="1" rgba="0.8 0.8 0.8 1"/>
    <body name="torso" pos="0 0 {hz}">
      <freejoint name="root"/>
      <inertial mass="{mt}" pos="{_f(com)}" diaginertia="{_f(I)}"/>
      <geom {vis} type="box" size="{ps[0]/2} {ps[1]/2} {ps[2]/2}" pos="0 0 {g['pelvis_center_z']}" rgba="0.82 0.82 0.85 1"/>
      <geom {vis} type="box" size="{bs[0]/2} {bs[1]/2} {bs[2]/2}" pos="0 0 {g['box_center_z']}" rgba="0.25 0.25 0.28 1"/>
      <geom {vis} type="box" size="{ts[0]/2} {ts[1]/2} {ts[2]/2}" pos="0 0 {g['top_center_z']}" rgba="0.6 0.6 0.62 1"/>
      <geom {vis} type="box" size="0.02 0.02 0.0325" pos="0.15 0.15 0.1425" rgba="0.5 0.5 0.5 1"/>
      <geom {vis} type="box" size="0.02 0.02 0.0325" pos="0.15 -0.15 0.1425" rgba="0.5 0.5 0.5 1"/>
      <geom {vis} type="box" size="0.02 0.02 0.0325" pos="-0.15 0.15 0.1425" rgba="0.5 0.5 0.5 1"/>
      <geom {vis} type="box" size="0.02 0.02 0.0325" pos="-0.15 -0.15 0.1425" rgba="0.5 0.5 0.5 1"/>
      <site name="torso_site"/>
      {leg_xml('L')}
      {leg_xml('R')}
    </body>
  </worldbody>
  <actuator>
    {actuators_xml()}
  </actuator>
</mujoco>'''
    return xml


def build_model():
    m = mujoco.MjModel.from_xml_string(build_xml())
    return m


class Index:
    """Name -> index lookups."""
    def __init__(self, m):
        self.m = m
        self.qadr = {}
        self.dadr = {}
        self.act = {}
        for side in SIDES:
            for j in LEG_JOINTS:
                nm = f'{side}_{j}'
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)
                self.qadr[nm] = m.jnt_qposadr[jid]
                self.dadr[nm] = m.jnt_dofadr[jid]
                self.act[nm] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
        self.body = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(m.nbody)}
        self.spikes = {}
        for side in SIDES:
            for nm in ('toe_o', 'toe_i', 'heel_o', 'heel_i'):
                self.spikes[(side, nm)] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f'{side}_spike_{nm}')
        self.spike_geom_to_key = {v: k for k, v in self.spikes.items()}
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self.torso = self.body['torso']
        self.foot = {s: self.body[f'{s}_foot'] for s in SIDES}
        self.ankle = {s: self.body[f'{s}_ankle_link'] for s in SIDES}
        self.leg_q = {s: [self.qadr[f'{s}_{j}'] for j in LEG_JOINTS] for s in SIDES}
        self.leg_d = {s: [self.dadr[f'{s}_{j}'] for j in LEG_JOINTS] for s in SIDES}
        self.leg_a = {s: [self.act[f'{s}_{j}'] for j in LEG_JOINTS] for s in SIDES}


if __name__ == '__main__':
    m = build_model()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    print('nq', m.nq, 'nv', m.nv, 'nu', m.nu, 'mass', sum(m.body_mass), 'params total', P.TOTAL_MASS)
    print('CoM height standing:', d.subtree_com[1])
