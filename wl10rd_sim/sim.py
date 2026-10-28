"""WL-10RD walking simulation: complete walking (stand -> walk -> stand) with the paper's control method.

Single support : program control of all 12 joints by the preset pattern (pattern.py), tracking mode at the start.
Change-over    : four phases A-D switched by the sole spike switches and the front ankle angle; the ankle
                 joints run the position/torque impedance control of Eq.(33) with Table-3-style parameters.
                 The other joints keep each leg's ankle on its footprint (or on the paper's locked-leg path for
                 a foot in the air), so the body moves as a pendulum driven by gravity and the ankle torques.
"""
import numpy as np
import mujoco
from . import params as P
from . import kin
from . import pattern as pt
from .model import build_model, Index, SIDES, LEG_JOINTS
from .design import design as make_design

HA = P.GEOM['ankle_height']
FY = P.GAIT['foot_y']
LSTEP = P.GAIT['step_length']
AP, AR = 4, 5  # ankle pitch / roll index within a leg


def other(s):
    return 'R' if s == 'L' else 'L'


class Walker:  # _stop_double is attached below
    def __init__(self, des=None, n_steps=6, log_hz=100):
        self.des = make_design() if des is None else des
        self.m = build_model()
        self.d = mujoco.MjData(self.m)
        self.idx = Index(self.m)
        self.body = kin.Body(self.m, self.idx)
        self.n_steps = n_steps
        self.dt = self.m.opt.timestep
        self.kp, self.kv = P.SERVO['kp'], P.SERVO['kv']
        self.kpa, self.kva = P.SERVO['kp_ankle'], P.SERVO['kv_ankle']
        self.mass = float(sum(self.m.body_mass))
        self.log_every = max(1, int(round(1.0 / (log_hz * self.dt))))
        self.log = []
        self.events = []
        self.supports = []   # (time, side, ankle position) at the start of each single support / change-over
        self.fail = None
        self._qset_prev = {}
        self.build_patterns()
        # reference ankle pitch = support ankle at the start of a steady step (body over the foot);
        # the paper's pitch angles (NP, the D-phase trigger) are measured from a vertical shank, i.e. from this posture
        self.q_ref_pitch = float(self.pat[('L', 'steady')].q(0.0)[0][AP])
        # mirrored support-ankle roll at the start of a steady step (D also waits for the lateral transfer)
        self.q_ref_roll = float(self.pat[('L', 'steady')].q(0.0)[0][AR])

    # ------------------------------------------------------------ patterns
    def _g(self, kind, name):
        """Design value for a pattern kind: '<kind>.<name>' overrides '<name>'."""
        return self.des.get(f'{kind}.{name}', self.des[name])

    def _ss_states(self, sup, kind):
        """Key states for a single-support pattern with support ankle at the local origin (0, s*FY, HA)."""
        g = self.des
        G = lambda n: self._g(kind, n)
        s = SIDES[sup]
        h = g['pelvis_height']
        a_s = np.array([0.0, s * FY, HA])
        yw = -s * FY
        if kind == 'start':
            S0 = pt.KeyState([g['shift_x'], s * g['shift_y'], h], [0, 0, 0], a_s, [0.0, yw, HA], 0.0)
            sw0x = 0.0
        else:
            S0 = pt.KeyState([g['S0_x'], s * g['S0_y'], g.get('S0_z', h)], [g['S0_vx'], s * g['S0_vy'], 0], a_s,
                             pt.swing_ankle_for_toe(-LSTEP, yw, g['S0_toe_clear'], g['S0_toe_pitch']), g['S0_toe_pitch'],
                             pelvis_a=[g.get('S0_ax', 0.0), s * g.get('S0_ay', 0.0), 0])
            sw0x = -LSTEP
        if kind == 'stop':
            # the stop step ends with the swing foot set down beside the support foot while the body is still
            # moving towards the middle; the double-support settling phase then absorbs that motion
            S2 = pt.KeyState([g['stop_S2_x'], s * g['stop_S2_y'], h], [g.get('stop_S2_vx', 0.0), s * g.get('stop_S2_vy', 0.0), 0],
                             a_s, [0.0, yw, HA + 0.004], 0.0)
            sw2x = 0.0
        else:
            S2 = pt.KeyState([g['S2_x'], s * g['S2_y'], g.get('S2_z', h)], [g['S2_vx'], s * g['S2_vy'], 0], a_s,
                             pt.swing_ankle_for_heel(LSTEP, yw, g['S2_heel_clear'], g['S2_land_pitch']), g['S2_land_pitch'],
                             pelvis_a=[g.get('S2_ax', 0.0), s * g.get('S2_ay', 0.0), 0])
            sw2x = LSTEP
        S1 = pt.KeyState([G('S1_x'), s * G('S1_y'), self.des.get(f'{kind}.S1_z', g.get('S1_z', h))], [0, 0, 0], a_s,
                         [0.5 * (sw0x + sw2x) + G('S1_sw_x'), yw, HA + G('S1_sw_h')], 0.0)
        return S0, S1, S2

    def build_patterns(self):
        g = self.des
        self.pat = {}
        for sup in SIDES:
            for kind in ('start', 'steady', 'stop'):
                S0, S1, S2 = self._ss_states(sup, kind)
                self.pat[(sup, kind)] = pt.SSPattern(self.body, sup, S0, S1, S2, g['T'], self._g(kind, 'Tm'))

    # ------------------------------------------------------------ actuators
    def pos(self, side, j, q, qd=0.0):
        a = self.idx.leg_a[side][j]
        kp, kv = (self.kpa, self.kva) if j >= 4 else (self.kp, self.kv)
        self.m.actuator_biasprm[a, 1] = -kp
        self.m.actuator_biasprm[a, 2] = -kv
        self.d.ctrl[a] = kp * q + kv * qd

    def imp(self, side, j, vis, ela, np_, trq):
        a = self.idx.leg_a[side][j]
        self.m.actuator_biasprm[a, 1] = -ela
        self.m.actuator_biasprm[a, 2] = -vis
        self.d.ctrl[a] = ela * np_ + trq

    def q(self, side):
        return self.d.qpos[self.idx.leg_q[side]].copy()

    def qd(self, side):
        return self.d.qvel[self.idx.leg_d[side]].copy()

    def pos_leg(self, side, q, qd=None, joints=range(6)):
        qd = np.zeros(6) if qd is None else qd
        for j in joints:
            self.pos(side, j, q[j], qd[j])

    # ------------------------------------------------------------ sensors
    def spike_forces(self):
        f = {k: 0.0 for k in self.idx.spikes}
        c6 = np.zeros(6)
        for i in range(self.d.ncon):
            con = self.d.contact[i]
            key = self.idx.spike_geom_to_key.get(con.geom1) or self.idx.spike_geom_to_key.get(con.geom2)
            if key is None:
                continue
            mujoco.mj_contactForce(self.m, self.d, i, c6)
            f[key] += abs(c6[0])
        return f

    def switches(self, f):
        th = P.SIM['switch_force']
        return {k: v > th for k, v in f.items()}

    def measured_zmp(self, f):
        tot = 0.0
        p = np.zeros(2)
        for k, v in f.items():
            if v > 0:
                gp = self.d.geom_xpos[self.idx.spikes[k]]
                p += v * gp[:2]
                tot += v
        return (p / tot) if tot > 1e-6 else np.array([np.nan, np.nan])

    def pelvis(self):
        return self.d.qpos[:3].copy(), self.d.xmat[self.idx.torso].reshape(3, 3).copy()

    def ankle_world(self, side):
        return self.d.xpos[self.idx.foot[side]].copy()

    def foot_Rw(self, side):
        return self.d.xmat[self.idx.foot[side]].reshape(3, 3).copy()

    def com(self):
        return self.d.subtree_com[self.idx.torso].copy()

    # ------------------------------------------------------------ setup
    def reset_standing(self):
        g = self.des
        h = g['pelvis_height']
        I = np.eye(3)
        pel = np.array([0.0, 0.0, h])
        qL = kin.leg_ik_full(self.body, 'L', pel, I, np.array([0, FY, HA]), I)
        self.body.d.qpos[self.idx.leg_q['L']] = qL
        qR = kin.leg_ik_full(self.body, 'R', pel, I, np.array([0, -FY, HA]), I)
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:3] = pel
        self.d.qpos[3:7] = [1, 0, 0, 0]
        self.d.qpos[self.idx.leg_q['L']] = qL
        self.d.qpos[self.idx.leg_q['R']] = qR
        mujoco.mj_forward(self.m, self.d)
        # put the spikes on the floor
        zmin = min(self.d.geom_xpos[g_][2] for g_ in self.idx.spikes.values()) - P.GEOM['spike_radius']
        self.d.qpos[2] -= zmin
        mujoco.mj_forward(self.m, self.d)
        self.stand_q = {'L': qL, 'R': qR}

    # ------------------------------------------------------------ helpers for closed-chain legs
    def ik_leg_ankle(self, side, ankle_target, pel_p):
        """Hip roll / hip pitch / knee setpoints that put the ankle at ankle_target for an upright pelvis
        at (pel_p[0], pel_p[1], design height)."""
        h = self._co_z if getattr(self, '_co_z', None) is not None else self.des['pelvis_height']
        pp = np.array([pel_p[0], pel_p[1], h])
        q = kin.leg_ik(side, pp, np.eye(3), ankle_target)
        return q

    def set_leg_ik(self, side, ankle_target, pel_p):
        q = self.ik_leg_ankle(side, ankle_target, pel_p)
        prev = self._qset_prev.get(side)
        qd = np.zeros(6) if prev is None else (q - prev) / self.dt
        qd = np.nan_to_num(qd)
        self._qset_prev[side] = q
        for j in (0, 1, 2, 3):
            self.pos(side, j, q[j], np.clip(qd[j], -5, 5))

    def flat_target(self, side, sw, rec):
        """Ankle target for a foot that should lie flat: the flat-foot ankle position over the first recorded
        contact line (heel at heel strike). The leg pushes the ankle there and the compliant ankle lets the
        sole conform to the floor."""
        if 'flat' not in rec:
            heel = sw[(side, 'heel_o')] or sw[(side, 'heel_i')]
            toe = sw[(side, 'toe_o')] or sw[(side, 'toe_i')]
            if not (heel or toe):
                return None
            key = 'heel' if heel else 'toe'
            o = kin.spike_offsets(side)
            r = P.GEOM['spike_radius']
            loc = 0.5 * (o[key + '_o'] + o[key + '_i']) + np.array([0, 0, r])
            pt_ = self.ankle_world(side) + self.foot_Rw(side) @ loc
            a = pt_ - loc
            a[2] = HA
            rec['flat'] = a
        return rec['flat']

    def contact_pivot_target(self, side, sw, rec):
        """Ankle target for a foot in contact: rotate about the recorded contact line using the
        actual foot orientation (a foot that rolls on heel or toe moves its ankle on an arc)."""
        R = self.foot_Rw(side)
        heel = sw[(side, 'heel_o')] or sw[(side, 'heel_i')]
        toe = sw[(side, 'toe_o')] or sw[(side, 'toe_i')]
        o = kin.spike_offsets(side)
        r = P.GEOM['spike_radius']
        if toe and not heel:
            key = 'toe'
        elif heel and not toe:
            key = 'heel'
        elif heel and toe:
            key = rec.get('flat_key', 'toe')
        else:
            # no contact (bounce): keep the last pivot; never record a pivot in the air
            key = rec.get('last')
            if key is None or key not in rec:
                return self.ankle_world(side)
        rec['last'] = key
        if key not in rec:
            # record the contact line centre (world) the first time it is used
            loc = 0.5 * (o[key + '_o'] + o[key + '_i']) + np.array([0, 0, r])
            rec[key] = self.ankle_world(side) + R @ loc
        loc = 0.5 * (o[key + '_o'] + o[key + '_i']) + np.array([0, 0, r])
        return rec[key] - R @ loc

    # ------------------------------------------------------------ logging
    def record(self, t, phase, sup, f=None):
        if f is None:
            f = self.spike_forces()
        z = self.measured_zmp(f)
        sw = self.switches(f)
        self.log.append(dict(t=t, phase=phase, sup=sup, qpos=self.d.qpos.copy(), qvel=self.d.qvel.copy(),
                             com=self.com(), zmp=z, sw={f'{k[0]}_{k[1]}': bool(v) for k, v in sw.items()},
                             ctrl_force=self.d.actuator_force.copy()))

    # ------------------------------------------------------------ main loop
    def step(self):
        mujoco.mj_step(self.m, self.d)
        self.t += self.dt
        self.k += 1

    def fallen(self):
        """Fall / invalid-gait detector: low or tilted pelvis, large yaw, or no floor contact for > 40 ms."""
        pel, R = self.pelvis()
        yaw = np.arctan2(R[1, 0], R[0, 0])
        if self.d.ncon == 0:
            self._air = getattr(self, '_air', 0) + 1
        else:
            self._air = 0
        return pel[2] < 0.55 or R[2, 2] < 0.85 or abs(yaw) > 0.25 or self._air * self.dt > 0.04

    def snapshot(self):
        dd = mujoco.MjData(self.m)
        mujoco.mj_copyData(dd, self.m, self.d)
        return (dd, self.m.actuator_biasprm.copy(), self.t, self.k)

    def restore(self, snap):
        """Exact continuation from a snapshot (full MjData copy, incl. solver warm start)."""
        dd, bias, t, k = snap
        mujoco.mj_copyData(self.d, self.m, dd)
        self.m.actuator_biasprm[:] = bias
        self.t = t; self.k = k
        self.log = []; self.events = []; self.supports = []; self.fail = None; self._qset_prev = {}

    def prepare_first_changeover(self):
        """Stand, shift and do the start step; return a snapshot at the start of phase A."""
        self.reset_standing(); self.t = 0.0; self.k = 0
        self._stand_and_shift('L')
        self._single_support('L', 'start', track=False)
        return self.snapshot()

    def run(self, verbose=False):
        g = self.des
        self.reset_standing()
        self.t = 0.0
        self.k = 0
        try:
            self._stand_and_shift('L')
            sup = 'L'
            double = g.get('stop_mode', 'double') == 'double'
            for n in range(self.n_steps):
                kind = 'start' if n == 0 else 'steady'
                self._single_support(sup, kind, track=(n > 0))
                last = n == self.n_steps - 1
                self._change_over(sup)
                sup = other(sup)
            if double:
                self._stop_static(sup)
            else:
                self._single_support(sup, 'stop', track=True)
                self._land_and_settle(sup)
        except _Fail as e:
            self.fail = str(e)
        if verbose:
            print('fail:', self.fail, 't=%.2f' % self.t)
        return self

    # ---- stand, weight shift
    def _stand_and_shift(self, sup):
        g = self.des
        s = SIDES[sup]
        h = g['pelvis_height']
        I = np.eye(3)
        aL = np.array([0, FY, HA]); aR = np.array([0, -FY, HA])
        T1, T2 = g['t_stand'], g['t_shift']
        t0 = self.t
        start = np.array([0.0, 0.0, h]); end = np.array([g['shift_x'], s * g['shift_y'], h])
        while self.t - t0 < T1 + T2:
            u = np.clip((self.t - t0 - T1) / T2, 0, 1)
            w = u ** 3 * (10 - 15 * u + 6 * u * u)   # minimum-jerk: zero velocity and acceleration at both ends
            pel = start + w * (end - start)
            qL = kin.leg_ik_full(self.body, 'L', pel, I, aL, I)
            self.body.d.qpos[self.idx.leg_q['L']] = qL
            qR = kin.leg_ik_full(self.body, 'R', pel, I, aR, I)
            self.pos_leg('L', qL); self.pos_leg('R', qR)
            self.step()
            if self.k % self.log_every == 0:
                self.record(self.t, 'stand' if u <= 0 else 'shift', 'LR')
            if self.fallen():
                raise _Fail('fell during weight shift')
        self.events.append((self.t, 'shift_end'))

    # ---- single support (program control + tracking mode)
    def _single_support(self, sup, kind, track):
        g = self.des
        pat = self.pat[(sup, kind)]
        t0 = self.t
        qe = {s_: self.q(s_) for s_ in SIDES}
        qp0L, qp0R = pat.q(0.0)
        qp0 = {'L': qp0L, 'R': qp0R}
        B = g['B_track']
        self.events.append((self.t, f'SS_{sup}_{kind}'))
        self.supports.append((self.t, sup, self.ankle_world(sup).tolist()))
        while self.t - t0 < pat.T - 1e-9:
            tt = self.t - t0
            qL, qR = pat.q(tt)
            vL, vR = pat.qd(tt)
            for s_, qq, vv in (('L', qL, vL), ('R', qR, vR)):
                if track and tt < B:
                    w = (B - tt) ** 3 / B ** 3
                    qq = qq + w * (qe[s_] - qp0[s_])
                    vv = vv - 3 * (B - tt) ** 2 / B ** 3 * (qe[s_] - qp0[s_])
                self.pos_leg(s_, qq, vv)
            self.step()
            if self.k % self.log_every == 0:
                self.record(self.t, 'SS', sup)
            if self.fallen():
                raise _Fail(f'fell in single support ({sup}, {kind})')
        self._qset_prev = {}

    # ---- change-over A-D
    def _ankle_mode(self, side, role, phase):
        """role 'H' (hind/rear) or 'F' (front); applies the table entry for pitch and roll."""
        tab = self.des['table']
        s = SIDES[side]
        for axis, j in (('P', AP), ('R', AR)):
            e = tab[role + axis][phase]
            if e == 'POS' and role == 'F' and phase == 'A':
                # swing foot before landing: keep the landing attitude in world (heel strike, toe up)
                self.pos(side, j, self._land_q(side)[j - AP], 0.0)
                continue
            if e == 'POS':
                key = (side, j)
                if key not in self._hold:
                    self._hold[key] = self.d.qpos[self.idx.leg_q[side][j]]
                self.pos(side, j, self._hold[key], 0.0)
            else:
                vis, ela, npd, trq = e
                npr = npd * np.pi / 180
                if axis == 'R':          # mirrored roll convention
                    self.imp(side, j, vis, ela, s * npr, s * trq)
                else:                    # pitch NP measured from the standing-over-the-foot posture
                    self.imp(side, j, vis, ela, self.q_ref_pitch + npr, trq)

    def _land_q(self, side):
        Rsh = self.d.xmat[self.m.body(f'{side}_shank').id].reshape(3, 3)
        Mm = Rsh.T @ kin.foot_R(self.des['S2_land_pitch'], 0.0)
        return (np.arctan2(-Mm[2, 0], Mm[0, 0]), np.arctan2(-Mm[1, 2], Mm[1, 1]))

    def _change_over(self, rear, final=False):
        g = self.des
        front = other(rear)
        t0 = self.t
        phase = 'A'
        self._hold = {}
        self._qset_prev = {}
        rec = {rear: {'flat_key': 'toe'}, front: {'flat_key': 'toe'}}
        a_r = self.ankle_world(rear); rec[rear]['flat0'] = np.array([a_r[0], a_r[1], HA])
        fA0 = self.ankle_world(front)
        pel0, _ = self.pelvis()
        rD0 = None
        deb = 0
        tD0 = np.inf
        x_end = fA0[0] + g['S0_x']
        self.events.append((self.t, f'A_{rear}->{front}'))
        self.supports.append((self.t, rear, self.ankle_world(rear).tolist()))
        self.co_log = dict(t0=t0)
        while True:
            f = self.spike_forces()
            sw = self.switches(f)
            pel, _ = self.pelvis()
            # pelvis height for the leg IK: from the end-of-step height to the next step's start height,
            # in proportion to the forward progress of the pelvis
            u = np.clip((pel[0] - pel0[0]) / max(1e-3, x_end - pel0[0]), 0.0, 1.0)
            self._co_z = pel0[2] + u * (g.get('S0_z', g['pelvis_height']) - pel0[2])
            # ---- phase transitions (sensor triggered, switch inputs debounced over P.SIM['debounce'])
            heelF = sw[(front, 'heel_o')] or sw[(front, 'heel_i')]
            toeF = sw[(front, 'toe_o')] or sw[(front, 'toe_i')]
            heelR = sw[(rear, 'heel_o')] or sw[(rear, 'heel_i')]
            cond = {'A': heelF or toeF, 'B': heelF and toeF, 'C': not heelR,
                    'D': (self.d.qpos[self.idx.leg_q[front][AP]] <= self.q_ref_pitch + g['D_end_angle']
                          and SIDES[front] * self.d.qpos[self.idx.leg_q[front][AR]] >= self.q_ref_roll - g['D_end_roll']
                          and all(sw[(front, k)] for k in ('toe_o', 'toe_i', 'heel_o', 'heel_i')))}[phase]
            if phase == 'D' and self.t - tD0 > g['D_max']:
                cond = True   # safety: hand over to the single-support pattern after D_max
            deb = deb + 1 if cond else 0
            if cond and deb * self.dt >= (P.SIM['debounce'] if phase != 'D' else 0.0):
                deb = 0
                if phase == 'A':
                    phase = 'B'; self._hold = {k: v for k, v in self._hold.items() if k[0] == rear}
                    self.events.append((self.t, 'B'))
                elif phase == 'B':
                    phase = 'C'; self.events.append((self.t, 'C'))
                elif phase == 'C':
                    phase = 'D'; self.events.append((self.t, 'D')); tD0 = self.t
                    self._hold = {}
                    rD0 = self.ankle_world(rear); pelD0 = pel.copy()
                else:
                    self.events.append((self.t, 'D_end'))
                    break
            if final and phase == 'C':
                # last step: stop in double support as soon as the front sole is flat (see _stop_double)
                self.events.append((self.t, 'stop_DS'))
                break
            if self.t - t0 > g['t_limit']:
                raise _Fail(f'change-over timeout in phase {phase}')
            # ---- legs
            if phase == 'A':
                tgt_f = fA0 + np.array([0, 0, -g['A_descent'] * max(0.0, pel[0] - pel0[0])])
                self.set_leg_ik(front, tgt_f, pel)
                self.set_leg_ik(rear, rec[rear]['flat0'], pel)
                self._ankle_mode(rear, 'H', 'A'); self._ankle_mode(front, 'F', 'A')
            elif phase in ('B', 'C'):
                tf = self.flat_target(front, sw, rec[front])
                self.set_leg_ik(front, tf if tf is not None else self.ankle_world(front), pel)
                if phase == 'B':
                    self.set_leg_ik(rear, rec[rear]['flat0'], pel)
                else:   # rear heel may lift: pivot about the toe line
                    rec[rear]['flat_key'] = 'toe'
                    self.set_leg_ik(rear, self.contact_pivot_target(rear, sw, rec[rear]), pel)
                self._ankle_mode(rear, 'H', phase); self._ankle_mode(front, 'F', phase)
            else:  # D: rear foot follows the locked-leg path (lifts as the body pivots on the front ankle)
                tf = self.flat_target(front, sw, rec[front])
                self.set_leg_ik(front, tf if tf is not None else self.ankle_world(front), pel)
                tgt_r = rD0 + np.array([0, 0, g['A_descent'] * max(0.0, pel[0] - pelD0[0])])
                self.set_leg_ik(rear, tgt_r, pel)
                self._ankle_mode(rear, 'H', 'D'); self._ankle_mode(front, 'F', 'D')
            self.step()
            if self.k % self.log_every == 0:
                self.record(self.t, phase, rear + front, f)
            if self.fallen():
                raise _Fail(f'fell in change-over phase {phase}')
        self._qset_prev = {}
        self._co_z = None

    # ---- final landing and settle
    def _land_and_settle(self, sup):
        g = self.des
        swg = other(sup)
        I = np.eye(3)
        a_sup = self.ankle_world(sup); a_sw = self.ankle_world(swg)
        pel0, _ = self.pelvis()
        h = g['pelvis_height']
        t0 = self.t
        T1, T2, T3 = 0.4, g['t_settle'], 0.5
        a_sw_end = np.array([a_sw[0], a_sw[1], HA])
        mid = 0.5 * (a_sup + a_sw_end)
        start = np.array([pel0[0], pel0[1], pel0[2]])
        v0 = np.array([self.d.qvel[0], self.d.qvel[1], 0.0])
        end = np.array([mid[0], mid[1], h])
        self.events.append((self.t, 'land'))
        self.supports.append((self.t, swg, self.ankle_world(swg).tolist()))
        Th = T1 + T2
        while self.t - t0 < T1 + T2 + T3:
            tt = self.t - t0
            u1 = np.clip(tt / T1, 0, 1)
            u = np.clip(tt / Th, 0, 1)
            # cubic Hermite from the measured pelvis state to rest over the middle of both feet
            pel = (2*u**3 - 3*u**2 + 1) * start + (u**3 - 2*u**2 + u) * Th * v0 + (-2*u**3 + 3*u**2) * end
            u2 = u
            aw = a_sw + u1 * (a_sw_end - a_sw)
            try:
                qs = kin.leg_ik_full(self.body, sup, pel, I, a_sup, I)
                self.body.d.qpos[self.idx.leg_q[sup]] = qs
                qw = kin.leg_ik_full(self.body, swg, pel, I, aw, I)
            except ValueError:
                raise _Fail('fell while stopping')
            self.pos_leg(sup, qs); self.pos_leg(swg, qw)
            self.step()
            if self.k % self.log_every == 0:
                self.record(self.t, 'land' if tt < T1 else 'settle', 'LR')
            if self.fallen():
                raise _Fail('fell while stopping')
        self.events.append((self.t, 'end'))


def _mj(u):
    u = np.clip(u, 0.0, 1.0)
    return u ** 3 * (10 - 15 * u + 6 * u * u)


class _StopMixin:
    def _stop_static(self, sup):
        """End of walk (the paper does not describe how its walks stop; README). After the last change-over
        the CoM is over the new support foot with little velocity (Fig. 12 start state), so: hold the pelvis
        where it is and bring the swing foot beside the support foot in a slow statically stable step, then
        centre the pelvis over both feet and stand up. All joints position-controlled."""
        g = self.des
        I = np.eye(3)
        swg = other(sup); s = SIDES[sup]
        aS = self.ankle_world(sup); aS = np.array([aS[0], aS[1], HA])
        aW0 = self.ankle_world(swg)
        aW1 = np.array([aS[0], aS[1] - s * 2 * FY, HA])
        p0, _ = self.pelvis()
        pC = np.array([aS[0], 0.5 * (aS[1] + aW1[1]), g['pelvis_height']])
        T_step, T_centre, T_hold = g.get('stop_T_step', 1.2), g.get('stop_T_centre', 2.0), 0.5
        lift = 0.05
        self.events.append((self.t, 'stop'))
        swR0 = self.foot_Rw(swg)

        def drive(pel, a_w, Rw, label):
            try:
                qs = kin.leg_ik_full(self.body, sup, pel, I, aS, I)
                self.body.d.qpos[self.idx.leg_q[sup]] = qs
                qw = kin.leg_ik_full(self.body, swg, pel, I, a_w, Rw)
            except ValueError:
                raise _Fail('fell while stopping')
            self.pos_leg(sup, qs); self.pos_leg(swg, qw)
            self.step()
            if self.k % self.log_every == 0:
                self.record(self.t, label, sup if label == 'stop' else 'LR')
            if self.fallen():
                raise _Fail('fell while stopping')

        p_hold = np.array([p0[0], p0[1], p0[2]])
        t0 = self.t
        while self.t - t0 < T_step:
            u = (self.t - t0) / T_step; w_ = _mj(u)
            a = aW0 + w_ * (aW1 - aW0)
            a[2] = (1 - w_) * aW0[2] + w_ * HA + lift * np.sin(np.pi * u)
            # swing foot attitude: from its attitude at the end of the change-over to flat
            ang = np.arctan2(swR0[0, 2], swR0[2, 2]) * (1 - _mj(min(1.0, 1.5 * u)))
            drive(p_hold, a, kin.foot_R(ang, 0.0), 'stop')
        t0 = self.t
        while self.t - t0 < T_centre + T_hold:
            w_ = _mj((self.t - t0) / T_centre)
            drive(p_hold + w_ * (pC - p_hold), np.array([aW1[0], aW1[1], HA - 0.001]), I, 'settle')
        self.events.append((self.t, 'end'))


class _Fail(Exception):
    pass


if __name__ == '__main__':
    import time
    t = time.time()
    w = Walker(n_steps=2).run(verbose=True)
    print('wall', time.time() - t)
    for e in w.events:
        print('%.3f %s' % e)


Walker._stop_static = _StopMixin._stop_static
