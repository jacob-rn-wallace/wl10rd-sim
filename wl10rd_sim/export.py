"""Export a simulated run for the browser viewer, and make paper-style plots."""
import json
import numpy as np
import mujoco
from .model import LEG_JOINTS, SIDES

GEOM_TYPES = {mujoco.mjtGeom.mjGEOM_BOX: 'box', mujoco.mjtGeom.mjGEOM_CYLINDER: 'cylinder',
              mujoco.mjtGeom.mjGEOM_SPHERE: 'sphere'}


def scene_geoms(m):
    out = []
    for g in range(m.ngeom):
        t = GEOM_TYPES.get(int(m.geom_type[g]))
        if t is None:
            continue
        out.append(dict(body=int(m.geom_bodyid[g]), type=t, size=[round(float(x), 4) for x in m.geom_size[g]],
                        pos=[round(float(x), 4) for x in m.geom_pos[g]], quat=[round(float(x), 5) for x in m.geom_quat[g]],
                        rgba=[round(float(x), 3) for x in m.geom_rgba[g]], spike=bool(m.geom_group[g] == 2)))
    return out


def run_to_json(w, path, fps=50, title='WL-10RD simulation'):
    m = w.m
    d = mujoco.MjData(m)
    frames = []
    idx = w.idx
    dt_log = w.log[1]['t'] - w.log[0]['t'] if len(w.log) > 1 else 0.01
    step = max(1, int(round((1.0 / fps) / dt_log)))
    jnames = [f'{s}_{j}' for s in SIDES for j in LEG_JOINTS]
    series = dict(t=[], phase=[], com=[], zmp=[], pelvis=[], q=[], sw=[], tau_ankle=[])
    for r in w.log:
        series['t'].append(round(r['t'], 3))
        series['phase'].append(r['phase'])
        series['com'].append([round(float(x), 4) for x in r['com']])
        z = r['zmp']
        series['zmp'].append(None if np.any(np.isnan(z)) else [round(float(x), 4) for x in z])
        series['pelvis'].append([round(float(x), 4) for x in r['qpos'][:3]])
        series['q'].append([round(float(np.degrees(r['qpos'][idx.qadr[n]])), 2) for n in jnames])
        series['sw'].append(''.join('1' if r['sw'][k] else '0' for k in
                                    ['L_toe_o', 'L_toe_i', 'L_heel_o', 'L_heel_i', 'R_toe_o', 'R_toe_i', 'R_heel_o', 'R_heel_i']))
        series['tau_ankle'].append([round(float(r['ctrl_force'][idx.act[f'{s}_{j}']]), 1) for s in SIDES for j in ('ankle_pitch', 'ankle_roll')])
    for r in w.log[::step]:
        d.qpos[:] = r['qpos']
        mujoco.mj_kinematics(m, d)
        fr = []
        for b in range(1, m.nbody):
            fr.extend([round(float(x), 4) for x in d.xpos[b]])
            fr.extend([round(float(x), 5) for x in d.xquat[b]])
        frames.append(dict(t=round(r['t'], 3), p=fr))
    # footprints (ankle xy of each flat-foot landing) and stick-diagram frames (cf. paper Fig.18)
    bid = {n: m.body(n).id for n in [f'{sd}_{b}' for sd in SIDES for b in ('thigh_upper', 'shank', 'foot')]}
    def sticks_at(r):
        d.qpos[:] = r['qpos']; mujoco.mj_kinematics(m, d)
        segs = []
        hips = [d.xpos[bid[f'{sd}_thigh_upper']].tolist() for sd in SIDES]
        segs.append(('b', hips))
        for sd in SIDES:
            fp = d.xpos[bid[f'{sd}_foot']]; R = d.xmat[bid[f'{sd}_foot']].reshape(3, 3)
            segs.append(('b', [d.xpos[bid[f'{sd}_thigh_upper']].tolist(), d.xpos[bid[f'{sd}_shank']].tolist(), fp.tolist()]))
            # sole as seen from the side (heel-toe) and from the front (outer-inner edge)
            segs.append(('b', [fp.tolist(), (fp + R @ np.array([0, 0, -0.138])).tolist()]))
            segs.append(('s', [(fp + R @ np.array([-0.14, 0, -0.138])).tolist(), (fp + R @ np.array([0.14, 0, -0.138])).tolist()]))
            segs.append(('f', [(fp + R @ np.array([0, -0.07, -0.138])).tolist(), (fp + R @ np.array([0, 0.07, -0.138])).tolist()]))
        return [[k, [[round(c, 4) for c in pnt] for pnt in sg]] for k, sg in segs]
    # footprints: where each foot was planted (ankle at the start of each support), incl. the start stance
    prints = []
    st0 = getattr(w, 'stand_q', None)
    for t_, sd, a in [(0.0, 'L', [0.0, 0.13, 0]), (0.0, 'R', [0.0, -0.13, 0])] + list(getattr(w, 'supports', [])):
        if not any(abs(a[0] - p[0]) < 0.05 and abs(a[1] - p[1]) < 0.05 for p in prints):
            prints.append((round(float(a[0]), 4), round(float(a[1]), 4)))
    down = {sd: False for sd in SIDES}
    for r in w.log:
        for sd in SIDES:
            flat = all(r['sw'][f'{sd}_{k}'] for k in ('toe_o', 'toe_i', 'heel_o', 'heel_i'))
            if flat and not down[sd]:
                d.qpos[:] = r['qpos']; mujoco.mj_kinematics(m, d)
                fp = d.xpos[bid[f'{sd}_foot']]
                pass
            down[sd] = flat
    sticks = {'co': [], 'ss': []}
    ev = [(t, e) for t, e in w.events]
    tA = [t for t, e in ev if e.startswith('A_')]
    tS = [t for t, e in ev if e.startswith('SS_') and 'steady' in e]
    if len(tA) > 1:
        t0 = tA[1]; t1 = next((t for t, e in ev if e == 'D_end' and t > t0), t0 + 0.5)
        sticks['co'] = [sticks_at(r) for r in w.log if t0 <= r['t'] <= t1][::max(1, int(0.04 / dt_log))]
    if tS:
        t0 = tS[0]
        sticks['ss'] = [sticks_at(r) for r in w.log if t0 <= r['t'] <= t0 + 1.0][::max(1, int(0.08 / dt_log))]
    out = dict(title=title, nbody=m.nbody - 1, geoms=scene_geoms(m), frames=frames, fps=fps,
               joint_names=jnames, series=series, events=[[round(t, 3), e] for t, e in w.events],
               fail=w.fail, footprints=prints,
               supports=[[round(t_, 3), sd, [round(float(x), 4) for x in a]] for t_, sd, a in getattr(w, 'supports', [])], sticks=sticks, design={k: v for k, v in w.des.items() if k != 'table'},
               table={k: {p: v for p, v in vv.items()} for k, vv in w.des['table'].items()})
    with open(path, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    return out
