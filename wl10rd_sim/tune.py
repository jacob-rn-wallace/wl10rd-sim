"""Parameter search by simulation (the paper: 'determined by motion simulation, finally by walking experiments').

tune_pattern(kind)    : single-support pattern - maximise the worst ZMP margin inside the support sole,
                        keep the swing foot clear of the floor (kinematic + inverse-dynamic evaluation).
tune_changeover()     : change-over table - start from the paper's Table 3, search for values that carry
                        the body from the pattern's end state to the next pattern's start state.
"""
import json
import numpy as np
import cma
import mujoco
from . import pattern as pt
from .design import design as make_design
from .sim import Walker

PATTERN_KEYS = {
    'steady': ['S0_x', 'S0_y', 'S0_vx', 'S0_vy', 'S0_z', 'S0_toe_pitch', 'S0_ax', 'S0_ay',
               'S2_x', 'S2_y', 'S2_vx', 'S2_vy', 'S2_z', 'S2_ax', 'S2_ay',
               'S1_x', 'S1_y', 'S1_z', 'S1_sw_x', 'S1_sw_h', 'Tm'],
    'start': ['start.S1_x', 'start.S1_y', 'start.S1_z', 'start.S1_sw_x', 'start.S1_sw_h', 'start.Tm', 'shift_y'],
    'stop': ['stop.S1_x', 'stop.S1_y', 'stop.S1_z', 'stop.S1_sw_x', 'stop.S1_sw_h', 'stop.Tm', 'stop_S2_x', 'stop_S2_y'],
}
BOUNDS = {'S1_x': (-0.12, 0.15), 'S1_y': (0.08, 0.24), 'S1_sw_x': (-0.15, 0.15), 'S1_sw_h': (0.03, 0.12), 'Tm': (0.3, 0.7),
          'S0_x': (-0.15, 0.10), 'S0_y': (0.08, 0.24), 'S0_vx': (-0.2, 0.5), 'S0_vy': (-0.3, 0.3),
          'S2_x': (-0.05, 0.25), 'S2_y': (-0.05, 0.22), 'S0_z': (0.78, 0.92), 'S1_z': (0.80, 0.93), 'S2_z': (0.80, 0.93),
          'S0_toe_pitch': (0.1, 0.7), 'S0_ax': (-3.0, 3.0), 'S0_ay': (-3.0, 3.0), 'S2_ax': (-3.0, 3.0), 'S2_ay': (-3.0, 3.0), 'S2_vx': (0.0, 0.6), 'S2_vy': (-0.45, 0.1),
          'shift_y': (0.12, 0.22), 'stop_S2_x': (-0.08, 0.15), 'stop_S2_y': (0.0, 0.22),
          'stop_S2_vx': (0.0, 0.3), 'stop_S2_vy': (-0.4, 0.05)}
SWAY_WEIGHT = 0.05

# The single-support pattern hands the change-over a CoM state on the diagonal between the feet
# (paper Fig.5: the CoM moves in the vertical plane through both ankles). Coordinates relative to the
# support ankle: x forward, u lateral (+ = away from the body midline).
_DIAG = np.array([0.40, -0.26]) / np.hypot(0.40, 0.26)      # outgoing direction (towards the next foot)
_DIAG_IN = np.array([0.40, 0.26]) / np.hypot(0.40, 0.26)    # incoming direction (from the previous foot)


def com_targets(des):
    """CoM hand-over states (x forward, u lateral + outboard) relative to the support ankle."""
    return dict(p2=np.array(des['tgt_S2']), v2=np.array(des['tgt_S2v']),
                p0=np.array(des['tgt_S0']), v0=np.array(des['tgt_S0v']))


def _base(k):
    return k.split('.')[-1]


def pattern_score(w, des, kind, detail=False):
    w.des = des
    try:
        S0, S1, S2 = w._ss_states('L', kind)
        pat = pt.SSPattern(w.body, 'L', S0, S1, S2, des['T'], w._g(kind, 'Tm'))
        ev = pt.evaluate(w.body, pat, n=81)
    except Exception as e:  # unreachable key state
        return (10.0, None) if detail else 10.0
    m = ev['margin']
    n_end = len(m)
    soft = -0.003 * np.log(np.mean(np.exp(-m[:n_end] / 0.003)))
    cost = -soft
    t = ev['t'] / des['T']
    # swing clearance window: a swing foot starting (start step) or ending (stop step) flat on the floor
    # needs time to lift / set down
    mid = (t > (0.25 if kind == 'start' else 0.12)) & (t < (0.75 if kind == 'stop' else 0.88))
    clr = ev['clearance'][mid].min()
    cost += 100 * max(0.0, 0.02 - clr)
    cost += 5 * max(0.0, -ev['clearance'].min())
    cost += 0.5 * max(0.0, np.abs(ev['pelvis_pitch']).max() - 0.05)
    cost += 0.1 * max(0.0, ev['clearance'][mid].max() - 0.20)
    errs = {}
    # CoM hand-over states (L support: ankle at (0, 0.13); u = y - 0.13)
    tg = com_targets(des)
    def xu(v):
        return np.array([v[0], v[1] - 0.13])
    def xu_v(v):
        return np.array([v[0], v[1]])
    if kind in ('steady', 'start'):
        errs['p2'] = xu(ev['com'][-1]) - tg['p2']; errs['v2'] = xu_v(ev['comv'][-1]) - tg['v2']
    if kind in ('steady', 'stop'):
        errs['p0'] = xu(ev['com'][0]) - tg['p0']; errs['v0'] = xu_v(ev['comv'][0]) - tg['v0']
    for k, e in errs.items():
        cost += (100.0 if k[0] == 'p' else 10.0) * float(e @ e)
    if detail:
        return cost, dict(min_margin=float(m[:n_end].min()), clearance=float(clr), ev=ev,
                          errs={k: np.round(v, 3).tolist() for k, v in errs.items()})
    return cost


def tune_pattern(des, kind, iters=60, sigma=0.3, seed=1, verbose=True, popsize=12):
    keys = PATTERN_KEYS[kind]
    w = Walker(des=des, n_steps=1)
    lo = np.array([BOUNDS[_base(k)][0] for k in keys]); hi = np.array([BOUNDS[_base(k)][1] for k in keys])
    x0 = np.array([des.get(k, des[_base(k)]) for k in keys])
    z0 = (x0 - lo) / (hi - lo)

    def f(z):
        z = np.clip(z, 0, 1)
        d = dict(des)
        for k, v in zip(keys, lo + z * (hi - lo)):
            d[k] = float(v)
        return pattern_score(w, d, kind)

    es = cma.CMAEvolutionStrategy(np.clip(z0, 0.05, 0.95), sigma, {'bounds': [0, 1], 'seed': seed, 'verbose': -9, 'popsize': popsize})
    best = (pattern_score(w, dict(des), kind), dict(des))   # keep the start if nothing beats it
    for it in range(iters):
        Z = es.ask()
        F = [f(z) for z in Z]
        es.tell(Z, F)
        i = int(np.argmin(F))
        if F[i] < best[0]:
            d = dict(des)
            for k, v in zip(keys, lo + np.clip(Z[i], 0, 1) * (hi - lo)):
                d[k] = float(v)
            best = (F[i], d)
        if verbose and it % 20 == 0:
            print(kind, it, round(best[0], 4), flush=True)
    out = best[1]
    c, det = pattern_score(w, out, kind, detail=True)
    if verbose:
        print(kind, 'best cost', round(c, 4), 'min ZMP margin %.3f m, swing clearance %.3f m' % (det['min_margin'], det['clearance']), det['errs'], flush=True)
    return out, det


# ---------------------------------------------------------------- change-over search
# (role/axis, phase, field) ; fields: 0 VIS [N m s/rad], 1 ELA [N m/rad], 2 NP [deg], 3 TRQ [N m]
CO_PARAMS = [
    ('HP', 'A', 0), ('HP', 'A', 3), ('HP', 'B', 0), ('HP', 'B', 3),
    ('HP', 'C', 0), ('HP', 'C', 1), ('HP', 'C', 2), ('HP', 'C', 3),
    ('HR', 'A', 0), ('HR', 'A', 3), ('HR', 'B', 0), ('HR', 'B', 3), ('HR', 'C', 0), ('HR', 'C', 3),
    ('FP', 'B', 0), ('FP', 'B', 1), ('FP', 'B', 2), ('FP', 'B', 3), ('FP', 'C', 0), ('FP', 'C', 1), ('FP', 'C', 2), ('FP', 'C', 3),
    ('FP', 'D', 0), ('FP', 'D', 1), ('FP', 'D', 2), ('FP', 'D', 3),
    ('FR', 'B', 0), ('FR', 'B', 3), ('FR', 'C', 0), ('FR', 'C', 1), ('FR', 'C', 2), ('FR', 'C', 3),
    ('FR', 'D', 0), ('FR', 'D', 1), ('FR', 'D', 2), ('FR', 'D', 3),
]
FIELD_BOUNDS = {0: (10.0, 600.0), 1: (0.0, 1600.0), 2: (-50.0, 30.0), 3: (-100.0, 100.0)}   # TRQ: paper's +-100 N m
EXTRA_CO = [('D_end_angle', -0.25, 0.25), ('A_descent', 0.3, 1.2)]


def co_vector(des):
    v = []
    for role, ph, f in CO_PARAMS:
        v.append(des['table'][role][ph][f])
    for k, lo, hi in EXTRA_CO:
        v.append(des[k])
    return np.array(v, float)


def co_apply(des, v):
    import copy
    d = copy.deepcopy(des)
    for (role, ph, f), x in zip(CO_PARAMS, v):
        e = list(d['table'][role][ph])
        e[f] = float(x)
        d['table'][role][ph] = tuple(e)
    for (k, lo, hi), x in zip(EXTRA_CO, v[len(CO_PARAMS):]):
        d[k] = float(x)
    return d


def co_bounds():
    lo = [FIELD_BOUNDS[f][0] for _, _, f in CO_PARAMS] + [e[1] for e in EXTRA_CO]
    hi = [FIELD_BOUNDS[f][1] for _, _, f in CO_PARAMS] + [e[2] for e in EXTRA_CO]
    return np.array(lo), np.array(hi)


def changeover_cost(w, snap, des, detail=False, n_steps=1, full=False):
    """Run phase A..D from the snapshot and the following single-support step(s); score the result.
    Failures get partial credit for progress so the search has a slope to follow."""
    from .sim import _Fail, SIDES, other
    w.des = des
    w.restore(snap)
    g = des
    rear = 'L'
    cost = 0.0
    info = dict(steps=[])
    t_start = w.t
    for step in range(n_steps):
        front = other(rear)
        s = SIDES[front]
        t0 = w.t
        n_ev0 = len(w.events)
        try:
            w._change_over(rear)
        except _Fail as e:
            ev = [x[1] for x in w.events[n_ev0:]]
            prog = sum(1 for p in ('B', 'C', 'D') if p in ev)
            com = w.com(); a = w.ankle_world(front); r_ = w.ankle_world(rear)
            lat = np.clip((com[1] - r_[1]) / (a[1] - r_[1]), 0, 1)
            cost += 40.0 * (n_steps - step) - 3 * prog - 4 * lat - 2 * min(w.t - t0, 1.0)
            info['fail'] = str(e)
            return (cost, info) if detail else cost
        t_co = w.t - t0
        com = w.com(); a = w.ankle_world(front)
        mujoco.mj_subtreeVel(w.m, w.d)
        cv = w.d.subtree_linvel[w.idx.torso].copy()
        tg = com_targets(g)
        ex, ey = (com[0] - a[0]) - tg['p0'][0], s * (com[1] - a[1]) - tg['p0'][1]
        evx, evy = cv[0] - tg['v0'][0], s * cv[1] - tg['v0'][1]
        ev = w.events[n_ev0:]
        tE = {e[1]: e[0] for e in ev}
        c = 100 * (ex ** 2 + ey ** 2) + 1.0 * (evx ** 2 + evy ** 2) + 10.0 * (t_co - 0.5) ** 2   # paper: 0.5 s
        for p1, p2 in (('B', 'C'), ('C', 'D'), ('D', 'D_end')):
            if p1 in tE and p2 in tE:
                c += 0.5 * max(0.0, 0.03 - (tE[p2] - tE[p1])) / 0.03
        cost += c
        info['steps'].append(dict(t_co=t_co, ex=ex, ey=ey, evx=evx, evy=evy,
                                  phases={k: round(v - t0, 3) for k, v in tE.items()}))
        t1 = w.t
        last = step == n_steps - 1
        nlog = len(w.log)
        try:
            w._single_support(front, 'stop' if (last and full) else 'steady', track=True)
            # paper: keep all four spikes of the support sole loaded -> penalise a measured ZMP near the sole edges
            a = w.ankle_world(front); sgn = SIDES[front]
            zs = [r['zmp'] for r in w.log[nlog:] if not np.any(np.isnan(r['zmp']))]
            if zs:
                zs = np.array(zs)
                du = np.maximum(0.0, np.abs(sgn * (zs[:, 1] - a[1])) - 0.035)
                dx = np.maximum(0.0, np.abs(zs[:, 0] - a[0]) - 0.09)
                cost += g.get('edge_weight', 0.0) * float(np.mean(du ** 2 + dx ** 2)) * 1e4
            if last and full:
                w._land_and_settle(front)
        except _Fail as e:
            cost += 40.0 * (n_steps - step) - 18 - 8 * min(w.t - t1, 1.0) + min(c, 3.0) - c
            if last and full and 'stopping' in str(e):
                cost -= 8.0
            info['fail'] = str(e)
            return (cost, info) if detail else cost
        rear = front
    return (cost, info) if detail else cost


def tune_changeover(des, iters=60, sigma=0.15, popsize=14, seed=3, verbose=True, snap=None, w=None, n_steps=1, full=False):
    """CMA on the change-over table from an exact snapshot at the first change-over. Keeps the best design
    seen, including the starting one (CMA's own xbest can be worse than the start)."""
    w = w or Walker(des=des, n_steps=1)
    if snap is None:
        snap = w.prepare_first_changeover()
    lo, hi = co_bounds()
    x0 = np.clip((co_vector(des) - lo) / (hi - lo), 0.02, 0.98)

    def f(z):
        return changeover_cost(w, snap, co_apply(des, lo + np.clip(z, 0, 1) * (hi - lo)), n_steps=n_steps, full=full)

    best = (changeover_cost(w, snap, des, n_steps=n_steps, full=full), des)
    es = cma.CMAEvolutionStrategy(x0, sigma, {'bounds': [0, 1], 'seed': seed, 'verbose': -9, 'popsize': popsize})
    for it in range(iters):
        Z = es.ask()
        F = [f(z) for z in Z]
        es.tell(Z, F)
        i = int(np.argmin(F))
        if F[i] < best[0]:
            best = (F[i], co_apply(des, lo + np.clip(Z[i], 0, 1) * (hi - lo)))
        if verbose and it % 5 == 0:
            print('changeover', it, round(best[0], 4), flush=True)
    c, det = changeover_cost(w, snap, best[1], detail=True, n_steps=n_steps, full=full)
    if verbose:
        print('best', round(c, 4), det)
    return best[1], det


STOP_KEYS = ['stop.S1_x', 'stop.S1_y', 'stop.S1_z', 'stop.S1_sw_x', 'stop.S1_sw_h', 'stop.Tm', 'stop_S2_x', 'stop_S2_y',
             'stop_S2_vx', 'stop_S2_vy']
EVENT_WEIGHT = {'A': 1.0, 'B': 0.5, 'C': 0.5, 'D': 0.5, 'D_end': 1.0, 'SS': 1.0, 'land': 2.0, 'end': 4.0}


def walk_score(des, n_steps, perturb=(0.0,)):
    """Cost of a complete walk run from standing (no snapshots): progress through the event sequence,
    then the hand-over quality. Averaged over small initial perturbations of the pelvis (robustness)."""
    tot = 0.0
    for dy in perturb:
        w = Walker(des=des, n_steps=n_steps, log_hz=20)
        if dy:
            orig = w.reset_standing
            def rs(orig=orig, dy=dy):
                orig(); w.d.qvel[1] = dy
            w.reset_standing = rs
        w.run()
        prog = 0.0
        for _, e in w.events:
            prog += EVENT_WEIGHT.get(e.split('_')[0] if not e.startswith('D_end') else 'D_end', 0.0)
        # time spent in the last (failed) segment counts a little
        c = -prog - 0.5 * min(w.t, 30) / 30
        if w.fail is None:
            c -= 10.0
        tot += c
    return tot / len(perturb)


def tune_walk(des, n_steps=3, iters=40, sigma=0.05, popsize=12, seed=1, perturb=(0.0, 0.02), keys_extra=STOP_KEYS):
    lo_c, hi_c = co_bounds()
    lo_s = np.array([BOUNDS[_base(k)][0] for k in keys_extra]); hi_s = np.array([BOUNDS[_base(k)][1] for k in keys_extra])
    lo = np.concatenate([lo_c, lo_s]); hi = np.concatenate([hi_c, hi_s])
    x0 = np.concatenate([co_vector(des), [des.get(k, des[_base(k)]) for k in keys_extra]])

    def mk(z):
        x = lo + np.clip(z, 0, 1) * (hi - lo)
        d = co_apply(des, x[:len(lo_c)])
        for k, v in zip(keys_extra, x[len(lo_c):]):
            d[k] = float(v)
        return d

    es = cma.CMAEvolutionStrategy(np.clip((x0 - lo) / (hi - lo), 0.02, 0.98), sigma,
                                  {'bounds': [0, 1], 'seed': seed, 'verbose': -9, 'popsize': popsize})
    best = (np.inf, None)
    for it in range(iters):
        Z = es.ask()
        F = [walk_score(mk(z), n_steps, perturb) for z in Z]
        es.tell(Z, F)
        i = int(np.argmin(F))
        if F[i] < best[0]:
            best = (F[i], mk(Z[i]))
        print('walk', it, round(best[0], 3), flush=True)
    return best[1], best[0]


def perturbed_snapshots(w, snap, dvs=((0, 0), (0, 0.03), (0, -0.03), (0.03, 0))):
    """Copies of a snapshot with small extra pelvis velocity (x, y) in m/s, for robust tuning."""
    out = []
    for dx, dy in dvs:
        w.restore(snap)
        w.d.qvel[0] += dx; w.d.qvel[1] += dy
        out.append(w.snapshot())
    return out


def tune_changeover_robust(des, snaps, w, iters=60, sigma=0.08, popsize=14, seed=1, n_steps=3, full=True):
    """CMA on the change-over table, scoring the sum over several perturbed starting snapshots."""
    lo, hi = co_bounds()
    x0 = np.clip((co_vector(des) - lo) / (hi - lo), 0.02, 0.98)
    def f(z):
        d = co_apply(des, lo + np.clip(z, 0, 1) * (hi - lo))
        return sum(changeover_cost(w, s, d, n_steps=n_steps, full=full) for s in snaps)
    es = cma.CMAEvolutionStrategy(x0, sigma, {'bounds': [0, 1], 'seed': seed, 'verbose': -9, 'popsize': popsize})
    best = (sum(changeover_cost(w, s, des, n_steps=n_steps, full=full) for s in snaps), des)
    for it in range(iters):
        Z = es.ask(); F = [f(z) for z in Z]; es.tell(Z, F)
        i = int(np.argmin(F))
        if F[i] < best[0]:
            best = (F[i], co_apply(des, lo + np.clip(Z[i], 0, 1) * (hi - lo)))
        print('robust', it, round(best[0], 3), flush=True)
    return best[1], best[0]


def stop_snapshots(des, n_steps, perturb=True):
    """Run walks (nominal and, if perturb, with the perturbed_snapshots kicks at the first change-over) up
    to the start of the stop step; return [(snapshot, support side)] for those that get there, and the walker."""
    from .sim import _Fail, other
    w = Walker(des=des, n_steps=n_steps, log_hz=20)
    snap0 = w.prepare_first_changeover()
    snaps = perturbed_snapshots(w, snap0) if perturb else [snap0]
    out = []
    for s in snaps:
        w.restore(s)
        sup = 'L'
        try:
            for k in range(n_steps):
                w._change_over(sup)
                sup = other(sup)
                if k < n_steps - 1:
                    w._single_support(sup, 'steady', track=True)
            out.append((w.snapshot(), sup))
        except _Fail:
            pass
    return out, w


def tune_stop(des, n_steps, iters=40, seed=1, sigma=0.2, popsize=12, perturb=True, verbose=True):
    """CMA on the stop-step pattern keys (+ t_settle), summed over the stop snapshots from stop_snapshots."""
    from .sim import _Fail
    stops, w = stop_snapshots(des, n_steps, perturb)
    if not stops:
        raise RuntimeError('no run reached the stop step')
    keys = STOP_KEYS + ['t_settle']
    B = dict(BOUNDS); B['t_settle'] = (0.8, 3.0)
    lo = np.array([B[_base(k)][0] for k in keys]); hi = np.array([B[_base(k)][1] for k in keys])
    x0 = np.clip((np.array([des.get(k, des[_base(k)]) for k in keys]) - lo) / (hi - lo), 0.05, 0.95)

    def mk(z):
        dd = dict(des)
        for k, v in zip(keys, lo + np.clip(z, 0, 1) * (hi - lo)):
            dd[k] = float(v)
        return dd

    def one(snap, sup):
        w.restore(snap); t0 = w.t
        try:
            w._single_support(sup, 'stop', track=True); w._land_and_settle(sup)
        except _Fail:
            return 20.0 - 5 * min(w.t - t0, 4.0)
        except Exception:
            return 50.0
        return -10.0 + float(np.abs(w.d.qvel[:6]).sum())

    def fd(dd):
        w.des = dd
        try:
            w.build_patterns()
        except Exception:
            return 50.0 * len(stops)
        return sum(one(s, sup) for s, sup in stops)

    def f(z):
        return fd(mk(z))

    es = cma.CMAEvolutionStrategy(x0, sigma, {'bounds': [0, 1], 'seed': seed, 'verbose': -9, 'popsize': popsize})
    best = (fd(dict(des)), dict(des))
    for it in range(iters):
        Z = es.ask(); F = [f(z) for z in Z]; es.tell(Z, F)
        i = int(np.argmin(F))
        if F[i] < best[0]:
            best = (F[i], mk(Z[i]))
        if verbose:
            print('stop', it, round(best[0], 3), f'({len(stops)} snapshots)', flush=True)
        if best[0] < -9.9 * len(stops):
            break
    return best[1], best[0]
