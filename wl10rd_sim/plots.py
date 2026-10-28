"""Paper-style plots of a simulated walk (black and white, like the figures of the 1985 paper).

    python -m wl10rd_sim.plots run.json out_dir
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PH_SHADE = {'A': '0.92', 'B': '0.85', 'C': '0.78', 'D': '0.70'}
plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.linewidth': 0.7,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'savefig.bbox': 'tight'})


def _spans(t, phase):
    out = []
    cur, t0 = phase[0], t[0]
    for ti, p in zip(t[1:], phase[1:]):
        if p != cur:
            out.append((cur, t0, ti)); cur, t0 = p, ti
    out.append((cur, t0, t[-1]))
    return out


def _shade(ax, t, phase):
    for p, a, b in _spans(t, phase):
        if p in PH_SHADE:
            ax.axvspan(a, b, color=PH_SHADE[p], lw=0)


def footprints(run):
    """Foot rectangles where each foot was flat on the floor (from the frame data)."""
    S = run['series']
    prints = []
    sw = S['sw']
    for side, k0 in (('L', 0), ('R', 4)):
        down = False
        for i, s in enumerate(sw):
            flat = s[k0:k0 + 4] == '1111'
            if flat and not down:
                prints.append((side, i))
            down = flat
    return prints


def plot_all(run, out):
    os.makedirs(out, exist_ok=True)
    S = run['series']
    t = np.array(S['t']); com = np.array(S['com']); pel = np.array(S['pelvis'])
    zmp = np.array([z if z is not None else [np.nan, np.nan] for z in S['zmp']])
    ph = S['phase']
    q = np.array(S['q']); names = run['joint_names']
    files = []

    # 1 plan view: CoM and measured ZMP over the footprints (cf. Fig.12)
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    L, W = 0.28, 0.14
    feet = run.get('footprints', [])
    for fx, fy in feet:
        ax.add_patch(plt.Rectangle((fx - L / 2, fy - W / 2), L, W, fill=False, lw=0.8, ec='k'))
    ax.plot(com[:, 0], com[:, 1], 'k-', lw=1.0, label='centre of gravity')
    ax.plot(zmp[:, 0], zmp[:, 1], 'k:', lw=0.6, label='ZMP (measured from the spike forces)')
    ax.set_aspect('equal'); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.32), fontsize=8, ncol=2)
    f = os.path.join(out, 'plan_view.pdf'); fig.savefig(f); fig.savefig(f.replace('.pdf', '.png'), dpi=200); plt.close(fig); files.append(f)

    # 2 lateral and forward CoM / ZMP vs time with change-over phases shaded
    fig, axs = plt.subplots(2, 1, figsize=(7.5, 4.2), sharex=True)
    for ax, k, lab in ((axs[0], 0, 'x [m]'), (axs[1], 1, 'y [m]')):
        _shade(ax, t, ph)
        ax.plot(t, com[:, k], 'k-', lw=1.0, label='CoG')
        ax.plot(t, zmp[:, k], 'k:', lw=0.8, label='ZMP')
        ax.set_ylabel(lab)
    axs[0].legend(frameon=False, fontsize=8, loc='upper left')
    axs[1].set_xlabel('time [s]')
    axs[0].set_title('Change-over phases A-D shaded light to dark', fontsize=8, loc='right')
    f = os.path.join(out, 'com_zmp_time.pdf'); fig.savefig(f); fig.savefig(f.replace('.pdf', '.png'), dpi=200); plt.close(fig); files.append(f)

    # 3 joint angles of one leg (cf. Fig.10/13 style)
    fig, axs = plt.subplots(6, 1, figsize=(7.5, 7.5), sharex=True)
    for j in range(6):
        ax = axs[j]
        _shade(ax, t, ph)
        ax.plot(t, q[:, j], 'k-', lw=0.9, label='left')
        ax.plot(t, q[:, 6 + j], 'k--', lw=0.8, label='right')
        ax.set_ylabel(names[j][2:].replace('_', ' ') + '\n[deg]', fontsize=8)
    axs[0].legend(frameon=False, fontsize=8, ncol=2, loc='upper right')
    axs[-1].set_xlabel('time [s]')
    f = os.path.join(out, 'joint_angles.pdf'); fig.savefig(f); fig.savefig(f.replace('.pdf', '.png'), dpi=200); plt.close(fig); files.append(f)

    # 4 ankle torques
    tau = np.array(S['tau_ankle'])
    fig, axs = plt.subplots(2, 1, figsize=(7.5, 3.8), sharex=True)
    for ax, k, lab in ((axs[0], 0, 'ankle pitch [N m]'), (axs[1], 1, 'ankle roll [N m]')):
        _shade(ax, t, ph)
        ax.plot(t, tau[:, k], 'k-', lw=0.9, label='left')
        ax.plot(t, tau[:, 2 + k], 'k--', lw=0.8, label='right')
        ax.set_ylabel(lab)
    axs[0].legend(frameon=False, fontsize=8, ncol=2)
    axs[1].set_xlabel('time [s]')
    f = os.path.join(out, 'ankle_torques.pdf'); fig.savefig(f); fig.savefig(f.replace('.pdf', '.png'), dpi=200); plt.close(fig); files.append(f)

    # 5 stick diagrams (cf. Fig.18): side and front views of one change-over and one single support
    st = run.get('sticks')
    if st:
        fig, axs = plt.subplots(2, 2, figsize=(7.0, 7.0))
        for (r, c), key, view in (((0, 0), 'co', 'side'), ((0, 1), 'co', 'front'), ((1, 0), 'ss', 'side'), ((1, 1), 'ss', 'front')):
            ax = axs[r, c]
            for fr in st[key]:
                for kind, seg in fr:
                    if (kind == 's' and view == 'front') or (kind == 'f' and view == 'side'):
                        continue
                    p = np.array(seg)
                    if view == 'side':
                        ax.plot(p[:, 0], p[:, 2], 'k-', lw=0.6)
                    else:
                        ax.plot(-p[:, 1], p[:, 2], 'k-', lw=0.6)
            ax.set_aspect('equal'); ax.axis('off')
            ax.set_title({'co': 'change-over', 'ss': 'single support'}[key] + f' ({view})', fontsize=8)
        f = os.path.join(out, 'stick_diagrams.pdf'); fig.savefig(f); fig.savefig(f.replace('.pdf', '.png'), dpi=200); plt.close(fig); files.append(f)
    f = paper_comparison(run, out)
    if f:
        files.append(f)
    return files


FIG8 = [(0.229, 0.594), (0.262, 0.653), (0.2947, 0.714), (0.3276, 0.634), (0.3645, 0.546), None,
        (0.3645, 0.283), (0.371, 0.299), (0.3842, 0.329), (0.3974, 0.355), (0.4105, 0.383), (0.4237, 0.403),
        (0.4336, 0.418), (0.4595, 0.355), (0.4855, 0.270), (0.5118, 0.185), (0.538, 0.100), (0.555, 0.046)]


def _ankle_near(feet, side, x, supports=None, t=None):
    if supports and t is not None:
        c = [s_ for s_ in supports if s_[1] == side and s_[0] <= t + 1e-6]
        if c:
            return np.array(c[-1][2][:2])
    c = [p for p in feet if (p[1] > 0) == (side == 'L')]
    return np.array(min(c, key=lambda p: abs(p[0] - x))) if c else None


def paper_comparison(run, out):
    """Simulated walk against the paper's Fig. 12 (CoM over the sole in single support) and Fig. 8
    (change-over phase plane). Both paper figures are design-stage computations, not measurements."""
    feet = run.get('footprints', [])
    if not feet:
        return None
    S = run['series']; t = np.array(S['t']); com = np.array(S['com']); ev = run['events']
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        f12 = json.load(open(os.path.join(here, 'paper_data', 'fig12_digitized.json')))
    except OSError:
        f12 = None
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.6), gridspec_kw=dict(width_ratios=[1, 1.6]))
    ax = axs[0]
    ax.add_patch(plt.Rectangle((-70, -140), 140, 280, fill=False, lw=0.8, ec='k'))
    if f12:
        cx, cy = f12['rect_center_px']; k = f12['scale_px_per_mm']
        mk = np.array(f12['com_markers_px'], float)
        ax.plot(-(cx - mk[:, 0]) / k, (cy - mk[:, 1]) / k, 's', mfc='none', mec='k', ms=4, mew=0.6, label='paper Fig. 12 (design)')
    for i, (ts, e) in enumerate(ev):
        if not (e.startswith('SS_') and e.endswith('steady')):
            continue
        side = e[3]; te = next((x for x, ee in ev[i + 1:] if not ee.startswith('SS')), ts + 1.0)
        m = (t >= ts) & (t <= te)
        a = _ankle_near(feet, side, com[m][0, 0], run.get('supports'), ts)
        if a is None:
            continue
        s = 1 if side == 'L' else -1
        ax.plot(s * (com[m][:, 1] - a[1]) * 1000, (com[m][:, 0] - a[0]) * 1000, 'k-', lw=0.8,
                label='simulation' if i == next(j for j, x in enumerate(ev) if x[1].startswith('SS_') and x[1].endswith('steady')) else None)
    ax.set_aspect('equal'); ax.set_xlabel('outboard  [mm]'); ax.set_ylabel('forward  [mm]')
    ax.set_title('CoM over the support sole (single support)', fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18))
    ax = axs[1]
    seg = [[]]
    for pnt in FIG8:
        if pnt is None:
            seg.append([])
        else:
            seg[-1].append(pnt)
    for k_, sg in enumerate(seg):
        sg = np.array(sg); ax.plot(sg[:, 0], sg[:, 1], 'k--', lw=1.0, label='paper Fig. 8 (simulated example)' if k_ == 0 else None)
    first = True
    for i, (ta, e) in enumerate(ev):
        if not e.startswith('A_'):
            continue
        rear, front = e[2], e[5]
        tb = next((x for x, ee in ev[i + 1:] if ee == 'D_end'), None)
        if tb is None:
            continue
        m = (t >= ta) & (t <= tb)
        ra = _ankle_near(feet, rear, com[m][0, 0], run.get('supports'), ta)
        fa = _ankle_near(feet, front, com[m][-1, 0], run.get('supports'), tb)
        if ra is None or fa is None:
            continue
        u = (fa - ra) / np.linalg.norm(fa - ra)
        d = (com[m][:, :2] - ra) @ u; h = com[m][:, 2] - 0.138
        th = np.arctan2(d, h); om = np.gradient(th, t[m])
        ax.plot(th, om, 'k-', lw=0.8, alpha=0.8, label='simulation' if first else None); first = False
    ax.set_xlabel('angle of CoM about the rear ankle  [rad]'); ax.set_ylabel('angular velocity  [rad/s]')
    ax.set_title('Change-over phase plane', fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    f = os.path.join(out, 'paper_comparison.pdf'); fig.savefig(f); fig.savefig(f.replace('.pdf', '.png'), dpi=200); plt.close(fig)
    return f


if __name__ == '__main__':
    run = json.load(open(sys.argv[1]))
    print(plot_all(run, sys.argv[2]))
