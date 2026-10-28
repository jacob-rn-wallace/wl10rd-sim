"""Run a complete walk and write the results.

    python -m wl10rd_sim.run_walk                      # tuned design, 4 change-overs
    python -m wl10rd_sim.run_walk --steps 4 --out out  # other step count / output folder

Writes  <out>/run.json      (time series, body poses, events)
        <out>/viewer.html   (self-contained 3D replay + charts; open in a browser)
        <out>/plots/*.pdf|png (paper-style figures)
"""
import argparse
import json
import os
from .design import load
from .sim import Walker
from .export import run_to_json
from . import plots

HERE = os.path.dirname(os.path.abspath(__file__))


def build_viewer(run_json_path, html_path):
    with open(os.path.join(HERE, 'viewer_template.html')) as f:
        tpl = f.read()
    with open(run_json_path) as f:
        data = f.read().replace('</', '<\\/')
    with open(html_path, 'w') as f:
        f.write(tpl.replace('__RUN_DATA__', data))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--design', default=os.path.join(HERE, 'designs', 'tuned.json'))
    ap.add_argument('--steps', type=int, default=4, help='number of change-overs (single-support steps = steps + 1)')
    ap.add_argument('--out', default='out')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    w = Walker(des=load(a.design), n_steps=a.steps, log_hz=100).run(verbose=True)
    p = os.path.join(a.out, 'run.json')
    run = run_to_json(w, p, fps=50, title='WL-10RD simulation')
    build_viewer(p, os.path.join(a.out, 'viewer.html'))
    print(plots.plot_all(run, os.path.join(a.out, 'plots')))
    print('events:', [(round(t, 2), e) for t, e in w.events])


if __name__ == '__main__':
    main()
