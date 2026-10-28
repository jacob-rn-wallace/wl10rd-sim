# CLAUDE.md — WL-10RD walking simulation

MuJoCo simulation of Waseda University's 1984–85 biped **WL-10RD** (Takanishi, Ishida, Yamazaki, Kato,
"The Realization of Dynamic Walking by the Biped Walking Robot WL-10RD", *JRSJ* 3(4), 1985). The aim is
to reproduce the paper's **own control method** as faithfully as the sources allow: preset single-support
patterns checked by ZMP, and a four-phase, sensor-switched ankle-impedance change-over. Read `README.md`
first. It records every estimate, every deviation from the paper, and the current results.

## Working on this project

- **Flag uncertainty explicitly.** Separate what a source says from what you estimated or inferred. Say
  "I don't know" rather than fill gaps.
- **Ground claims in the provided papers.** Cite figure, table or section numbers. The PDFs of the papers
  listed under "Sources" are not in this repo, so ask the user for a page when you need one.
- **Report results honestly**, including regressions. Do not tune quietly until something looks good and
  then present it as faithful. If a change makes the model more faithful but less robust, say so.
- **Ask before long tuning runs** (anything over about 30 min), and say what they are expected to settle.

## Layout

```
wl10rd-sim/                  repository root (Python package name: wl10rd_sim)
README.md                    estimates, deviations, results, fragility table (keep it current)
CLAUDE.md                    this file
wl10rd_sim/                  the package (run from the repo root: python -m wl10rd_sim.<module>)
  params.py                  geometry, masses, servos, sim constants; each tagged [paper] or [est]
  model.py                   MJCF builder: 12 DOF, freejoint torso, 4 spike spheres per sole (group 2, the only colliders)
  kin.py                     analytic leg IK + DLS refinement (yaw), Body scratch data, ZMP helpers
  pattern.py                 KeyState, 6th-order joint polynomial (pos/vel/acc at both ends, pos at Tm), tracking mode, evaluate()
  design.py                  DEFAULT design dict, TABLE3 (paper) / TABLE_SIM, design(), load(), save()
  sim.py                     Walker: reset/stand/shift, _single_support, _change_over (A-D), stop step, settling
  tune.py                    CMA-ES searches (patterns, change-over table, robust variant, stop step) + cost functions
  export.py, plots.py        run -> JSON for the viewer; paper-style PDFs/PNGs incl. paper_comparison (Figs. 8 and 12)
  viewer_template.html       standalone three.js viewer; run_walk.py fills the __RUN_DATA__ placeholder
  run_walk.py                CLI: run a walk -> out/run.json, out/viewer.html, out/plots/
  designs/tuned.json         the current tuned design (FULL float precision; never round it)
  paper_data/fig12_digitized.json  paper Fig. 12 digitized from the scan, with the reading notes
scripts/                     CLI drivers for tuning and checks (thin wrappers over wl10rd_sim.tune)
examples/                    viewer.html + plots/ from the current tuned design (regenerate after re-tuning)
```

## Setup and commands

```
pip install -r requirements.txt                       # pinned: mujoco 3.17.0, numpy 2.5.3, ... (Python 3.13 used)
python -m wl10rd_sim.run_walk --out out              # tuned design, 4 change-overs, ~1 min
python scripts/check_walk.py --design wl10rd_sim/designs/tuned.json      # robustness battery (README table)
python scripts/trace_changeover.py --design wl10rd_sim/designs/tuned.json # text trace of one change-over
python scripts/tune_patterns.py  --kind steady --design IN.json --out OUT.json      # ~1-2 min
python scripts/tune_changeover.py --design IN.json --steps 4 --full --out OUT.json  # ~20-30 min
python scripts/tune_stop.py      --design IN.json --steps 4 --out OUT.json          # ~5-10 min
```

Run long tuning jobs in the background with logs (`nohup ... > runs/x.log 2>&1 &`), and run two seeds in
parallel if there are cores. All designs are JSON: `design.load()` / `design.save()`.

**Done means:**
- `check_walk` and `run_walk` were run on the new design;
- `paper_comparison.png` was looked at;
- `examples/` was regenerated (`python -m wl10rd_sim.run_walk --out examples`, then delete `examples/run.json`,
  which is large);
- the README "Results" and "Fragility" sections were updated with the real numbers;
- the notes panel in `viewer_template.html` ("What is from the paper and what is not") still matches.

## How the walk runs (`Walker.run`)

1. **Stand and shift:** stand at `pelvis_height`, then shift the weight with a min-jerk profile (zero
   acceleration at the ends) so the start step begins static.
2. **Steps:** for each step, a single-support step, then a change-over. The first step is the `start`
   pattern, later ones are `steady`.
   - **Single support:** all 12 joints position-servoed to the pattern. When `track` is on, Eq. 45
     tracking mode runs for the first `B_track` s.
   - **Change-over, phases A→B→C→D:** switches are debounced over 10 ms.
     - A ends on front heel or toe contact.
     - B ends when the front foot has heel and toe down.
     - C ends when the rear heel opens.
     - D ends when the front ankle pitch is at or below `q_ref_pitch + D_end_angle`, the roll is within
       `D_end_roll` of the next pattern's start and all four front spikes are loaded. There is a
       `D_max` time-out.
     - Ankles run Eq. 33 impedance per `table[role+axis][phase]`.
     - Hips and knees follow IK from the *measured* pelvis, so the feet stay on their footprints.
3. **Stop** (`stop_mode='single'`): a `stop` pattern sets the swing foot down beside the support foot
   while the body still moves inward. `_land_and_settle` then Hermite-blends the pelvis from its measured
   state to rest over both feet. (`'double'` = slow static step after the last change-over; less reliable.)

## Conventions (easy to get wrong)

- **Frame.** x forward, y left, z up. Each pattern is built in a local frame with the support ankle at
  `(0, s*FY, HA)`, where `s` = +1 for L and −1 for R (`SIDES`).
- **Key-state lateral `*_y`.** Measured from the midline *toward the support side*. The velocity
  `*_vy` has the same sign convention.
- **Hand-over targets `tgt_S0/S2(v)`.** Coordinates are (x forward, u lateral), with **+u outboard** of the
  support ankle.
- **Ankle pitch.** Joint `q > 0` = toe down relative to the shank. Torque τ > 0 lifts the heel and brakes
  the body; τ < 0 presses the heel and pushes the body forward.
- **Roll.** Mirrored per side: (`s*q`) > 0 = hip outboard of the foot. Table roll NP and TRQ are given in
  this mirrored sense. The rear roll TRQ must be **negative** to push the body toward the front foot.
- **Pitch NP.** Relative to `q_ref_pitch`, the support-ankle pitch at the start of a steady step. The
  paper's 0° is "vertical shank", which has no clean equivalent with bent knees.
- **Eq. 45.** Implemented with the sign that converges (θ_r(0) = θ_e). The printed sign is reversed.
- **Table entry.** `(VIS, ELA, NP_deg, TRQ)` or `'POS'`. Search bounds (`FIELD_BOUNDS`): TRQ ±100 N·m
  (paper's range), VIS 10–600, ELA 0–1600.
- **Override keys.** `'<kind>.<name>'` (e.g. `stop.S1_x`) overrides `<name>` for that pattern kind
  (`Walker._g`).

## Gotchas learned the hard way

- **The dynamics are chaotic.** Rounding the table to 0.01 broke a tuned walk. Always save at full
  precision with `design.save`. A different MuJoCo or numpy version may not reproduce
  `designs/tuned.json` exactly either. If `check_walk` fails on a fresh install, check the versions
  against `requirements.txt` before debugging the model.
- **Snapshots must be exact.** `Walker.snapshot()/restore()` copies the whole `MjData` with
  `mj_copyData`, plus `actuator_biasprm`. The earlier version used `qpos/qvel` + `mj_forward`. That
  changed the solver warm start and the one-step-stale sensor data, so tuned results did not reproduce in
  a full run. Do not "simplify" this.
- **Any model change invalidates the tuned table.** That includes masses, servos, spike geometry,
  `pelvis_height`, the patterns and `params.SIM`. Re-tune in order:
  1. Patterns: `steady`, then `start` and `stop`. This step is kinematic.
  2. `tune_changeover --steps 1`, then `--steps 4 --full`.
  3. `tune_stop`.
  4. `check_walk`.
- **CMA reports.** CMA's `es.result.xbest` can be worse than the start. The tuners now keep the best
  design seen, including the start. Keep it that way.
- **Measured vs designed ZMP.** The designed ZMP (`pattern.evaluate`) matches MuJoCo `mj_inverse` exactly.
  The *measured* ZMP in single support swings about ±4 cm rms around it, from a body-on-ankle-servo
  mode. Results from the ankle servo sweep:
  - 20000/250: 2–3 Hz rocking;
  - 60000/1500: about 60 Hz chatter;
  - 40000/800: the current compromise.
- **Debounce trap.** The `if cond and deb*dt >= ...` guard on phase transitions matters: with zero
  debounce, D used to end on its first step.
- **Contact pivots.** Contact-pivot targets must never record a pivot while the foot is in the air.
  Front feet in B–D use a fixed flat target (`flat_target`).
- **Dropped approaches.** Already tried and dropped; don't repeat without a new idea:
  - rigidly locked legs in the change-over: the CoM cannot move along the diagonal with both feet flat;
  - a preset pelvis path for the hips in the change-over: it makes the pelvis rotate about the free
    ankles;
  - whole-walk CMA from standing (`tune_walk`): no gradient, wasted hours;
  - robust (multi-snapshot) change-over tuning: it traded away the nominal walk;
  - a static stop after `D_end`: the CoM is not at rest there.

## Sources (PDFs not in this repo; key facts already extracted)

- **WL-10RD, JRSJ 1985 (main paper).**
  - Geometry: Fig. 16.
  - Mass: 83 kg.
  - Gait: 40 cm steps, 1.0 s single support + 0.5 s change-over.
  - Control: Eq. 33, Tables 1–3, Fig. 7 phases.
  - Ankle torque range: ±100 N·m.
  - Fig. 8 (change-over phase plane, a simulated example): θ 0.23→0.56 rad.
  - Fig. 12 (designed CoM/ZMP over the sole): digitized in `wl10rd_sim/paper_data/`.
  - Fig. 18: stick diagrams.
  - An English translation was produced earlier in this project.
- **WL-10R, RoManSy '84 (pub. 1985).** About 80 kg; CFRP frame; 4 micro switches per sole;
  quasi-dynamic gait at 4.8 s/step; Figs. 7 and 10 show nearly straight support legs; Fig. 11 is a phase
  plane.
- **Advanced Robotics 1989 (WL-12R, known external force).** Says the WL-10RD walked **1.3 s/step** in
  1984 (vs 1.5 s in the JRSJ paper; unresolved). It cites an English **ICAR '85** WL-10RD paper, which
  has not been obtained.
- **Others checked, no WL-10RD masses:** JRSJ 1983 (quasi-dynamic walking), IROS '89 and IROS '90
  (WL-12R/RIII), and Li, Takanishi & Kato's 1992 JRSJ paper on ZMP measurement with six-axis
  force-moment sensors (identified from its abstract).
- **No source gives link masses, servo stiffness, spike positions or how walks were stopped.** These are
  `[est]` in `params.py` / `design.py`.

## Current status and open problems (most useful first)

The tuned design completes stand → 5 single-support steps / 4 change-overs → stop → settle, in 13.8 s.
It is fragile (see the README table).

1. **Support foot rides its outer edge in single support:** all-four-spike contact is 0–7 % of each steady
   step. The likely cause is lateral hand-over error: the CoM enters single support 2–3 cm too far
   inboard. An attempt to penalise the measured ZMP during single support (`edge_weight` in
   `changeover_cost`) did not converge in 40 iterations.
2. **Phase D always ends on the `D_max` time-out**, not on the paper's ankle-angle trigger.
3. **Change-over takes about 0.8 s** (paper 0.5 s).
4. **Robustness:** ±0.02 m/s initial sideways velocity, or ±1 kg, makes it fall by the 3rd change-over. A
   5th change-over fails.
5. **Mass split is the largest unknown.** CoM height is calibrated to ~0.77 m above the ankle from
   Figs. 8 and 12 combined (an interpretation; see the README and `paper_data/fig12_digitized.json`).
