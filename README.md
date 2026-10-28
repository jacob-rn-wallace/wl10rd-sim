# WL-10RD walking simulation (MuJoCo)

This package simulates the biped WL-10RD from Takanishi, Ishida, Yamazaki and Kato, "The Realization of
Dynamic Walking by the Biped Walking Robot WL-10RD", *J. Robotics Society of Japan* 3(4), 1985. It does a
complete walk: stand, shift the weight, take steps and stop again.

The control method is the paper's:

* **Single-support phase.** Program control of all 12 joints by a preset pattern. Three key states (start,
  knee-bent middle, end) are joined by a 6th-order joint-space polynomial. The angle, velocity and
  acceleration are set at both ends, and the result is checked by ZMP. At the start of each step the
  tracking mode of Eq. 45 is used.
* **Change-over phase.** Four phases, A to D. The sole micro switches (four per sole) and the front ankle
  angle end each phase. The ankle joints use the impedance law of Eq. 33,
  `tau = TRQ - VIS*dq - ELA*(q - NP)`, with a Table-3-style parameter set for each phase.

The masses, servo properties and several design values are **not** in any source we have, so they are
estimates. The change-over table and the walking patterns were found by search in this simulation. The
result is one way the method can work on this estimated model. It does not reconstruct the real robot's
numbers.

## Running it

```
pip install -r requirements.txt   # pinned versions; developed on Python 3.13
python -m wl10rd_sim.run_walk --out out     # tuned design: 4 change-overs, 5 single-support steps
                                            # (a copy of this output is in examples/)
```

This writes three things to `out/`:

* `run.json`: the time series.
* `viewer.html`: a self-contained 3D replay with charts. It needs an internet connection to load
  three.js.
* `plots/`: black-and-white figures, including `paper_comparison` (simulation vs the paper's Fig. 8
  and Fig. 12).

Tuning and check scripts live in `scripts/` (see `CLAUDE.md`, which also covers conventions, gotchas and
open problems for anyone, or any agent, continuing the work).

| file (in `wl10rd_sim/`) | contents |
|---|---|
| `params.py` | geometry, masses, servos, simulation constants, each tagged `[paper]` or `[est]` |
| `model.py` | MuJoCo model: 12 DOF, 4 spike spheres per sole (only the spikes collide) |
| `kin.py` | analytic leg IK (+ DLS refinement), ZMP from CoM and angular momentum |
| `pattern.py` | key states, 6th-order polynomial with end accelerations, tracking mode, ZMP evaluation |
| `design.py` | gait design values and change-over tables (`TABLE3` = the paper's) |
| `designs/tuned.json` | the design used for the published run (full precision; see "Fragility") |
| `paper_data/fig12_digitized.json` | the paper's Fig. 12, digitized from the scan, with how it was read |
| `sim.py` | walker: stand/shift, single support, change-over A–D, stop step, settling |
| `tune.py` | CMA-ES searches: pattern ZMP margin and hand-over states, change-over table, stop step |
| `export.py`, `plots.py`, `viewer_template.html`, `run_walk.py` | output |

## Sources used

* **The WL-10RD paper (JRSJ 1985).**
  * Geometry: Fig. 16.
  * Total mass: 83 kg.
  * Gait: 40 cm steps, 1.0 s single support, 0.5 s change-over.
  * Control method: Secs. 3–4, Tables 1–3.
  * Ankle torque range: ±100 N·m.
* **Fig. 12 (designed CoM and ZMP paths over the sole in single support).** Digitized from the 600 dpi
  scan.
  * Start of single support: the CoM is about over the support ankle (8 mm behind, 18 mm outboard) and
    nearly at rest.
  * End: the CoM is 167 mm ahead and 86 mm inboard, moving at 45° toward the next foot at roughly
    0.45 m/s.
  * How it was read: the figure has no arrows. The markers are evenly spaced in time and bunched at one
    end, and Fig. 8 shows the change-over ending slow and starting fast, so the bunched end is the start
    of single support. The speed assumes about 40 marker intervals in 1.0 s.
* **Fig. 8 (the paper's simulated change-over phase plane).** The CoM angle runs 0.23 → 0.56 rad, with
  angular velocity about 0.6–0.7 rad/s at the start and near zero at the end. Combined with Fig. 12, this
  puts the CoM about **0.75–0.80 m above the ankle axis** if the angle is atan(horizontal / height), or
  0.82–0.90 m if it is measured along the pendulum arm. Fig. 8 is a worked example made with the Table 2
  values, not the experiment.
* **Fig. 18 and the WL-10R paper (RoManSy '84, published 1985), Figs. 7 and 10.** The support leg is
  nearly straight while walking.
* **Takanishi et al., "Realization of Dynamic Biped Walking Stabilized with Trunk Motion Under Known
  External Force", Advanced Robotics 1989, pp. 299–310.** Mostly about the later WL-12R. Two points bear
  on this model:
  * Its introduction says the WL-10RD's 1984 dynamic walk took **1.3 s per step** with a 0.4 m step.
    The JRSJ paper says 1.5 s (1.0 s single support + 0.5 s change-over). These sources don't say which
    is right; 1.3 s may come from later or better runs. This model is slower than both, at about 1.8 s
    per step.
  * Its reference [1] is a separate English conference paper: Takanishi, Ishida, Yamazaki and Kato,
    "The Realization of Dynamic Walking by The Biped Walking Robot WL-10RD", ICAR '85, 1985. It has not
    been consulted here and may contain different figures or numbers.
* **The WL-10R paper, other facts.**
  * The predecessor weighed about 80 kg with its onboard computer.
  * Its frame is CFRP.
  * Each sole has four micro switches.
  * Its four-phase change-over used preset ankle accelerations, not impedance control.

## Estimates (not in any source)

* **Mass split.** Pelvis 17 kg, control box 29 kg and top items 9 kg, so 55 kg above the hips. Each leg
  has a 1 kg hip link, 2 + 1.5 kg thigh, 4 kg shank, 1 kg ankle link and 4.5 kg foot, so 14 kg per leg.
  This puts the CoM 0.76–0.77 m above the ankle axis with nearly straight legs, the lower end of what
  Figs. 8 and 12 imply.
* **Posture.**
  * Standing hip-axis height: 0.91 m.
  * Pelvis heights at the three pattern states: tuned between 0.80 and 0.88 m.
  * Knee bend while walking: median about 45°. The first version bent the knees about 63°.
  * Why not straighter: with a 40 cm stance and the rear toe still on the floor at the start of single
    support, straighter legs are out of reach in this model.
* **Servos.**
  * Position servo: 60000 N·m/rad and 600 N·m·s/rad.
  * Ankle position servo: 40000 and 800, chosen from a small sweep that minimized ZMP tracking error in
    single support.
  * Torque limits: ankles 100 N·m (the paper's range), hip and knee 300, hip yaw 100.
* **Contacts.** Spikes 15 mm in from the sole edges, radius 8 mm. Switch threshold 20 N with a 10 ms
  debounce.

## Deviations from the paper (needed to get a walk)

1. **Hip and knee joints in the change-over.** Each leg's hip and knee are set every millisecond by IK
   from the measured pelvis position, so the feet stay on their footprints. The paper says these joints
   do not move.
   * Rigidly locked legs were tried. With both feet flat, they leave only a rotation about the line
     between the ankles, so the CoM cannot travel from the rear foot to the front.
   * Fig. 18(b) also shows a level pelvis with the legs leaning, which needs hip motion.
2. **Hind ankle roll in A and B** is compliant instead of position-controlled.
3. **Swing foot in phase A** keeps its landing attitude fixed in world coordinates (heel first).
4. **End of phase D.** It also waits for the front ankle roll to reach the next pattern's start value and
   for all four front spikes to be loaded, with a 0.35 s time-out. **In the published run every D phase
   ends on the time-out**, not on the angle trigger.
5. **Pitch NP reference.** Measured from the support-ankle angle at the start of a steady step. Roll NP is
   mirrored per side: + means the hip is outboard of the foot.
6. **Sign of Eq. 45.** Used with the sign that converges.
7. **Stop.** The paper does not describe how its walks stop. Here the last step is a preset stop step
   that sets the swing foot down beside the support foot while the body is still moving toward the
   middle. A double-support settling phase then brings the body to rest. A version that stopped on one
   foot was much less reliable.
8. **Weight shift at the start.** Uses a minimum-jerk profile, which has zero acceleration at both ends,
   so the first step starts from a static state.

## Results (designs/tuned.json, `--steps 4`)

* **The walk completes.**
  * Stand, weight shift, then 5 single-support steps of about 0.41 m with 4 change-overs.
  * Stop step, then settle: 13.8 s in total.
* **Change-over vs Fig. 8.**
  * The CoM angle runs about 0.25 → 0.61 rad (paper 0.23 → 0.56).
  * The peak angular velocity is 0.74–0.77 rad/s (paper about 0.71).
  * It ends at 0.11–0.16 rad/s (paper about 0.05). The paper's sharp drop in speed at the B/C boundary
    is missing.
  * Phase durations: A about 0.10 s, B about 0.34 s, C 0.01 s, D 0.35 s (time-out). Total about
    0.8 s, against the paper's 0.5 s.
* **Single support vs Fig. 12.**
  * The CoM path has the paper's shape and end point. It enters single support about 2–3 cm further
    inboard than the design, because of lateral error at the hand-over.
  * **The support foot is rarely loaded on all four spikes:** 0–7 % of each steady step. Most of the time
    the load is on the outer edge. The paper aims to keep all four loaded.
  * The patterns are designed with about 2 cm of ZMP margin, and the designed ZMP agrees exactly with
    MuJoCo inverse dynamics.
  * In the dynamic simulation the measured ZMP still swings about ±4 cm (rms) around the design.
    Joint tracking errors are only 0.1–0.3°, but small errors at the support ankle rotate the whole body
    about it. Stiffer or better-damped ankle servos reduced this only a little.

### Fragility

This version is more faithful to the paper's figures than the first one, but less robust:

| test | result |
|---|---|
| 5 or more change-overs | falls in the 5th change-over |
| ±0.02 m/s initial sideways velocity | falls in the 3rd change-over |
| +1 kg on the upper body | falls in the 3rd change-over |
| −1 kg on the upper body | falls in the 3rd single-support step |

Searches that scored each design over several perturbed starts did not find a table that worked both
nominally and under perturbation. Load `tuned.json` at full precision: the first version broke when its
table was rounded to 0.01.

## Ideas for improving it

* Real link masses, or the actuator masses, would remove the biggest unknown.
* Get the ICAR '85 English paper on the WL-10RD (see "Sources used"). It may settle 1.3 s vs 1.5 s per
  step and could include data the JRSJ paper doesn't.
* Find out how the paper measures the CoM angle in Fig. 8, and whether Figs. 8 and 12 come from the same
  design.
* Reduce the lateral hand-over error so single support can start on a flat foot. That probably means
  scoring the measured ZMP during single support in the change-over search; one attempt did not
  converge in the time available.
* Better data on the electro-hydraulic ankle servos, or a softer sole model, which may change the rocking
  in single support.

## License

MIT (see `LICENSE`). This is an independent reconstruction from published papers. It is not affiliated
with Waseda University or the paper's authors, and it contains no text or images from the papers; only
data points read off Fig. 12 (`wl10rd_sim/paper_data/fig12_digitized.json`).
