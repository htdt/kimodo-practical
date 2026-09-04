# Guard / transfer-modifier ablation evidence

Recorded measurements behind every keep-or-delete decision on the
retargeter's guards and transfer modifiers. Numbers were produced by `ablate.mjs` on the regenerated MK move
set (17 clips) and the deterministic e2e constraint suite, on both certified
reference rigs, BEFORE the removals — the deleted code paths live in git
history at the commit tagged by these files.

Reproduction:

```bash
# generation (model Kimodo-SOMA-RP-v1.1, seed 42, 8 samples, 100 steps)
python kimodo/kimogen.py gen --spec kimodo/moveset_mk.json  --samples 8 --seed 42
python kimodo/kimogen.py gen --spec kimodo/moveset_e2e.json --samples 8 --seed 42
python kimodo/bake_kimodo.py --spec kimodo/moveset_mk.json  --web kimodo/out/web_mk
python kimodo/bake_kimodo.py --spec kimodo/moveset_e2e.json --web kimodo/out/web_e2e
# measurement (pre-removal tree)
node ablate.mjs <fighter.glb> kimodo/out/web_mk --json ablation_fighter.json
node ablate.mjs <char2.glb>   kimodo/out/web_mk --json ablation_char2.json
node ablate.mjs <fighter.glb> kimodo/out/web_e2e --ik --json ablation_fighter_e2e.json
node ablate.mjs <char2.glb>   kimodo/out/web_e2e --ik --json ablation_char2_e2e.json
```

## Decisions (see KIMODO.md for the narrative)

| intervention | evidence (max over clips, both rigs) | decision |
|---|---|---|
| `continuity` 40°/frame slew | Δ vs baseline **0°** on all 34 clip-runs — never engages; baseline branch flips: **0**; history-dependent by construction | **deleted** |
| `ground` root lift | Δ **0°** — never engages (default-off, temporal settling is history-dependent) | **deleted** |
| `handClamp` 85°/70° | Δ up to **15.5°** (char2) / 2.3° (fighter) on VALID Kimodo poses — clips authored wrists; baseline shows 0 flips and twist within anatomical limits without it | **deleted** |
| `torsoCapsule` | Δ up to **6.7°/7.2 mm**; baseline capsule clearance min **1.33–1.55 × R** — no torso penetration exists on the representative suite; displaces valid near-face guard poses; could silently move a constrained hand | **deleted** (clearance stays a measured QA metric) |
| `foreRollSrc` | disabling it costs **179°** forearm roll error (chest-rebase projection) | **promoted to the only quaternion path** (auto-on whenever the clip has quats; flag removed) |
| `handFollow` (baked 0.3) | discards **22.8° mean / 60.8° p95** of real wrist articulation — an intentional stylization, now measured separately from raw fidelity | **retained** as a per-clip style option; forced to 1.0 for clips with authored hand constraints; QA reports raw transfer and stylization delta separately |
| constraint IK (new) | direction-only retarget misses REACHABLE mapped authored hand targets by **8.3–15.3 cm / 57–92°** across the two rigs (qa_e2e_*_noik.json); with IK the same targets hit exactly (0.0000 m / 0.00°); deterministic under seek/stride/loop-wrap | **added**, constrained clips only |
| `root_margin` (upstream post-processing) | the upstream default 0.04 m leaves corrected roots exactly 4 cm off authored waypoints; 0.0 is exact (measured) | kimogen defaults to **0.01 m** (half the 2 cm adherence gate) |
| `foot_excursion` apex gate (kimogen) | the reference `sweep` (gated on `root_dip` ≥ 0.2 m) shipped a crouch: max horizontal foot travel **2.5 cm**, strike-tip peak 0.67 m/s — the best-of-N score (skate + jitter + stance) prefers the smoothest sample, and a sample that never sweeps is the smoothest. Regenerated under `foot_excursion ≥ 0.5 m` (seed 42, 8 samples): the same prompt yields real sweeps in **2/8** samples (foot travel **1.50 / 1.28 m**, toe peak 11.7 m/s) and the six crouch-only samples (0.005–0.49 m) are rejected; a reworded prompt also gives 2/8 (1.48 / 0.65 m). Skate 0.024, jitter 0.007, stance error 0.002 on the winner | **added** as an apex kind; the MK spec's sweep now gates on it (`min: 0.5`). The local `out/web_mk` sweep predates this and is a crouch |
| planted-foot hold (constraint IK) | a ±6-frame blend toward a foot key on a PLANTED foot drags the foot by the rig's proportion mismatch and back: on the certified Tripo skater rig (scaleRoot 0.871) the retargeted foot sits **5.35 cm** from the mapped `e2e_lfoot` target and the toe moved at up to **0.34 m/s** through the blend while the source foot was still (new contact-frame skate **0.033 / 0.031 m/s** on `e2e_lfoot` / `e2e_hands_feet` vs the 0.03 gate; the reference fighter showed the same shape at 2 cm / 0.07 m/s, under the gate). Holding the key across the predicted contact run and blending only in the air: shipped contact-frame skate falls **below** the unconstrained baseline (Δ −0.036 / −0.015 m/s on `e2e_feet` / `e2e_hands_feet`, 0 on `e2e_lfoot`), keys still land at 0.0000 m, seek == sequential, hands untouched | **added** (ik.js `footContactRuns`; selftest_constraints covers hold span, zero motion in the run, determinism). Reference-rig JSONs in this directory predate the hold; re-record them with the skater numbers when those rigs are available |

Baseline (all guards off, handFollow 1, source forearm roll): 0 branch flips,
round-trip mean 12.8 mm, no NaN/denormalized quaternions, no history
dependence, on both rigs across the full suite.
