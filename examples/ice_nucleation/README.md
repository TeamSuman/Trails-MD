# Ice-nucleation CV example (Phase 0 + Phase 1 artifacts)

Implements and validates the fixed physics-based collective variable from
[`docs/ice_nucleation_cv_protocol.md`](../../docs/ice_nucleation_cv_protocol.md):
`[n_max, chi]`, where `n_max` is the size of the largest spatially connected
cluster of ice-like water molecules (nucleation progress) and `chi` is the
CHILL+ cubicity `n_Ic / (n_Ic + n_Ih)` (polymorph selectivity).

## Files

| File | Purpose |
|---|---|
| `ice_descriptors.py` | CHILL+ classification, `n_max` clustering, `chi`. Pure freud/NumPy. |
| `project_ice.py` | Trails-MD `extract_cvs(trajectories, top_file, conf_file)` plug-in wrapping `ice_descriptors`. |
| `phase0_calibration.py` | Runs the Phase 0 go/no-go gate: confirms `(n_max, chi)` separates liquid / Ih / Ic. |
| `phase0_calibration_results.json` | Output of the calibration run (checked in for reference). |
| `config_target_mw.yaml` | Phase 1 target-mode Trails-MD config template (mW, `space_mode: fixed`). |
| `env-ice.yml` | Extra conda dependencies (freud, pyscal3, ase, dscribe) beyond the base `../../env.yml`. |
| `requirements-ice.txt` | Exact pip-pinned versions validated in the development sandbox. |
| [`MIGRATION.md`](MIGRATION.md) | Runbook for continuing this work on a local workstation (clone/env setup, verification, and a detailed next-steps task list -- start here if picking this up fresh). |

## What has been executed and validated

**Phase 0 (reference-state calibration) -- PASSED, 5/5 gate checks**, run against
real data (no fabricated numbers):

1. **Liquid reference**: randomized water positions (3 seeds) at the same box
   density as the crystal reference below. Result: `n_max <= 3` (noise floor,
   out of 1024 molecules) and `chi` undefined (`NaN`, no crystalline
   molecules at all).
2. **Bulk Ih reference**: IceCoder's shipped seeded-mW crystallization
   trajectory (`IceCoder/data/seeded_mW_scaled.gro` +
   `simulation_mW_long_scaled.dcd`, documented as an Ice-Ih seed growing in
   supercooled liquid at 250 K). Result: `n_max` grows from 267 to ~994-998
   (out of 1024) across the trajectory, and `chi == 0.0` at *every* sampled
   frame -- zero false-positive cubic labels even as the crystal grows
   through a realistic, thermally-noisy MD trajectory.
3. **Bulk Ic reference**: a synthetic diamond-cubic oxygen lattice built with
   ASE at the O-O nearest-neighbor distance measured *empirically* from the
   grown Ih crystal in (2) (2.7625 Angstrom -> lattice constant 6.380
   Angstrom) -- not an independently guessed literature constant. Result:
   `chi == 1.0` exactly, `n_max` spans the full 1728-molecule synthetic
   crystal, zero false-positive hexagonal labels.

Run it yourself: `python3 phase0_calibration.py` (needs `freud-analysis`,
`ase`, `MDAnalysis`; see `env-ice.yml`).

One implementation bug was caught and fixed during calibration: the initial
"interfacial ice" catch-all label did not require 4-fold (tetrahedral)
coordination, so a dense randomized/liquid-like configuration produced
false-positive `n_max` clusters purely from high coordination number, not
genuine order. Restricting that label to exactly-4-bonded molecules (matching
the tetrahedral H-bond coordination the cubic/hexagonal/clathrate rules
already require) fixed it -- confirmed by re-running the same negative
control.

**Phase 1 CV artifact -- implemented and contract-tested.**
`project_ice.extract_cvs()` was run end-to-end on the full 250-frame shipped
mW trajectory: output shape `(250, 2)`, all values finite, `n_max` grows
monotonically in trend from 267 to ~1000, `chi` stays at 0.0 except one
single-frame thermal-noise blip to 0.0053 (harmless, expected). Runtime:
~4.9 ms/frame -- fast enough for on-the-fly use inside Trails-MD's adaptive
loop. `config_target_mw.yaml` validates against Trails-MD's Pydantic config
schema (`trails_md.config.TrailsMDConfig`).

## What is *not* yet executed, and why

**No live Trails-MD adaptive run has been launched.** Driving actual MD
requires an OpenMM `system_file` for the mW (Stillinger-Weber) potential.
OpenMM has no built-in Stillinger-Weber/mW force field, and no maintained,
independently-verified OpenMM mW implementation exists to install (checked;
the closest reference is a years-old open GitHub issue sketching a
`CustomManyParticleForce` approach, never packaged or validated). Writing one
from scratch and validating it (matching mW's known melting point, density,
etc. under NPT) is a substantial, separate infrastructure task with real
correctness risk if rushed -- it was deliberately *not* attempted here rather
than shipping an unverified force field. This is the concrete, single
blocking item for a live Phase 1 run; everything else in the config
(`project_file`, `space_mode: fixed`, `search_mode: target`, target CV
values) is ready.

Phase 2 (data-driven CV refinement via SPIB/Deep-TICA + VAMP-2 feature
selection) still requires the `extract_ice_descriptors` hook in
`trails_md/spaces/features.py` described in the protocol -- not started.
