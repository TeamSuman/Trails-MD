# Protocol: An Optimal Collective Variable for Polymorph-Selective (Ih vs Ic) Ice Nucleation via Unbiased Adaptive Sampling (Trails-MD)

> Scope: a written, executable protocol for designing and optimizing a collective
> variable (CV) that drives the liquid → ice transition **and** selects the
> polymorph (hexagonal Ih vs cubic Ic), developed on coarse-grained **mW** water
> and transferred to an **all-atom** model. The transition is driven by
> **unbiased adaptive sampling in Trails-MD** (target-mode restart selection +
> supercooling), not by a biasing force. This document is a plan; no CV code is
> committed yet.

## Context

**Why this is needed — the published IceCoder VAE cannot drive the transition.**
The IceCoder VAE latent (`IceCoder/Scripts/model.py`, `IceCoder/Scripts/icecoder.py`) is:

1. **Reconstruction-only.** Trained on SOAP descriptors with pure MSE + KL loss
   (β=1, `model.py:110-112`). It carries **no dynamical, time-lagged, or committor
   information**, so latent proximity does not imply dynamical proximity — the
   coordinate is not aligned with the slow nucleation reaction coordinate.
2. **Per-molecule, not collective.** SOAP is centered on each water
   (`Scripts/soaper.py:12-18`, `average="off"`); the VAE + SVM label *individual*
   molecules. Nucleation's rate-limiting slow variable is a **global** quantity —
   the size of the largest solid-like cluster (classical nucleation theory) —
   which a per-molecule autoencoder does not represent.
3. **Not biasable / not driveable.** Exported only as a PyTorch `state_dict` +
   pickled sklearn scaler/SVM; **no gradients, no differentiable / PLUMED /
   TorchScript export**, and `project()` detaches to NumPy. Its own transition
   demo (`NoteBooks/mW_transition.ipynb`) is a **seeded, unbiased** run analyzed
   post-hoc — never CV-driven.

**Intended outcome.** Replace the VAE-as-driver with a physically grounded,
collective, polymorph-resolving CV that plugs directly into Trails-MD's
`extract_cvs` interface for immediate target-mode driving, then upgrade it to a
*data-driven, VAMP-2-optimized* CV once a feature-extraction gap in Trails-MD is
filled. IceCoder is retained as a **validator/labeler**, not a driver.

---

## Design principle: physics-first CV, data-driven refinement

The optimal CV is a small, interpretable vector combining a *nucleation-progress*
coordinate with a *polymorph-selectivity* coordinate, later refined into a single
optimized slow coordinate.

**Component 1 — Nucleation progress (drives liquid → solid).**
- Per-molecule solid/liquid classification via **Lechner–Dellago averaged
  Steinhardt** parameters (q̄4, q̄6, w̄4, w̄6) and the ten Wolde–Frenkel
  bond-connectivity criterion (count of "crystalline" bonds from the normalized
  q6(i)·q6(j)* dot product).
- Cluster the solid-like molecules and take **n_max = size of the largest
  solid-like cluster** as the primary progress variable — the CNT-consistent slow
  coordinate the VAE lacks.

**Component 2 — Polymorph selectivity (Ih vs Ic).**
- **CHILL+** (Nguyen–Molinero) per-molecule classification into cubic /
  hexagonal / interfacial ice from staggered vs eclipsed H-bond counts — the
  standard, cheap Ih/Ic discriminator, valid for both mW and all-atom.
- Define **cubicity χ = n_Ic / (n_Ic + n_Ih)** over solid-like molecules (and
  optionally largest-cluster-restricted χ). Target-mode driving toward χ→0
  selects Ih; χ→1 selects Ic.
- Optional sharper discriminator: **environment-similarity** kernels (SOAP
  power-spectrum overlap to reference Ih and Ic environments), reusing IceCoder's
  SOAP featurizer (`Scripts/soaper.py`) and, for validation, its trained SVM
  (`Saved/svm.save`).

**Component 3 (refinement) — Optimized slow CV.**
- Feed the *physical descriptors* (Steinhardt vector + CHILL+ counts + SOAP
  similarities, pooled to a system-level feature vector) into Trails-MD's
  **SPIB / Deep-TICA / VAMPNet** learners with **VAMP-2 greedy feature
  selection** and **`vamp_adaptive` retraining**. This produces a single, smooth,
  dynamically-optimized coordinate that maximizes kinetic variance and
  self-discovers the metastable (liquid / Ih / Ic) states — the property the VAE
  was missing.

The fixed physical CV is **immediately driveable** and works from iteration 0;
the learned CV is **optimal in the slow-mode sense** but needs data, so it
bootstraps from the physical CV (mirroring PathGennie's SPIB
"bootstrap-from-geometric-CV then retrain" pattern,
`PathGennie devel: docs/data-driven-cv.md`).

---

## Tooling / dependencies

- **freud** — `freud.order.Steinhardt` (averaged ql/wl), `freud.locality`
  neighbor lists, `freud.cluster.Cluster` for n_max. Fast, PBC-aware.
- **pyscal3** — CHILL+, Steinhardt, averaged, disorder parameters (CHILL+ Ih/Ic
  out of the box). Alternative: OVITO PTM/CHILL+.
- **dscribe** SOAP (already in IceCoder's `environment.yml`) for
  environment-similarity and SVM validation.
- **MDAnalysis** (already a Trails-MD dependency) for trajectory reading inside
  `extract_cvs`.
- Add these to a dedicated `env-ice.yml`; do not perturb Trails-MD's base env.

---

## Protocol phases

### Phase 0 — Reference states & descriptor calibration (mW)
- Collect equilibrated references: supercooled mW **liquid**, bulk **Ih**, bulk
  **Ic**. Reuse IceCoder's shipped mW system (`IceCoder/data/seeded_mW_scaled.gro`,
  `simulation_mW_long_scaled.dcd`; setup in `NoteBooks/mW_transition.ipynb`).
- Calibrate the q6-bond threshold and CHILL+ cutoffs; confirm **(n_max, χ)**
  cleanly separates liquid / Ih / Ic (three separated clouds). **Go/no-go gate.**
- Fix a deep-supercooling state point (mW nucleates spontaneously at strong
  supercooling — this is what makes *unbiased* adaptive sampling viable).

### Phase 1 — Physical fixed CV, target-mode driving (mW)
- Implement `extract_cvs(trajectories, top_file, conf_file) -> (n_frames, n_cvs)`
  returning `[n_max, χ, (optional) Ih_similarity, Ic_similarity]` — the exact
  Trails-MD plug-in contract (`examples/AIB9/project_vae_latent.py:81-97`,
  `examples/AlaD/project_phi_psi.py:8-37`).
- Configure `space_mode: fixed`, `system.project_file: project_ice.py`,
  `spawning.search_mode: target`, `spawning.target: [<large n_max>, <χ target>]`,
  `spawn_scheme: density` (or `we`). Target-mode restart selection preferentially
  re-launches walkers that progressed toward the crystalline basin
  (`trails_md/spawners/density.py:66-70,81-82`), building a liquid→ice pathway
  without a bias force.
- Reference config to clone: `examples/AIB9/config_target.yaml`.

### Phase 2 — Data-driven CV optimization (mW) — **requires closing a Trails-MD gap**
- **Gap:** learned CV modes get input features only from `_extract_feature_type`
  (`trails_md/core.py:925-933`), which dispatches to three protein-centric
  extractors (`pairwise_distances`, `fitted_coords`, `aib9_phi_psi`) in
  `trails_md/spaces/features.py`. There is **no hook for crystallinity features**,
  so SPIB/Deep-TICA cannot currently learn an ice CV from good inputs.
- **Fill it:** add `extract_ice_descriptors` to `FeatureExtractor` returning
  per-frame system-level pooled features (moments/histograms of averaged-Steinhardt
  q̄4/q̄6/w̄4/w̄6, CHILL+ population fractions, SOAP-similarity moments); register a
  new `adaptive_feature_type: "ice_descriptors"` in the dispatch (`core.py:925`)
  and as a `candidate_feature_types` option (`config.py:233`).
- Run `space_mode: spib` (or `deep-tica` / `vampnet`) with
  `feature_selection.enabled: true` (greedy VAMP-2,
  `trails_md/spaces/feature_selection.py:104`) and
  `retrain_policy: vamp_adaptive` (`trails_md/spaces/retraining.py:73-93`). The
  learned latent becomes the optimized, polymorph-resolving driving CV; the
  physical CV stays as the bootstrap/target reference (Trails-MD projects physical
  CVs alongside adaptive ones, `core.py:540-543`).
- Reference configs: `examples/AIB9/config_spib.yaml`, `config_deep_tica.yaml`,
  `config_msm_vampnet.yaml`.

### Phase 3 — Validation & CV quality metrics
- **Committor test** (the decisive check the VAE never passed): from putative
  barrier configurations (mid-CV), fire N short unbiased shots and confirm
  p_B ≈ 0.5 there with monotonic p_B vs CV — a genuine reaction coordinate, not
  just a classifier.
- **VAMP-2 / implied timescales:** confirm the learned CV's VAMP-2 score and slow
  timescales exceed the fixed CV's (scored natively, `core.py:_cv_vamp_score`).
- **Polymorph purity:** classify the final nucleus with CHILL+ *and* independently
  with IceCoder's SVM (`IceCoder/Scripts/icecoder.py` + `Saved/svm.save`) to
  confirm the driven polymorph matches the target χ.
- **Mechanism:** reconstruct the connected pathway with `trails-md-path`
  (lineage); optionally seed an MSM (`msm.enabled: true`) for kinetics.

### Phase 4 — Transfer to all-atom
- Re-run Phase 0 calibration for the all-atom model (recommend **TIP4P/Ice**;
  recalibrate CHILL+ H-bond geometry and q6 thresholds — descriptors transfer,
  thresholds do not).
- Keep the CV architecture. Expect a **higher barrier**; mitigate within the
  unbiased-adaptive paradigm via (i) deeper supercooling, (ii) larger swarm +
  longer τ, (iii) a staged/curriculum target on n_max, and (iv) a
  **seeded-nucleus fallback** (start walkers from small pre-formed clusters à la
  `mW_transition.ipynb`) if barrier crossing stalls. If unbiased adaptive cannot
  cross the all-atom barrier in budget, the same physical CV is directly reusable
  for **biased sampling** — PathGennie's `devel` branch already provides an
  OPES/PLUMED bridge (`pathgennie/sampling/opes.py`) and on-the-fly SPIB
  (`pathgennie/cv/spib.py`) as the escalation path.

---

## Implementation map (when the build starts)

**New (ice-specific, self-contained):**
- `project_ice.py` — the `extract_cvs(...)` physical CV (freud + pyscal3/CHILL+ +
  optional dscribe SOAP). Primary driver artifact.
- `ice_descriptors.py` — shared per-frame descriptor library used by both
  `project_ice.py` and the new Trails-MD extractor.
- `env-ice.yml` — freud + pyscal3 + dscribe added to the Trails-MD env.
- Example case dir mirroring `examples/AIB9/` with `config_target.yaml`,
  `config_spib.yaml`, and reference Ih/Ic structures.

**Modify in Trails-MD (only for Phase 2 learned CV):**
- `trails_md/spaces/features.py` — add `extract_ice_descriptors`.
- `trails_md/core.py` — extend `_extract_feature_type` (line 925) to dispatch
  `"ice_descriptors"`.
- `trails_md/config.py` — allow `adaptive_feature_type: "ice_descriptors"` and add
  it to `candidate_feature_types` (line 233).

**Reuse as-is (do not modify):**
- IceCoder `Scripts/soaper.py`, `Scripts/icecoder.py` + `Saved/svm.save`
  (validation labeler).
- Trails-MD `spaces/{registry,model,spib,feature_selection,retraining}.py`,
  `spawners/density.py`, target-mode machinery — consumed via config, no edits.

**Not used for driving:** IceCoder's VAE latent as a biasing CV; PathGennie (held
as the biased-sampling escalation path only).

---

## Verification (end-to-end)

1. **Descriptor separation (Phase 0 gate):** compute `(n_max, χ)` on the three mW
   reference ensembles; assert three separated clusters + correct CHILL+/SVM
   labels on bulk Ih and Ic. Run before any dynamics.
2. **`extract_cvs` contract:** run `project_ice.extract_cvs` on `IceCoder/data/*mW*`;
   assert shape `(n_frames, n_cvs)`, finite, monotone increase across the known
   seeded-growth trajectory.
3. **Trails-MD smoke run:** short `space_mode: fixed` target-mode mW run via
   `trails-md --config config_target.yaml --check`, then a short real run; confirm
   walkers advance in n_max and the lineage path connects liquid→ice.
4. **Learned-CV run (after gap fix):** short `space_mode: spib` run; confirm the
   new `ice_descriptors` features load, SPIB trains, VAMP-2 score logged, and
   emergent states map to liquid/Ih/Ic.
5. **Committor validation:** the Phase 3 shooting test — the acceptance criterion
   for calling the CV "optimal."
6. **Polymorph correctness:** CHILL+ and IceCoder-SVM agreement on the final
   nucleus polymorph vs target χ.

---

## Key risks & honest caveats
- **Unbiased adaptive may not cross the all-atom barrier** in a fixed budget; the
  seeded fallback and biased-sampling escalation (PathGennie OPES/PLUMED) are the
  documented mitigations. mW is expected to work well unbiased.
- **Threshold transferability:** Steinhardt/CHILL+ cutoffs must be recalibrated
  per water model (Phase 0/4), not copied.
- **Stacking-disordered ice:** χ (cubicity) treats stacking disorder as a
  continuum between Ih and Ic — a feature for polymorph control, but pure-phase
  targets need a tight χ target plus largest-cluster-restricted counting.
- **Trails-MD learned-CV features are protein-centric today;** Phase 2 is blocked
  until the `extract_ice_descriptors` hook is added — the single most important
  code change.
