# Migration guide: moving the ice-nucleation CV work to a local workstation

This work was developed in an ephemeral, sandboxed cloud container with no
GPU, no OpenMM, and restricted internet access (the outbound proxy blocked
`arxiv.org` and `ovito.org`, among others). Actually driving a live ice
nucleation simulation needs a validated OpenMM implementation of the mW
(Stillinger-Weber) potential -- an infrastructure task that benefits from
real compute, a GPU, and unrestricted internet access to cross-check the
physics against the literature. This document is a self-contained runbook
for continuing that work on a local workstation.

**Read this together with
[`docs/ice_nucleation_cv_protocol.md`](../../docs/ice_nucleation_cv_protocol.md)
(the full protocol) and
[`README.md`](README.md) (this directory's status summary) -- this file is
the *migration* runbook, those are the *scientific* references.**

## 1. What has been done vs. what is left

| Phase | Status | Where |
|---|---|---|
| Diagnose why IceCoder's VAE can't drive the transition | Done (written analysis) | `docs/ice_nucleation_cv_protocol.md` (Context section) |
| Phase 0: reference-state calibration (liquid / Ih / Ic separation) | **Done, 5/5 gate checks pass** | `examples/ice_nucleation/phase0_calibration.py` + `.json` results |
| Phase 1: fixed CV implementation (`[n_max, chi]`) | **Done, contract-tested** | `examples/ice_nucleation/project_ice.py`, `ice_descriptors.py` |
| Phase 1: live Trails-MD adaptive run on mW | **Not done** -- blocked on an mW-capable OpenMM system | `examples/ice_nucleation/config_target_mw.yaml` (ready, needs `system_mw.py`) |
| Phase 2: data-driven CV refinement (SPIB/Deep-TICA + VAMP-2) | Not started -- needs a code change in Trails-MD core | See protocol doc, "Phase 2" |
| Phase 3: committor validation, polymorph-purity cross-check | Not started | See protocol doc, "Phase 3" |
| Phase 4: all-atom (TIP4P/Ice) transfer | Not started | See protocol doc, "Phase 4" |

**The single concrete blocking item for real progress is building and
validating an OpenMM mW (Stillinger-Weber) system.** Everything else
(the CV itself, the calibration, the Trails-MD config) is ready and waiting
on it.

## 2. Clone the repositories

All work lives in three GitHub repos under `TeamSuman`. Clone all three into
sibling directories (mirroring this sandbox's layout, `~/IceCoder`,
`~/PathGennie`, `~/Trails-MD`) since `project_ice.py` and the protocol
reference IceCoder's data by relative-sibling assumptions in places -- adjust
paths in `config_target_mw.yaml` if you use a different layout.

```bash
mkdir -p ~/ice-nucleation-work && cd ~/ice-nucleation-work

git clone https://github.com/TeamSuman/IceCoder.git
git clone https://github.com/TeamSuman/PathGennie.git
git clone https://github.com/TeamSuman/Trails-MD.git

cd Trails-MD && git checkout ice-nucleation && cd ..
cd IceCoder && git checkout claude/ice-nucleation-cv-optimization-pi3kcz && cd ..
cd PathGennie && git checkout claude/ice-nucleation-cv-optimization-pi3kcz && cd ..
```

Notes:
- **Trails-MD** (`ice-nucleation` branch) is where all the new work lives:
  the protocol doc and `examples/ice_nucleation/`. This branch has an open
  PR: https://github.com/TeamSuman/Trails-MD/pull/6 -- pull latest from
  there for anything done after this document was written.
- **IceCoder** and **PathGennie** on `claude/ice-nucleation-cv-optimization-pi3kcz`
  have *no new commits* relative to their `main` branches -- that branch
  name is just where the session started. You can equally well use `main`
  for those two repos. IceCoder's `main` is what's actually referenced
  (its shipped mW data and trained SVM).
- If `git clone` over HTTPS prompts for credentials and these are private
  repos, use an SSH remote or a GitHub personal access token instead.

## 3. Set up the environment

You need Trails-MD's base environment plus the ice-nucleation additions.

```bash
cd ~/ice-nucleation-work/Trails-MD

# Base Trails-MD environment (numpy, MDAnalysis, pydantic, OpenMM, pytorch,
# deeptime, etc. -- see env.yml for the full list)
conda env create -n trails-md -f env.yml
conda activate trails-md

# Ice-nucleation additions (freud, pyscal3, ase, dscribe)
conda env update -n trails-md -f examples/ice_nucleation/env-ice.yml

# Alternative / supplement: exact pip-pinned versions validated in the
# development sandbox, if you prefer pip or hit a conda solver conflict
pip install -r examples/ice_nucleation/requirements-ice.txt
```

If you don't need Trails-MD's full base environment yet (e.g. you only want
to rerun the Phase 0 calibration, which needs no OpenMM/torch/deeptime), a
minimal environment is enough:

```bash
pip install freud-analysis==3.4.0 pyscal3==3.3.2 ase==3.29.0 dscribe==2.1.2 \
            MDAnalysis==2.10.0 numpy==2.4.6 scipy==1.17.1 \
            scikit-learn==1.9.0 joblib==1.5.3 matplotlib==3.11.0 \
            pydantic==2.13.4 PyYAML==6.0.1
```

## 4. Verify the environment reproduces the sandbox results exactly

```bash
cd ~/ice-nucleation-work/Trails-MD/examples/ice_nucleation
python3 phase0_calibration.py
```

Expected output: all 5 gate checks `[PASS]`, ending in `OVERALL: PASS - go`,
with these specific numbers (from the development sandbox run, checked into
`phase0_calibration_results.json` for comparison):

- Liquid reference: `n_max <= 3`, `chi = NaN`, for all 3 random seeds.
- Ih reference (real trajectory): `n_max` grows from 267 to ~994-998 (out of
  1024), `chi == 0.0` at every sampled frame, empirical NN distance
  `2.7625 Angstrom`.
- Ic reference (synthetic): `n_max = 1728` (full crystal), `chi == 1.0`.

If your numbers differ, that's a real signal something in the environment or
data differs -- diff against `phase0_calibration_results.json` before
proceeding, since the OpenMM mW work below builds on this CV being correct.

Also re-run the `extract_cvs` contract test:

```bash
python3 -c "
import project_ice
cvs = project_ice.extract_cvs(
    trajectories=['../../../IceCoder/data/simulation_mW_long_scaled.dcd'],
    top_file='../../../IceCoder/data/seeded_mW_scaled.gro',
    conf_file='../../../IceCoder/data/seeded_mW_scaled.gro',
)
print(cvs.shape, cvs.min(axis=0), cvs.max(axis=0))
"
# Expect: (250, 2), n_max range ~[267, ~1000], chi range ~[0, ~0.005]
```

Adjust the relative IceCoder path above if you cloned to a different layout
than Section 2's recommended sibling-directory structure.

## 5. File manifest (everything relevant to this work)

**Trails-MD** (`ice-nucleation` branch):
```
docs/ice_nucleation_cv_protocol.md      # the full written protocol
examples/ice_nucleation/
  README.md                             # status summary (read this first)
  MIGRATION.md                          # this file
  ice_descriptors.py                    # CHILL+ + n_max clustering (core physics)
  project_ice.py                        # Trails-MD extract_cvs plug-in
  phase0_calibration.py                 # go/no-go gate script
  phase0_calibration_results.json       # reference output for comparison
  config_target_mw.yaml                 # Phase 1 Trails-MD config (needs system_mw.py)
  env-ice.yml                           # conda additions
  requirements-ice.txt                  # exact pip pins
```

**IceCoder** (`main`, unmodified -- referenced, not changed):
```
data/seeded_mW_scaled.gro               # mW seeded-Ih crystal, used as the Ih reference
data/simulation_mW_long_scaled.dcd      # 250-frame growth trajectory
NoteBooks/Saved/{pyvenc.pt,scaler.save,svm.save}   # trained VAE+SVM, for Phase 3 validation
Scripts/{model.py,icecoder.py,soaper.py}           # for the Phase 3 polymorph-purity cross-check
```

**PathGennie** (`main`/`devel`, unmodified -- referenced only as the Phase 4
biased-sampling escalation path if unbiased adaptive sampling can't cross the
all-atom barrier; not needed for Phase 0-2).

## 6. Next steps, in order, with technical detail

### 6.1 Build and validate an OpenMM mW (Stillinger-Weber) system -- the blocking item

No maintained OpenMM mW package exists (checked; the closest public reference
is a years-old, never-packaged discussion in
[openmm/openmm#2637](https://github.com/openmm/openmm/issues/2637) sketching
a `CustomManyParticleForce` approach). This needs to be built and validated
from scratch:

1. **Source the exact mW parameters** from the original paper: Molinero &
   Moore, *J. Phys. Chem. B* 2009, 113, 4008 ("Water Modeled As an
   Intermediate Element Between Carbon and Silicon") -- freely available on
   arXiv (blocked in the sandbox, reachable locally):
   `https://arxiv.org/pdf/0809.2811`. Get the two-body (modified
   Stillinger-Weber pair term: A, B, sigma, a, epsilon) and three-body
   (lambda, gamma, cos(theta0)) parameters precisely from there, not from
   memory or a paraphrase.
2. **Implement** via OpenMM's `CustomManyParticleForce` with
   `permutationMode=UniqueCentralParticle` for the 3-body angular term, plus
   a standard `CustomNonbondedForce` (or the 2-body piece folded into the
   same many-particle force) for the pairwise term. Use the expression form
   referenced in the GitHub issue above as a starting point, but verify every
   term against the paper's equations directly.
3. **Validate before trusting it for anything else** -- this is the step
   that must not be skipped:
   - Run NPT equilibration of bulk liquid mW at 1 bar across a temperature
     range and confirm the density curve matches published mW values
     (density maximum near 250 K, anomalous expansion on cooling below that
     -- this is mW's signature qualitative behavior and a strong sanity
     check).
   - Confirm the liquid-ice coexistence / melting point is close to the
     published ~274.6 K at 1 bar (Molinero & Moore report mW's melting point
     within ~1 K of real water's 273.15 K by construction).
   - Compute the O-O radial distribution function g(r) for equilibrated
     liquid mW and compare its first-peak position/height against published
     mW g(r) curves.
   - Run `examples/ice_nucleation/ice_descriptors.py`'s `ice_cv()` on a
     short equilibrated-liquid trajectory from this new system and confirm
     it reports `n_max` near the noise floor and `chi = NaN` (mirroring the
     Phase 0 randomized-liquid control, now on genuine liquid dynamics
     instead of a synthetic random configuration) -- and equally, run it on
     a bulk mW Ih crystal built at your system's equilibrium density and
     confirm `chi == 0.0` there too, as a second independent check beyond
     the synthetic-lattice test already done.
4. Wire the validated system into a `system_file` (e.g.
   `examples/ice_nucleation/system_mw.py`, mirroring
   `examples/AIB9/system.py`'s convention of a function OpenMM's engine
   backend calls to build the `System` object) and point
   `config_target_mw.yaml`'s `system.system_file` at it.

### 6.2 Launch the Phase 1 live Trails-MD run

Once 6.1 is done:

```bash
cd ~/ice-nucleation-work/Trails-MD
trails-md --config examples/ice_nucleation/config_target_mw.yaml --check   # dry-run validation
trails-md --config examples/ice_nucleation/config_target_mw.yaml           # real run
```

Watch `n_max` and `chi` climb toward the `target: [900, 0.0]` in
`config_target_mw.yaml` across iterations; use `trails-md-path` afterward to
reconstruct the connected liquid-to-ice lineage.

### 6.3 Phase 2: the `extract_ice_descriptors` hook

Described in full in the protocol doc's Phase 2 section. Concretely:

- Add `extract_ice_descriptors(self, trajectories)` to `FeatureExtractor` in
  `trails_md/spaces/features.py`, returning per-frame pooled moments of the
  Steinhardt/CHILL+ quantities already computed by
  `examples/ice_nucleation/ice_descriptors.py` (reuse that module rather than
  reimplementing).
- Extend `_extract_feature_type` in `trails_md/core.py` (search for that
  function name) to dispatch a new `"ice_descriptors"` feature type string.
- Add `"ice_descriptors"` as a valid `adaptive_feature_type` /
  `candidate_feature_types` entry in `trails_md/config.py`.
- Then a `space_mode: spib` (or `deep-tica`/`vampnet`) config becomes usable
  for the learned-CV refinement, mirroring
  `examples/AIB9/config_spib.yaml`.

### 6.4 Phase 3 / 4

Follow the protocol doc directly -- committor shooting from barrier
configurations, CHILL+ + IceCoder-SVM cross-validation of the final nucleus,
then re-run Phase 0's calibration procedure (this codebase, unmodified) on a
TIP4P/Ice all-atom system to transfer.

## 7. Things that were constrained by the sandbox and may resolve themselves locally

- **CHILL+ thresholds were sourced from search-engine result snippets, not
  the primary sources**, because the sandbox's outbound proxy returned
  `403` on direct `CONNECT` to `arxiv.org` and `ovito.org` (confirmed via
  `curl`, not a bot-detection issue -- a network policy block). The values
  used (`c3 < -0.8` staggered, `-0.35 <= c3 <= 0.25` eclipsed, hexagonal = 3
  staggered + 1 eclipsed, cubic = 4 staggered) are corroborated by multiple
  independent search results and validated empirically in
  `phase0_calibration.py` (perfect separation on real Ih and synthetic Ic
  references) -- but with full internet access, it is worth directly
  reading Nguyen & Molinero, *J. Phys. Chem. B* 2015, 119, 9369 (DOI
  `10.1021/jp510289t`) and/or installing OVITO locally (its "Chill+"
  modifier is a maintained reference implementation) to cross-check the
  *complete* decision tree, including the "interfacial ice"/"interfacial
  water" sub-classification that this implementation deliberately
  simplified (see the docstring in `ice_descriptors.py`).
- **No GPU was available**, so nothing here was performance-tuned or run at
  production scale (the calibration script runs on a 1024/1728-molecule
  system in about a second on CPU). A local GPU workstation changes the
  economics of the mW validation runs in 6.1 (which need NPT equilibration
  runs of meaningful length) and any subsequent adaptive-sampling swarm
  sizes.
- **OpenMM, PyTorch, and deeptime were never installed** in the sandbox
  (only what Phase 0/1 needed). Trails-MD's `env.yml` already lists them;
  installing the full base environment (Section 3) pulls them in.
