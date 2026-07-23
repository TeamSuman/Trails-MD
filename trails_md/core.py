import importlib.util
import json
import logging
import shutil
import sys
import warnings
from pathlib import Path

# Suppress common non-critical warnings from dependencies
warnings.filterwarnings(
    "ignore", message="Non-optimal GB parameters detected for GB model HCT"
)
warnings.filterwarnings("ignore", message="Reload offsets from trajectory")
warnings.filterwarnings("ignore", message=".*Reader has no dt information.*")
from typing import Any

import numpy as np
from pydantic import ValidationError

from trails_md.checkpoints.manager import CheckpointManager
from trails_md.config import TrailsMDConfig
from trails_md.engines.amber import amber_trajectory_suffix
from trails_md.engines.base import EngineFactory
from trails_md.paths import build_frame_records, map_global_frame
from trails_md.reporting import IterationReporter
from trails_md.spaces import FeatureExtractor
from trails_md.spaces.registry import is_adaptive_space
from trails_md.spawners.base import SpawnerFactory
from trails_md.spawners.history import pooled_history_iterations, projection_dim
from trails_md.utils.seeds import SeedManager
from trails_md.workflows.parallel import run_iteration_parallel


class TrailsMDCore:
    """Main orchestrator for the Trails-MD framework."""

    def __init__(self, config_dict: dict[str, Any]):
        try:
            self.config = TrailsMDConfig(**config_dict)
        except ValidationError as e:
            logging.error(f"Configuration validation failed: {e}")
            raise

        self.outdir = Path(self.config.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.output_log = self.outdir / "output.log"
        self._ensure_output_log_header()

        # Initialize SeedManager
        self.seed_manager = SeedManager(self.config.random_seed)
        self.seed_manager.set_seed()

        # Initialize Checkpoint Manager
        self.checkpoint_manager = CheckpointManager(str(self.outdir / "checkpoints"))
        if self.config.checkpoint_freq == 0:
            logging.warning(
                "checkpoint_freq=0: checkpointing is DISABLED. A walltime kill or "
                "node failure will lose the whole run with nothing to resume from."
            )

        # Terminal progress reporter (presentation only).
        self.reporter = IterationReporter()

        # Initialize Engine
        _engine_kwargs = {
            k: v for k, v in self.config.engine.model_dump().items() if k != "md_engine"
        }
        self.engine = EngineFactory.get(self.config.engine.md_engine, **_engine_kwargs)

        is_adaptive = is_adaptive_space(self.config.space_mode)
        # Initialize Spawner
        self.spawner = SpawnerFactory.get(
            self.config.spawning.spawn_scheme,
            n_bins=self.config.n_bins,
            min_values=None if is_adaptive else self.config.min_values,
            max_values=None if is_adaptive else self.config.max_values,
            mode=self.config.spawning.search_mode,
            probabilistic=self.config.spawning.spawn_type != "hard",
            target=self.config.spawning.target,
            recent_window=self.config.spawning.recent_density_window,
            n_clusters=self.config.spawning.voronoi_clusters,
            periodic=self.config.spawning.voronoi_periodic,
            grid_size=self.config.spawning.voronoi_grid_size,
            n_neighbors=self.config.spawning.lof_neighbors,
            target_per_bin=self.config.spawning.we_target_per_bin,
            recycle_target=self.config.spawning.recycle_target,
            recycle_basis_index=self.config.spawning.recycle_basis_index,
            alpha=self.config.msm.spawn_alpha,
            leverage=self.config.msm.spawn_leverage,
            uncertainty=self.config.msm.spawn_uncertainty,
            seed=self.config.random_seed,
        )

        # Landscape-adaptive binning (opt-in via config.binning.scheme); supplied
        # to the density / WE spawners. `uniform` leaves the spawners on the grid.
        binning_cfg = getattr(self.config, "binning", None)
        if (
            binning_cfg is not None
            and binning_cfg.scheme != "uniform"
            and hasattr(self.spawner, "binner")
        ):
            from trails_md.binning.adaptive import make_binner

            self.spawner.binner = make_binner(
                binning_cfg.scheme,
                n_bins=self.config.n_bins,
                min_values=None if is_adaptive else self.config.min_values,
                max_values=None if is_adaptive else self.config.max_values,
                target=self.config.spawning.target
                if self.config.spawning.search_mode == "target"
                else None,
                n_fine=binning_cfg.n_fine,
                smoothing=binning_cfg.smoothing,
            )

        # State variables
        self.iteration = 0
        self.history = {}
        # Structure a recycled walker restarts from (source->sink kinetics mode).
        # Captured once, on the first spawn, and thereafter held fixed -- see
        # `_recycling_basis_state` for why a frame index cannot carry this.
        self._basis_state: dict[str, Any] | None = None
        self.occupancy_history = []
        self.bin_state = {}
        self.scaler = None
        self.space_model = None
        self.feature_memory = []
        self.adaptive_space_version = 0
        self.walker_parents = []
        self.last_occupied_bins = None
        self.convergence_stall_count = 0
        self.converged = False
        self.convergence_reason = None

        # VAMP-2 input-feature selection (opt-in via config.feature_selection).
        self.feature_selector = None
        self.feature_selection_indices = None
        self.last_feature_selection = None
        self.selected_feature_type = None
        fs_cfg = getattr(self.config, "feature_selection", None)
        if fs_cfg is not None and fs_cfg.enabled:
            from trails_md.spaces.feature_selection import FeatureSelector

            self.feature_selector = FeatureSelector(
                lagtime=fs_cfg.lagtime,
                method=fs_cfg.method,
                max_features=fs_cfg.max_features,
                dim=fs_cfg.dim,
                min_gain=fs_cfg.min_gain,
            )

        # Adaptive CV-retraining policy (VAMP-2 driven when configured).
        from trails_md.spaces.retraining import RetrainController

        self.retrain_controller = RetrainController(
            policy=self.config.retrain_policy,
            retrain_freq=self.config.retrain_freq,
            vamp_tol=self.config.vamp_retrain_tol,
            min_interval=self.config.retrain_min_interval,
            max_interval=self.config.retrain_max_interval,
        )

        # MSM subsystem (opt-in via config.msm.enabled).
        self.msm_estimator = None
        self.msm_monitor = None
        self.last_msm_result = None
        msm_cfg = getattr(self.config, "msm", None)
        if msm_cfg is not None and msm_cfg.enabled:
            from trails_md.msm import MSMEstimator, build_monitor_from_config

            self.msm_estimator = MSMEstimator(
                lagtime=msm_cfg.lagtime,
                n_microstates=msm_cfg.n_microstates,
                cluster_method=msm_cfg.cluster_method,
                estimator=msm_cfg.estimator,
                n_metastable=msm_cfg.n_metastable,
                n_timescales=msm_cfg.n_timescales,
                lagtimes=msm_cfg.lagtimes,
                n_bayesian_samples=msm_cfg.n_bayesian_samples,
                stable_clustering=getattr(msm_cfg, "stable_clustering", False),
                seed=self.config.random_seed,
            )
            self.msm_monitor = build_monitor_from_config(msm_cfg)

    def prepare(self):
        """Prepare the simulation system."""
        self.engine.prepare(
            conf=Path(self.config.system.conf_file),
            top=Path(self.config.system.top_file),
            system_file=Path(self.config.system.system_file)
            if self.config.system.system_file
            else None,
        )
        logging.info("System prepared successfully.")

    def validate_preflight(self) -> None:
        """Validate local inputs before launching any production walkers."""
        errors: list[str] = []

        self._require_file(errors, "system.conf_file", self.config.system.conf_file)
        self._require_file(errors, "system.top_file", self.config.system.top_file)
        self._require_optional_file(
            errors, "system.system_file", self.config.system.system_file
        )
        self._require_optional_file(
            errors,
            "system.trajectory_topology_file",
            self.config.system.trajectory_topology_file,
        )
        self._validate_project_file(errors)
        self._validate_engine_preflight(errors)
        self._validate_sampling_settings(errors)

        if errors:
            joined = "\n  - ".join(errors)
            raise RuntimeError(f"Preflight checks failed:\n  - {joined}")

    def _adaptive_model_kwargs(self) -> dict:
        kwargs = self.config.adaptive_model.model_dump()
        kwargs["space_mode"] = self.config.space_mode
        kwargs["seed"] = self.config.random_seed  # reproducible CV training
        return kwargs

    def restore_checkpoint(self, iteration: int, ignore_missing_history: bool = False):
        """Restore sampler state from a saved checkpoint and resume at the next iteration."""
        restored_model = self.space_model
        if restored_model is None and is_adaptive_space(self.config.space_mode):
            from trails_md.spaces import AdaptiveSpaceModel

            restored_model = AdaptiveSpaceModel(**self._adaptive_model_kwargs())

        (
            self.space_model,
            self.scaler,
            self.bin_state,
            self.history,
            sampler_state,
        ) = (
            self.checkpoint_manager.load(
                iteration=iteration,
                space_model=restored_model,
                ignore_missing_history=ignore_missing_history,
            )
        )
        if self.space_model is not None:
            if hasattr(self.space_model, "ensure_config_defaults"):
                self.space_model.ensure_config_defaults(
                    **self._adaptive_model_kwargs()
                )
            self.space_model.scaler = self.scaler
        self._restore_feature_memory_from_history()
        self._restore_walker_parents_from_history()
        self._restore_sampler_state(sampler_state)
        self.iteration = iteration + 1

    def latest_checkpoint_iteration(self) -> int:
        return self.checkpoint_manager.latest_iteration()

    def resume_walkers(self) -> list[Any]:
        """Rebuild next walkers from the latest restored checkpoint history."""
        if not self.history:
            return [self.engine.positions for _ in range(self.config.spawning.walker)]

        latest_iteration = max(self.history)
        latest_entry = self.history[latest_iteration]
        if not isinstance(latest_entry, dict):
            raise RuntimeError(
                f"Checkpoint history entry {latest_iteration} is not resumable."
            )

        spawn_indices = latest_entry.get("spawn_indices")
        if spawn_indices is None:
            raise RuntimeError(
                f"Checkpoint history entry {latest_iteration} has no spawn_indices."
            )

        # These spawn indices were computed at ``latest_iteration`` against the
        # pool of history frames matching that iteration's projection dimension.
        # Rebuild the trajectory list over the same iterations so the indices
        # still resolve to the intended frames after a resume.
        latest_projection = latest_entry.get("projection")
        target_dim = (
            projection_dim(latest_projection)
            if latest_projection is not None
            else None
        )
        trajectories = self._sampling_trajectories([], target_dim=target_dim)
        if not trajectories:
            raise RuntimeError(
                f"Checkpoint history entry {latest_iteration} has no trajectories."
            )
        self._validate_sampling_trajectories(trajectories, context="resume")

        trajectory_topology = (
            self.config.system.trajectory_topology_file or self.config.system.top_file
        )
        feature_extractor = FeatureExtractor(
            topology=trajectory_topology,
            selection=self.config.system.feature_selection,
        )
        return feature_extractor.extract_positions_by_indices(
            trajectories,
            list(spawn_indices),
        )


    def generate_initial_walkers(self) -> list[Any]:
        if not self.config.system.initial_trajectory:
            return [self.engine.positions for _ in range(self.config.spawning.walker)]

        import logging
        from pathlib import Path

        import MDAnalysis as mda
        import numpy as np

        from trails_md.spaces import FeatureExtractor

        traj_path = str(Path(self.config.system.initial_trajectory).resolve())
        logging.info(f"Initializing walkers from trajectory: {traj_path}")

        trajectory_topology = (
            self.config.system.trajectory_topology_file or self.config.system.top_file
        )

        u = mda.Universe(trajectory_topology, traj_path)
        n_frames = len(u.trajectory)
        n_walkers = self.config.spawning.walker

        if n_frames == 0:
            raise ValueError(f"Initial trajectory {traj_path} contains 0 frames.")

        points = None
        if hasattr(self, "_extract_physical_cvs") and self.config.system.project_file:
            try:
                points = self._extract_physical_cvs([traj_path])
            except Exception as e:
                logging.warning(f"Failed to extract CVs from initial trajectory, falling back to random sampling: {e}")

        if points is not None and len(points) > n_walkers:
            logging.info(f"Selecting {n_walkers} starting walkers from {n_frames} frames using spawning scheme...")
            try:
                # the spawner might require history for some things, but mostly just points.
                # Since history is empty, pass it empty.
                spawn_indices = self.spawner.sample(points, n_walkers, history={})
            except Exception as e:
                logging.warning(f"Spawning scheme failed on initial trajectory: {e}. Falling back to uniform.")
                spawn_indices = list(np.linspace(0, n_frames - 1, n_walkers, dtype=int))
        else:
            if n_frames >= n_walkers:
                logging.info(f"Using uniform sampling for {n_walkers} starting walkers from {n_frames} frames.")
                spawn_indices = list(np.linspace(0, n_frames - 1, n_walkers, dtype=int))
            else:
                logging.info(f"Only {n_frames} frames available for {n_walkers} walkers; replicating randomly.")
                spawn_indices = [
                    int(i)
                    for i in self.seed_manager.rng.integers(0, n_frames, size=n_walkers)
                ]

        feature_extractor = FeatureExtractor(
            topology=trajectory_topology,
            selection=self.config.system.feature_selection,
        )
        walkers = feature_extractor.extract_positions_by_indices(
            [traj_path], spawn_indices
        )

        if points is not None:
            from trails_md.core import build_frame_records
            frames = build_frame_records(
                iteration=-1,
                trajectories=[traj_path],
                points=np.asarray(points),
                walker_parents=["initial"],
                expected_frames=n_frames,
            )

            next_walker_parents = [frames[idx]["key"] for idx in spawn_indices]

            self.history[-1] = {
                "projection": points,
                "spawning_scheme": "initial",
                "trajectories": [traj_path],
                "spawn_indices": list(spawn_indices),
                "frames": frames,
                "walker_parents": ["initial"],
                "next_walker_parents": next_walker_parents,
            }
            self.walker_parents = next_walker_parents
            logging.info(f"Injected {n_frames} frames from initial trajectory into permanent history (iteration -1).")

        return walkers

    def _traj_suffix(self) -> str:
        if self.config.engine.md_engine == "amber":
            return amber_trajectory_suffix(
                self.config.engine.amber_trajectory_format,
                self.config.engine.amber_executable,
            )
        return "xtc"

    def _inherit_walker_states(self, parents: list[int], fallback: list) -> list:
        """Build velocity-inheriting start states for the next iteration's walkers.

        ``parents[i]`` is the current-iteration walker index that spawned walker
        ``i`` (from the WE spawner). Each walker wrote its endpoint State beside its
        trajectory as ``<traj>.endstate.npz``; we load positions/velocities/box from
        the selected parents so the next segment continues the parent's dynamics.
        Split children share a parent State and decorrelate through the Langevin
        noise -- the standard weighted-ensemble split. Any walker whose endstate is
        missing (e.g. it failed) falls back to its position-only start state, so a
        single bad walker degrades to a fresh-velocity restart rather than aborting.
        """
        import numpy as np
        from openmm import unit

        iter_dir = self.outdir / f"iter_{self.iteration}"
        suffix = self._traj_suffix()
        states: list = []
        for i, parent in enumerate(parents):
            if parent < 0:
                # Recycled walker: a NEW trajectory launched from the basis state, so
                # it must draw fresh Maxwell-Boltzmann velocities rather than inherit
                # a parent's. Inheriting here would continue the very trajectory that
                # was just terminated at the target, breaking the steady state.
                states.append(fallback[i])
                continue
            # `parent` indexes the LIVE ensemble, but endstate files are named by the
            # RAW walker index. Those coincide only when nothing failed; after a drop
            # they diverge, and the walker would silently inherit velocities from a
            # different trajectory than its positions -- continuous-looking dynamics
            # stitched from two different walkers.
            live = getattr(self, "_live_walker_indices", None)
            raw = live[parent] if live is not None and parent < len(live) else parent
            endstate = iter_dir / f"iteration_{self.iteration}_{raw}.{suffix}.endstate.npz"
            if not endstate.exists():
                states.append(fallback[i])
                continue
            data = np.load(endstate)
            states.append(
                {
                    "positions": data["positions"] * unit.nanometer,
                    "velocities": data["velocities"] * (unit.nanometer / unit.picosecond),
                    "box_vectors": data["box"] * unit.nanometer,
                }
            )
        return states

    def _recycling_basis_state(
        self,
        feature_extractor,
        sampling_trajectories: list[str],
        basis_index: int,
        expected_cv: Any = None,
        observed_cv: Any = None,
    ) -> dict[str, Any]:
        """Structure a recycled walker restarts from, captured once and held fixed.

        A recycled walker is a new trajectory launched from the basis (source) state,
        and its weight is booked as flux into the target. The spawner names it by
        frame index -- but WE does not pool history, so an index only ever addresses
        the CURRENT iteration's frames. At iteration N, `basis_index` therefore
        resolves to wherever walker `basis_index` happens to be *now*, which is not
        the basis. If that walker has drifted toward the target, the recycled walker
        restarts next to the sink, re-enters almost immediately, and the flux -- hence
        MFPT = 1/flux -- is biased fast. The index cannot carry this; only the
        structure can.

        Capturing on the first spawn matches `WESpawner.basis_cv`, which freezes on
        its first call for exactly the same reason, so the CV a recycled walker is
        binned at and the structure it actually restarts from are the same state.
        Persisted to disk so a resumed run reloads the original basis instead of
        re-capturing from a mid-run frame.
        """
        import numpy as np
        from openmm import Vec3
        from openmm.unit import nanometer

        if self._basis_state is not None:
            return self._basis_state

        path = self.outdir / "recycling_basis_state.npz"
        if path.exists():
            data = np.load(path)
            box = data["box"]
            self._basis_state = {
                "positions": data["positions"] * nanometer,
                "box_vectors": (
                    None
                    if box.shape[0] == 0
                    else tuple(Vec3(*row) * nanometer for row in box)
                ),
            }
            return self._basis_state

        # Fresh capture. The CV a recycled walker is BINNED at (the spawner's frozen
        # `basis_cv`) and the structure it RESTARTS from (captured here) live in two
        # separate persistence paths -- the spawner checkpoint and an .npz -- so
        # nothing but this check keeps them describing the same state. On a fresh run
        # they agree by construction. They disagree exactly when the spawner restored
        # `basis_cv` from a checkpoint but the .npz is absent (an interrupted or
        # hand-moved run dir): the structure would then be re-captured from a mid-run
        # frame while the CV stayed original, reinstating the very drift this exists to
        # prevent -- and doing it silently, since a plausible structure is still
        # produced. Refuse rather than emit a biased rate.
        if expected_cv is not None and observed_cv is not None:
            expected_arr = np.asarray(expected_cv, dtype=float)
            observed_arr = np.asarray(observed_cv, dtype=float)
            if not np.allclose(expected_arr, observed_arr):
                raise RuntimeError(
                    "Recycling basis is inconsistent: the spawner is binning recycled "
                    f"walkers at CV {expected_arr.tolist()}, but frame {basis_index} of "
                    f"this iteration -- the frame the basis structure would be captured "
                    f"from -- is at CV {observed_arr.tolist()}. This usually means the "
                    "run resumed from a checkpoint while "
                    f"'{path.name}' was missing, so the structure would come from a "
                    "mid-run frame and the recycled walkers would restart somewhere "
                    "other than the source, biasing MFPT = 1/flux. Restore that file "
                    "from the original run directory, or start a fresh run."
                )

        state = feature_extractor.extract_positions_by_indices(
            sampling_trajectories, [basis_index]
        )[0]
        box_vectors = state.get("box_vectors")
        box_array = (
            np.zeros((0, 3), dtype=float)
            if box_vectors is None
            else np.array(
                [list(v.value_in_unit(nanometer)) for v in box_vectors], dtype=float
            )
        )
        # np.savez appends ".npz" to a name that lacks it, which silently breaks an
        # atomic rename onto `path`; write to a temp that already carries the suffix.
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez(
            tmp,
            positions=np.asarray(state["positions"].value_in_unit(nanometer), dtype=float),
            box=box_array,
        )
        tmp.replace(path)
        self._basis_state = state
        return self._basis_state

    def run_iteration(self, walkers: list[Any]):
        """Run a single adaptive sampling iteration."""

        # 1. Run production MD
        import time

        runner_start_time = time.time()

        engine_kwargs = {
            k: v for k, v in self.config.engine.model_dump().items() if k != "md_engine"
        }
        if engine_kwargs.get("seed") is None:
            engine_kwargs["seed"] = self.config.random_seed
        # Kinetics mode: ask the engine to persist each walker's endpoint State so
        # the next segment can inherit its velocities (continuous WE dynamics).
        if getattr(self.config.spawning, "inherit_velocities", False):
            engine_kwargs["save_endstate"] = True
        prepare_kwargs = {
            "conf": Path(self.config.system.conf_file),
            "top": Path(self.config.system.top_file),
            "system_file": Path(self.config.system.system_file)
            if self.config.system.system_file
            else None,
        }

        results = run_iteration_parallel(
            engine_name=self.config.engine.md_engine,
            engine_kwargs=engine_kwargs,
            prepare_kwargs=prepare_kwargs,
            walkers=walkers,
            steps=self.config.spawning.step,
            stride=self.config.spawning.stride,
            outdir=self.outdir / f"iter_{self.iteration}",
            iteration=self.iteration,
            max_workers=self.config.spawning.max_workers,
            gpu_ids=self.config.engine.gpu_ids,
            execution=getattr(self.config, "execution", None),
        )

        runner_time = time.time() - runner_start_time
        n_walkers = len(results)
        n_ok = sum(1 for ok in results if ok)
        min_fraction = getattr(self.config, "min_success_fraction", 1.0)
        # Abort only when too few walkers succeeded. With the default
        # min_success_fraction=1.0 this fires on any failure (strict legacy
        # behaviour); a lower threshold lets a long HPC campaign shrug off a few
        # transient walker failures per iteration and continue with the
        # survivors instead of discarding days of progress.
        if n_ok == 0 or n_ok < min_fraction * n_walkers:
            failed = n_walkers - n_ok
            iteration_dir = self.outdir / f"iter_{self.iteration}"
            expected = [
                iteration_dir / f"iteration_{self.iteration}_{idx}.{self._traj_suffix()}"
                for idx, ok in enumerate(results)
                if not ok
            ]
            missing = [str(path) for path in expected if not path.exists()]
            detail = (
                f" Missing trajectory files: {', '.join(missing[:5])}"
                if missing
                else ""
            )
            hint = ""
            if failed == n_walkers and self.iteration == 0:
                # Every walker died on the very first iteration. Overwhelmingly the
                # cause is an unrelaxed starting structure (steric clashes -> NaN),
                # not a broken engine or GPU. Say so, instead of leaving the user with
                # N identical OpenMM NaN tracebacks to interpret.
                hint = (
                    "\nHINT: all walkers failed on iteration 0. The most common cause is a "
                    "starting structure that has not been energy-minimized/equilibrated "
                    "(steric clashes integrate to NaN on the first steps). Minimize and "
                    "equilibrate the structure before the campaign, set 'engine.equilibrate: "
                    "true', or reduce 'engine.dt'. A NaN here is almost never a force-field "
                    "or GPU fault."
                )
            raise RuntimeError(
                f"{failed}/{n_walkers} walker(s) failed during iteration "
                f"{self.iteration} (min_success_fraction={min_fraction}); "
                "stopping before CV extraction." + detail + hint
            )
        other_start_time = time.time()
        if len(self.walker_parents) != len(walkers):
            self.walker_parents = [None for _ in walkers]

        # Keep only the walkers that produced usable trajectories; drop the
        # failed ones (and their lineage parents) so downstream indexing stays
        # consistent. When every walker succeeds this is a no-op.
        ok_indices = [idx for idx, ok in enumerate(results) if ok]
        # Publish the survivor mapping for spawners that carry per-walker state across
        # iterations (weighted ensemble carries WEIGHTS). Walker i of the live ensemble
        # is walker ok_indices[i] of the previous resampling; without that mapping a
        # spawner can only guess, and a wrong guess re-attaches weights to the wrong
        # trajectories while sum(w) == 1 still holds -- silent, and fatal to a rate.
        self._live_walker_indices = list(ok_indices)
        if len(ok_indices) < n_walkers:
            logging.warning(
                "Iteration %d: %d/%d walkers failed; continuing with %d survivors.",
                self.iteration,
                n_walkers - n_ok,
                n_walkers,
                n_ok,
            )
            self.walker_parents = [self.walker_parents[i] for i in ok_indices]

        # 2. Extract and project coordinates
        trajectories = [
            str(
                self.outdir
                / f"iter_{self.iteration}"
                / f"iteration_{self.iteration}_{idx}.{self._traj_suffix()}"
            )
            for idx in ok_indices
        ]
        self._validate_trajectory_files(trajectories)
        trajectory_topology = (
            self.config.system.trajectory_topology_file or self.config.system.top_file
        )
        feature_extractor = FeatureExtractor(
            topology=trajectory_topology, selection=self.config.system.feature_selection
        )

        if is_adaptive_space(self.config.space_mode):
            features = self._extract_adaptive_features(
                feature_extractor, trajectories
            )

            if self.config.save_features:
                np.savez_compressed(
                    self.outdir / f"iter_{self.iteration}" / "features.npz",
                    features=features,
                )

            if self.config.aggregate_memory:
                self.feature_memory.append(features)
                self._prune_feature_memory()

            # Train or update the Space Model (cadence decided by RetrainController)
            n_frames = self.config.spawning.step // self.config.spawning.stride
            has_model = self.space_model is not None
            current_cv_score = None
            if has_model and self.config.retrain_policy == "vamp_adaptive":
                current_cv_score = self._cv_vamp_score(features, n_frames, len(walkers))
            do_retrain = self.retrain_controller.should_retrain(
                self.iteration, has_model, current_cv_score
            )

            if do_retrain:
                if self.space_model is None:
                    from trails_md.spaces import AdaptiveSpaceModel

                    logging.info(f"Training new {self.config.space_mode} model...")
                    self.space_model = AdaptiveSpaceModel(
                        **self._adaptive_model_kwargs()
                    )
                else:
                    reason = self.retrain_controller.last_reason or "scheduled"
                    logging.info(
                        "Retraining %s model (iteration %d): %s",
                        self.config.space_mode,
                        self.iteration,
                        reason,
                    )

                if self.config.aggregate_memory:
                    self._prune_feature_memory()
                    fit_features = np.vstack(self.feature_memory)
                    total_walkers = len(self.feature_memory) * len(walkers)
                else:
                    fit_features = features
                    total_walkers = len(walkers)

                self._maybe_select_features(fit_features, n_frames, total_walkers)
                self.space_model.fit(
                    self._fs_apply(fit_features),
                    walker_length=n_frames,
                    n_walkers=total_walkers,
                )
                self.scaler = self.space_model.scaler
                self.adaptive_space_version += 1
                self._refresh_adaptive_history_projections()
                self.retrain_controller.notify_retrained(
                    self._cv_vamp_score(features, n_frames, len(walkers))
                )
            else:
                self.retrain_controller.notify_skipped()

            # Project onto latent space (project only current iteration features for history tracking)
            logging.info("Projecting features to latent space...")
            points = self.space_model.project(self._fs_apply(features))
        else:
            # Physical fixed space
            if self.config.system.project_file:
                points = self._extract_physical_cvs(trajectories)
            else:
                # Fallback to internal Rg-RMSD
                points = feature_extractor.extract_rg_rmsd(
                    trajectories=trajectories,
                    reference_pdb=self.config.system.conf_file,
                )

        # Always extract physical CVs for evaluation/plotting if project_file is provided
        if (
            is_adaptive_space(self.config.space_mode)
            and self.config.system.project_file
        ):
            try:
                physical_points = self._extract_physical_cvs(trajectories)
                np.savez_compressed(
                    self.outdir / f"iter_{self.iteration}" / "physical_cvs.npz",
                    cvs=physical_points,
                )
            except (ImportError, AttributeError, RuntimeError, OSError, ValueError) as e:
                logging.warning(f"Failed to extract physical CVs for evaluation: {e}")

        np.savez_compressed(
            self.outdir / f"iter_{self.iteration}" / "cvs.npz", cvs=points
        )

        expected_frames = self.config.spawning.step // self.config.spawning.stride
        current_frame_records = build_frame_records(
            iteration=self.iteration,
            trajectories=trajectories,
            points=np.asarray(points),
            walker_parents=self.walker_parents,
            expected_frames=expected_frames,
        )
        self._prune_unusable_history_trajectories()
        # Hand the MSM-guided spawner the previous iteration's MSM + its clustering
        # (consistent with each other; the spawner falls back to least-counts when
        # either is None, e.g. iteration 0 or just after a resume).
        if hasattr(self.spawner, "msm_result"):
            self.spawner.msm_result = self.last_msm_result
            self.spawner.cluster_model = getattr(
                self.msm_estimator, "_cluster_model", None
            )
        # Which of last iteration's walkers survived. A weighted-ensemble spawner
        # cannot infer this from the frame count: failed walkers are dropped, so the
        # frames it receives are the survivors' only, and guessing the ensemble size
        # geometrically misaligns walker <-> weight <-> endpoint (see
        # WESpawner._live_weights). `self._live_walker_indices` is set wherever the
        # failed walkers are dropped, and indexes the PREVIOUS resampling's output.
        if hasattr(self.spawner, "live_walker_indices"):
            self.spawner.live_walker_indices = getattr(
                self, "_live_walker_indices", None
            )
        # Spawners that only ever pick from the CURRENT walkers (weighted ensemble --
        # a historical frame has no well-defined weight) must not be handed the
        # history, and must not have it pooled for index mapping either: pooling costs
        # one file-open per past iteration on every spawn step, so the per-iteration
        # overhead grows without bound (measured: +0.8 s per iteration, 7.9 s at
        # iteration 5 -> 11.1 s at iteration 9), which makes long kinetics runs
        # impossible. Exploration spawners DO reach into history and keep the pooling.
        uses_history = getattr(self.spawner, "uses_history", True)
        spawn_indices = self.spawner.sample(
            points,
            self.config.spawning.walker,
            history=self.history if uses_history else {},
        )
        # The spawner pooled historical frames matching the current projection
        # dimension; build the trajectory and frame-record lists over the *same*
        # iterations so each spawn index maps to the frame it was scored on.
        target_dim = projection_dim(points)
        if uses_history:
            sampling_trajectories = self._sampling_trajectories(
                trajectories, target_dim=target_dim
            )
            sampling_frame_records = self._sampling_frame_records(
                current_frame_records, target_dim=target_dim
            )
        else:
            # Indices are already local to this iteration's frames.
            sampling_trajectories = list(trajectories)
            sampling_frame_records = current_frame_records
        self._validate_sampling_trajectories(
            sampling_trajectories,
            context=f"spawning iteration {self.iteration}",
        )
        next_walker_parents = [
            map_global_frame(sampling_frame_records, index)["key"]
            for index in spawn_indices
        ]
        next_walkers = feature_extractor.extract_positions_by_indices(
            sampling_trajectories, spawn_indices
        )
        # Source->sink recycling: a recycled walker (parent -1) must restart from the
        # BASIS structure. `spawn_indices` cannot express that -- see
        # `_recycling_basis_state` -- so substitute the frozen basis here, before
        # velocity inheritance, which already treats parent -1 as a fresh-velocity
        # start and so needs only the right positions underneath it.
        if getattr(self.config.spawning, "recycle_target", None) is not None:
            # Capture on the FIRST spawn, whether or not anything recycled this
            # iteration: the spawner freezes `basis_cv` on its first sample() call,
            # and the CV and the structure must be captured from the same state.
            # Deferring to the first recycling event would re-introduce the very
            # drift this exists to prevent.
            basis_index = int(
                getattr(self.config.spawning, "recycle_basis_index", 0)
            )
            # Cross-check the structure against the CV the spawner froze; see
            # `_recycling_basis_state`. `points` and `sampling_trajectories` index the
            # same frames here (WE does not pool history), so points[basis_index] is
            # the CV of the very frame the structure is taken from.
            basis = self._recycling_basis_state(
                feature_extractor,
                sampling_trajectories,
                basis_index,
                expected_cv=getattr(self.spawner, "basis_cv", None),
                observed_cv=(
                    points[basis_index] if basis_index < len(points) else None
                ),
            )
            parents = getattr(self.spawner, "selected_parents", None)
            if parents is not None:
                for i, parent in enumerate(parents):
                    if parent < 0:
                        next_walkers[i] = dict(basis)
        # Kinetics mode: replace the position-only start states with full endpoint
        # States (positions + velocities + box) of the parent walkers, so the next
        # segment CONTINUES the dynamics instead of redrawing velocities. Only WE
        # exposes `selected_parents` (current-iteration walker indices), and only
        # there is velocity inheritance meaningful, so this is a no-op otherwise.
        if getattr(self.config.spawning, "inherit_velocities", False):
            parents = getattr(self.spawner, "selected_parents", None)
            if parents is not None:
                next_walkers = self._inherit_walker_states(parents, next_walkers)

        # Save the current projection and trajectory files after sampling; spawners
        # combine completed history with current points internally.
        self.history[self.iteration] = {
            "projection": points,
            "spawning_scheme": self.config.spawning.spawn_scheme,
            "trajectories": trajectories,
            "spawn_indices": spawn_indices,
            "frames": current_frame_records,
            "walker_parents": list(self.walker_parents),
            "next_walker_parents": next_walker_parents,
        }
        if is_adaptive_space(self.config.space_mode):
            self.history[self.iteration]["features"] = features
            self.history[self.iteration]["space_version"] = self.adaptive_space_version

        self.iteration += 1
        self.walker_parents = next_walker_parents
        other_time = time.time() - other_start_time

        # Compute bin occupancy for logging
        occupied_bins = None
        total_bins = None
        cumulative_frames = None
        try:
            from trails_md.binning.spatial import RegularBinner

            is_adaptive = is_adaptive_space(self.config.space_mode)
            binner = RegularBinner(
                n_bins=self.config.n_bins,
                min_values=None if is_adaptive else self.config.min_values,
                max_values=None if is_adaptive else self.config.max_values,
            )
            # Gather historical projections in the current CV dimension only, so a
            # lower-dimensional initial-trajectory injection cannot make vstack
            # raise and silently disable occupancy/convergence tracking.
            cumulative_points = [
                np.asarray(self.history[iteration]["projection"])
                for iteration in pooled_history_iterations(
                    self.history, projection_dim(points)
                )
            ]
            if cumulative_points:
                all_points = np.vstack(cumulative_points)
                table = binner.fit(all_points)
                occupied_bins = len(table.occupied_indices)
                total_bins = len(table.ids)
                cumulative_frames = len(all_points)
            else:
                occupied_bins = 0
                total_bins = np.prod(binner.n_bins)
                cumulative_frames = 0
            bin_occupancy_str = f"{occupied_bins}/{total_bins}"

        except Exception as e:
            bin_occupancy_str = "N/A"
            # WARNING, not DEBUG: occupancy is a reported diagnostic, and at the
            # default log level a DEBUG line is invisible.
            logging.warning(
                "Iteration %d: failed to compute bin occupancy: %s",
                self.iteration,
                e,
            )

        # Adaptive resolution + convergence. Deliberately OUTSIDE the occupancy
        # try/except that used to enclose it: any exception raised while computing
        # occupancy also disabled resolution bumps AND convergence detection for the
        # remainder of the run, visible only as "bin_occupancy: N/A" plus a DEBUG
        # line -- so a campaign could run to completion having silently never
        # evaluated convergence. A failure here is worth surfacing on its own.
        try:
            self._update_resolution_and_convergence(occupied_bins)
        except Exception as e:
            logging.warning(
                "Iteration %d: resolution/convergence update failed: %s. "
                "Convergence was NOT evaluated for this iteration.",
                self.iteration,
                e,
            )

        # MSM estimation + MSM-based convergence (opt-in).
        self._maybe_build_msm()

        current_iteration = self.iteration - 1

        # Save the checkpoint LAST, after occupancy/resolution/convergence and
        # MSM bookkeeping have run, so a resumed ``iter_N`` reflects the fully
        # completed iteration N (an adaptive n_bins bump or a convergence flag
        # set during post-processing would otherwise be lost on resume).
        if (
            self.config.checkpoint_freq > 0
            and current_iteration % self.config.checkpoint_freq == 0
        ):
            self.checkpoint_manager.save(
                iteration=current_iteration,
                space_model=self.space_model,
                scaler=self.scaler,
                bin_state=self.bin_state,
                history=self.history,
                sampler_state=self._checkpoint_state(),
            )

        self._append_iteration_log(
            iteration=current_iteration,
            runner_time=runner_time,
            other_time=other_time,
            success=results,
            points=points,
            occupied_bins=occupied_bins,
            total_bins=total_bins,
            cumulative_frames=cumulative_frames,
            spawn_indices=spawn_indices,
            trajectories=trajectories,
        )

        # Informative per-iteration banner (presentation extracted to reporting).
        self.reporter.print_summary(
            iteration=self.iteration - 1,
            runner_time=runner_time,
            other_time=other_time,
            occupancy=bin_occupancy_str,
        )

        return {
            "success": results,
            "spawn_indices": spawn_indices,
            "walkers": next_walkers,
            "projection": points,
            "converged": self.converged,
            "convergence_reason": self.convergence_reason,
        }

    def _sampling_trajectories(
        self, current_trajectories: list[str], target_dim: int | None = None
    ) -> list[str]:
        # ``target_dim`` selects the same historical iterations the spawner pooled
        # (via pooled_history_iterations), so a spawn index maps to the intended
        # trajectory even when history mixes projection dimensionalities.
        trajectories: list[str] = []
        for iteration in pooled_history_iterations(self.history, target_dim):
            entry = self.history[iteration]
            stored = entry.get("trajectories")
            if stored:
                trajectories.extend(str(path) for path in stored)
                continue
            trajectories.extend(
                self._infer_iteration_trajectories(iteration, entry["projection"])
            )
        trajectories.extend(current_trajectories)
        return trajectories

    def _sampling_frame_records(
        self,
        current_frame_records: list[dict[str, Any]],
        target_dim: int | None = None,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for iteration in pooled_history_iterations(self.history, target_dim):
            entry = self.history[iteration]
            stored = entry.get("frames")
            if stored:
                records.extend(stored)
                continue
            inferred = build_frame_records(
                iteration=iteration,
                trajectories=self._infer_iteration_trajectories(
                    iteration, entry["projection"]
                ),
                points=np.asarray(entry["projection"]),
                walker_parents=[],
                expected_frames=self.config.spawning.step
                // self.config.spawning.stride,
            )
            records.extend(inferred)
        records.extend(current_frame_records)
        return records

    def _infer_iteration_trajectories(
        self, iteration: int, projection: Any
    ) -> list[str]:
        projection = np.asarray(projection)
        frames_per_walker = self.config.spawning.step // self.config.spawning.stride
        if frames_per_walker <= 0:
            return []
        n_walkers = int(np.ceil(len(projection) / frames_per_walker))
        return [
            str(self.outdir / f"iter_{iteration}" / f"iteration_{iteration}_{idx}.{self._traj_suffix()}")
            for idx in range(n_walkers)
        ]

    def _refresh_adaptive_history_projections(self) -> None:
        """Keep cumulative adaptive sampling history in the current latent space."""
        if self.space_model is None:
            return

        for iteration in sorted(self.history):
            entry = self.history[iteration]
            if not isinstance(entry, dict):
                continue
            features = entry.get("features")
            if features is None:
                if entry.get("projection") is not None:
                    logging.warning(
                        "Dropping iteration %s from adaptive sampling history because "
                        "its raw features are unavailable after retraining.",
                        iteration,
                    )
                    entry["projection"] = None
                continue
            entry["projection"] = self.space_model.project(
                self._fs_apply(np.asarray(features, dtype=float))
            )
            entry["space_version"] = self.adaptive_space_version

    def _prune_feature_memory(self) -> None:
        """Cap feature memory size to prevent O(N) memory growth and slow training."""
        if not getattr(self.config, "aggregate_memory", False) or not self.feature_memory:
            return
        max_frames = getattr(self.config, "max_adaptive_memory_frames", 50000)
        first_len = len(self.feature_memory[0]) if len(self.feature_memory[0]) > 0 else 1
        max_iters = max(5, max_frames // first_len)
        if len(self.feature_memory) > max_iters:
            # Keep initial anchor (0), keep recent quarter, randomly sample middle
            anchor = [self.feature_memory[0]]
            n_recent = max(1, (max_iters - 1) // 4)
            n_middle = (max_iters - 1) - n_recent
            recent = self.feature_memory[-n_recent:]
            middle_candidates = self.feature_memory[1:-n_recent] if n_middle > 0 else []
            if len(middle_candidates) > n_middle:
                # Instance-bound RNG (not the global ``random``) so pruning cannot
                # be desynchronised by external RNG use; chronological order kept.
                picks = sorted(
                    self.seed_manager.rng.choice(
                        len(middle_candidates), size=n_middle, replace=False
                    )
                )
                middle = [middle_candidates[i] for i in picks]
            else:
                middle = middle_candidates
            self.feature_memory = anchor + middle + recent

    def _restore_feature_memory_from_history(self) -> None:
        """Rebuild adaptive feature memory from checkpointed history when possible."""
        if not is_adaptive_space(self.config.space_mode):
            return

        self.feature_memory = []
        for iteration in sorted(self.history):
            entry = self.history[iteration]
            if not isinstance(entry, dict):
                continue
            features = entry.get("features")
            if features is not None:
                self.feature_memory.append(np.asarray(features, dtype=float))
        self._prune_feature_memory()

        versions = [
            entry.get("space_version")
            for entry in self.history.values()
            if isinstance(entry, dict) and entry.get("space_version") is not None
        ]
        if versions:
            self.adaptive_space_version = max(int(version) for version in versions)

    def _restore_walker_parents_from_history(self) -> None:
        if not self.history:
            self.walker_parents = []
            return
        latest_iteration = max(self.history)
        latest_entry = self.history[latest_iteration]
        if isinstance(latest_entry, dict):
            self.walker_parents = list(latest_entry.get("next_walker_parents") or [])

    def _checkpoint_state(self) -> dict[str, Any]:
        from trails_md.utils.seeds import capture_rng_state

        return {
            "n_bins": list(self.config.n_bins),
            "voronoi_clusters": self.config.spawning.voronoi_clusters,
            "occupancy_history": list(self.occupancy_history),
            "last_occupied_bins": self.last_occupied_bins,
            "convergence_stall_count": self.convergence_stall_count,
            "converged": self.converged,
            "convergence_reason": self.convergence_reason,
            "msm_monitor": self.msm_monitor.state_dict()
            if self.msm_monitor is not None
            else None,
            "feature_selection_indices": self.feature_selection_indices,
            "selected_feature_type": self.selected_feature_type,
            "retrain_controller": self.retrain_controller.state_dict(),
            "spawner": self.spawner.state_dict()
            if hasattr(self.spawner, "state_dict")
            else None,
            # RNG state so a resumed run reproduces an uninterrupted one's stream.
            "rng_state": capture_rng_state(),
            # Instance-bound sampling generator (initial-walker replication,
            # feature-memory pruning); separate from the global RNG stream above.
            "seed_manager_rng": self.seed_manager.rng.bit_generator.state,
        }

    def _restore_sampler_state(self, state: dict[str, Any] | None) -> None:
        state = state or {}
        if "n_bins" in state:
            self.config.n_bins = list(state["n_bins"])
            if hasattr(self.spawner, "n_bins"):
                self.spawner.n_bins = list(state["n_bins"])
        if "voronoi_clusters" in state:
            self.config.spawning.voronoi_clusters = int(state["voronoi_clusters"])
            if hasattr(self.spawner, "n_clusters"):
                self.spawner.n_clusters = int(state["voronoi_clusters"])
        self.occupancy_history = list(state.get("occupancy_history", []))
        self.last_occupied_bins = state.get("last_occupied_bins")
        self.convergence_stall_count = int(state.get("convergence_stall_count", 0))
        self.converged = bool(state.get("converged", False))
        self.convergence_reason = state.get("convergence_reason")
        if state.get("feature_selection_indices") is not None:
            self.feature_selection_indices = list(state["feature_selection_indices"])
        if state.get("selected_feature_type") is not None:
            self.selected_feature_type = state["selected_feature_type"]
        self.retrain_controller.load_state_dict(state.get("retrain_controller", {}))
        if self.msm_monitor is not None and state.get("msm_monitor"):
            self.msm_monitor.load_state_dict(state["msm_monitor"])
        if state.get("spawner") is not None and hasattr(self.spawner, "load_state_dict"):
            self.spawner.load_state_dict(state["spawner"])
        # Restore the RNG stream last so it overrides the base seed set in
        # __init__ — resumed spawn/training draws then match an uninterrupted run.
        if state.get("rng_state") is not None:
            from trails_md.utils.seeds import restore_rng_state

            restore_rng_state(state["rng_state"])
        if state.get("seed_manager_rng") is not None:
            try:
                self.seed_manager.rng.bit_generator.state = state["seed_manager_rng"]
            except (ValueError, KeyError, TypeError) as exc:
                logging.warning(
                    "Could not restore instance sampling RNG on resume: %s", exc
                )

    @staticmethod
    def _trajectory_file_problems(trajectories: list[str]) -> list[str]:
        bad: list[str] = []
        for path in trajectories:
            p = Path(path)
            if not p.is_file():
                bad.append(f"missing: {path}")
            elif p.stat().st_size == 0:
                bad.append(f"empty: {path}")
        return bad

    @staticmethod
    def _validate_trajectory_files(trajectories: list[str]) -> None:
        """Ensure each expected trajectory exists and is non-empty before reading.

        A walker can report success yet leave a missing or truncated file (disk
        full, killed writer); catching it here gives a clear error instead of an
        opaque downstream parse failure.
        """
        bad = TrailsMDCore._trajectory_file_problems(trajectories)
        if bad:
            joined = "\n  - ".join(bad)
            raise RuntimeError(
                "Trajectory files are not usable for CV extraction:\n  - " + joined
            )

    @staticmethod
    def _validate_sampling_trajectories(
        trajectories: list[str], context: str = "sampling"
    ) -> None:
        """Ensure cumulative trajectories are still available before spawning.

        Current iteration outputs are validated before CV extraction. Spawning can
        also sample frames from older history, especially after resume, so check
        the complete sampling pool before handing paths to MDAnalysis.
        """
        try:
            TrailsMDCore._validate_trajectory_files(trajectories)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Cannot extract walker start coordinates during {context}; "
                "one or more sampled-history trajectory files are missing or empty. "
                "Restore the listed files, remove the incomplete run directory, or "
                "resume from a checkpoint whose trajectory files are present.\n"
                f"{exc}"
            ) from exc

    def _prune_unusable_history_trajectories(self) -> None:
        """Drop old history entries whose trajectory files can no longer be read."""
        dropped: list[tuple[int, list[str]]] = []
        for iteration in sorted(list(self.history)):
            entry = self.history[iteration]
            if not isinstance(entry, dict) or entry.get("projection") is None:
                continue
            stored = entry.get("trajectories")
            trajectories = (
                [str(path) for path in stored]
                if stored
                else self._infer_iteration_trajectories(iteration, entry["projection"])
            )
            problems = self._trajectory_file_problems(trajectories)
            if problems:
                dropped.append((iteration, problems))
                del self.history[iteration]

        for iteration, problems in dropped:
            logging.warning(
                "Dropping iteration %s from sampling history because trajectory "
                "files are missing or empty: %s",
                iteration,
                "; ".join(problems[:5]),
            )

    def _fs_apply(self, features: np.ndarray) -> np.ndarray:
        """Restrict features to the VAMP-selected columns (no-op if disabled)."""
        if self.feature_selection_indices is None:
            return features
        return np.asarray(features)[:, self.feature_selection_indices]

    def _extract_feature_type(
        self, feature_extractor: FeatureExtractor, trajectories: list[str], ftype: str
    ) -> np.ndarray:
        if ftype == "phi_psi":
            angles = feature_extractor.extract_aib9_phi_psi(trajectories)
            if getattr(self.config, "adaptive_angle_encoding", "raw") == "sincos":
                # Continuous [sin, cos] embedding so the periodic dihedral CVs are
                # not torn at ±pi before scaling / CV learning.
                from trails_md.utils.math import encode_angles_sincos

                return encode_angles_sincos(angles)
            return angles
        if ftype == "fitted_coords":
            return feature_extractor.extract_fitted_coords(trajectories)
        return feature_extractor.extract_pairwise_distances(trajectories)

    def _extract_adaptive_features(
        self, feature_extractor: FeatureExtractor, trajectories: list[str]
    ) -> np.ndarray:
        """Extract input features, optionally ranking candidate feature *types* by VAMP-2."""
        fs_cfg = getattr(self.config, "feature_selection", None)
        candidates = list(getattr(fs_cfg, "candidate_feature_types", []) or [])
        default_type = getattr(self.config, "adaptive_feature_type", "distances")

        if not (fs_cfg and fs_cfg.enabled and candidates):
            logging.info("Extracting %s features...", default_type)
            return self._extract_feature_type(
                feature_extractor, trajectories, self.selected_feature_type or default_type
            )

        due = self.selected_feature_type is None or (
            fs_cfg.cadence > 0 and self.iteration % fs_cfg.cadence == 0
        )
        if not due:
            return self._extract_feature_type(
                feature_extractor, trajectories, self.selected_feature_type
            )

        # Extract every candidate type and rank them by VAMP-2.
        from trails_md.spaces.feature_selection import rank_candidates

        n_frames = self.config.spawning.step // self.config.spawning.stride
        extracted: dict[str, np.ndarray] = {}
        for ftype in candidates:
            try:
                extracted[ftype] = self._extract_feature_type(
                    feature_extractor, trajectories, ftype
                )
            except Exception as exc:  # noqa: BLE001 - skip system-incompatible types
                logging.warning("Skipping feature type %r: %s", ftype, exc)
        if not extracted:
            return self._extract_feature_type(feature_extractor, trajectories, default_type)

        candidate_trajs = {
            ftype: [
                feats[i * n_frames : (i + 1) * n_frames]
                for i in range(max(1, len(feats) // max(1, n_frames)))
            ]
            for ftype, feats in extracted.items()
        }
        ranked = rank_candidates(candidate_trajs, fs_cfg.lagtime)
        best = ranked[0][0]
        if best != self.selected_feature_type:
            # Column indices are tied to a feature type; reset on a type change.
            self.feature_selection_indices = None
            logging.info(
                "VAMP-2 feature-type selection -> %s (scores: %s)",
                best,
                ", ".join(f"{n}={s:.3f}" for n, s in ranked),
            )
        self.selected_feature_type = best
        return extracted[best]

    def _cv_vamp_score(
        self, features: np.ndarray, walker_length: int, n_walkers: int
    ) -> float | None:
        """VAMP-2 score of the current CV's latent projection of ``features``."""
        if self.space_model is None:
            return None
        try:
            latent = np.asarray(self.space_model.project(self._fs_apply(features)))
        except Exception:  # noqa: BLE001 - scoring is best-effort
            return None
        if latent.ndim == 1:
            latent = latent.reshape(-1, 1)
        from trails_md.spaces.feature_selection import vamp2_score

        lag = self.config.adaptive_model.lagtime
        trajs = [
            latent[i * walker_length : (i + 1) * walker_length]
            for i in range(max(1, n_walkers))
        ]
        trajs = [t for t in trajs if len(t) > lag]
        if not trajs:
            return None
        try:
            return vamp2_score(trajs, lag)
        except (ValueError, np.linalg.LinAlgError):
            return None

    def _maybe_select_features(
        self, fit_features: np.ndarray, walker_length: int, n_walkers: int
    ) -> None:
        """(Re)select input-feature columns by VAMP-2 when due (opt-in)."""
        if self.feature_selector is None:
            return
        cadence = self.config.feature_selection.cadence
        due = self.feature_selection_indices is None or (
            cadence > 0 and self.iteration % cadence == 0
        )
        if not due:
            return

        trajs = [
            np.asarray(fit_features[i * walker_length : (i + 1) * walker_length])
            for i in range(max(1, n_walkers))
        ]
        trajs = [t for t in trajs if len(t) > self.feature_selector.lagtime]
        if not trajs:
            return
        try:
            selection = self.feature_selector.select(trajs)
        except (ValueError, np.linalg.LinAlgError) as exc:
            logging.warning("Feature selection skipped: %s", exc)
            return
        self.feature_selection_indices = selection.columns
        self.last_feature_selection = selection
        logging.info(
            "VAMP-2 feature selection: kept %d/%d features (score %.3f).",
            len(selection.columns),
            np.asarray(fit_features).shape[1],
            selection.score,
        )

    def _extract_physical_cvs(self, trajectories: list[str]) -> np.ndarray:
        """Load the user project file and extract physical CVs for ``trajectories``.

        Centralises the ``project_file`` import + ``extract_cvs`` call that was
        previously duplicated for both the fixed-space projection and the
        evaluation-only physical CVs.
        """
        project_path = Path(self.config.system.project_file)
        spec = importlib.util.spec_from_file_location(
            "custom_project", str(project_path)
        )
        custom_project = importlib.util.module_from_spec(spec)
        sys.modules["custom_project"] = custom_project
        spec.loader.exec_module(custom_project)
        return custom_project.extract_cvs(
            trajectories=trajectories,
            top_file=self.config.system.top_file,
            conf_file=self.config.system.conf_file,
        )

    def _collect_msm_trajectories(self) -> list[Any]:
        """Split cumulative history projections into continuous per-walker trajectories.

        Each short walker is one continuous trajectory; transition counts are
        pooled across all of them by the estimator. Segmentation uses the stored
        per-walker frame records (``entry["frames"]``), which carry the *actual*
        frame count each walker's trajectory was written with. This matters
        because engines disagree on frame count: GROMACS writes the ``t=0`` frame
        (``step//stride + 1`` frames per walker) while OpenMM and Amber write
        ``step//stride``; an early-terminated walker differs again. Slicing a
        concatenated projection by a *constant* ``step//stride`` would stitch the
        tail of one walker to the head of the next and inject spurious
        inter-walker transitions into the MSM count matrix. Single-frame segments
        are dropped (they carry no transitions).

        Only projections in the current (most recent) CV/latent dimensionality
        are included, so a lower-dimensional initial-trajectory injection (e.g. a
        2-D physical-CV ``iter -1`` alongside an n-D adaptive space) cannot be
        mixed into the clustering.
        """
        trajs: list[Any] = []
        target_dim = self._latest_projection_dim()
        for iteration in sorted(self.history):
            entry = self.history[iteration]
            if not isinstance(entry, dict):
                continue
            projection = entry.get("projection")
            if projection is None:
                continue
            projection = np.asarray(projection, dtype=float)
            if projection.ndim == 1:
                projection = projection.reshape(-1, 1)
            if target_dim is not None and projection.shape[1] != target_dim:
                continue
            for segment in self._segment_projection_by_walker(
                projection, entry.get("frames")
            ):
                if len(segment) > 1:
                    trajs.append(segment)
        return trajs

    def _latest_projection_dim(self) -> int | None:
        """Feature dimension of the most recent non-empty projection, or None."""
        for iteration in sorted(self.history, reverse=True):
            entry = self.history[iteration]
            if isinstance(entry, dict) and entry.get("projection") is not None:
                proj = np.asarray(entry["projection"])
                return 1 if proj.ndim == 1 else int(proj.shape[1])
        return None

    def _segment_projection_by_walker(
        self, projection: np.ndarray, frames: list[dict[str, Any]] | None
    ):
        """Yield one contiguous projection segment per walker.

        Uses the stored frame records (row-aligned 1:1 with ``projection``) to
        find exact per-walker boundaries. Falls back to a constant
        ``step//stride`` slice only for legacy history entries written before
        frame-record storage (or when the record count does not match the
        projection length).
        """
        n = len(projection)
        if frames and len(frames) == n:
            groups: dict[int, list[int]] = {}
            order: list[int] = []
            for row, record in enumerate(frames):
                walker = int(record.get("walker", 0))
                if walker not in groups:
                    groups[walker] = []
                    order.append(walker)
                groups[walker].append(row)
            for walker in order:
                yield projection[groups[walker]]
            return

        if not getattr(self, "_warned_msm_frame_fallback", False):
            logging.debug(
                "MSM trajectory segmentation: a history entry has no per-walker "
                "frame records; falling back to constant step//stride slicing."
            )
            self._warned_msm_frame_fallback = True
        frames_per_walker = self.config.spawning.step // self.config.spawning.stride
        if frames_per_walker <= 0:
            yield projection
            return
        n_walkers = int(np.ceil(n / frames_per_walker))
        for walker in range(n_walkers):
            yield projection[
                walker * frames_per_walker : (walker + 1) * frames_per_walker
            ]

    def _maybe_build_msm(self) -> None:
        """Estimate the MSM for the just-completed iteration and update convergence."""
        if self.msm_estimator is None or self.msm_monitor is None:
            return

        msm_cfg = self.config.msm
        current_iteration = self.iteration - 1
        if msm_cfg.cadence > 0 and current_iteration % msm_cfg.cadence != 0:
            return

        trajs = self._collect_msm_trajectories()
        total_frames = sum(len(t) for t in trajs)
        if total_frames < msm_cfg.min_frames:
            logging.info(
                "MSM skipped at iteration %d: %d cumulative frames < min_frames %d.",
                current_iteration,
                total_frames,
                msm_cfg.min_frames,
            )
            return

        lag_ok = self._msm_lag_is_assessable(trajs, msm_cfg.lagtime)

        try:
            result = self.msm_estimator.fit(trajs, iteration=current_iteration)
        except Exception as exc:  # noqa: BLE001 - MSM is diagnostic, never fatal
            logging.warning(
                "MSM estimation failed at iteration %d: %s", current_iteration, exc
            )
            return

        self.last_msm_result = result
        logging.info("Iteration %d %s", current_iteration, result.summary())
        self._save_msm_result(current_iteration, result)

        converged = self.msm_monitor.update(result)
        logging.info("MSM convergence %s", self.msm_monitor.status_line())
        if converged and not lag_ok:
            # The criteria are satisfied, but the implied-timescale plateau cannot be
            # assessed within the available segment length -- do NOT certify (see
            # _msm_lag_is_assessable). Convergence across iterations is not the same
            # thing as convergence in lag time.
            logging.warning(
                "MSM criteria satisfied at iteration %d but convergence is NOT certified: "
                "lagtime is too large a fraction of the walker segment length.",
                current_iteration,
            )
            return
        if converged and not self.converged:
            self.converged = True
            self.convergence_reason = self.msm_monitor.reason

    def _msm_lag_is_assessable(self, trajs: list[Any], lagtime: int) -> bool:
        """Can an implied-timescale plateau actually be resolved with these segments?

        Walkers spawn with velocities redrawn from a Maxwell-Boltzmann distribution, which
        severs phase-space continuity at every parent-child boundary. Each walker segment is
        therefore an independent trajectory and the MSM lag time is hard-capped by the segment
        length ``L``: a transition cannot be counted at a lag longer than the trajectory that
        carries it.

        Markovianity additionally requires ``lagtime`` to exceed the momentum relaxation time,
        so the usable window is ``1/gamma << lagtime <= L``. If ``lagtime`` approaches ``L``
        there are too few lagged pairs per segment to see whether the implied timescales have
        plateaued -- and the failure is silent: the convergence monitor watches the ITS across
        *iterations*, so as data accumulate the estimate stops moving and looks converged even
        when it is systematically underestimated.

        We therefore require ``lagtime <= L/5`` before allowing the monitor to certify
        convergence. Campaigns that need a longer lag should lengthen ``spawning.step`` (longer
        walkers) so that ``L`` grows.
        """
        if not trajs:
            return False
        shortest = min(len(t) for t in trajs)
        if lagtime * 5 <= shortest:
            return True
        logging.warning(
            "MSM lagtime=%d is more than 1/5 of the shortest walker segment (%d frames). "
            "The implied-timescale plateau cannot be assessed within a single segment, so "
            "slow timescales may be systematically underestimated. Increase spawning.step "
            "(longer walkers) or reduce msm.lagtime.",
            lagtime,
            shortest,
        )
        return False

    def _save_msm_result(self, iteration: int, result: Any) -> None:
        try:
            vamp2 = np.nan if result.vamp2_score is None else result.vamp2_score
            arrays = {
                "lagtime": np.asarray(result.lagtime),
                "timescales": np.asarray(result.timescales, dtype=float),
                "stationary_distribution": np.asarray(
                    result.stationary_distribution, dtype=float
                ),
                "transition_matrix": np.asarray(
                    result.transition_matrix, dtype=float
                ),
                "cluster_centers": np.asarray(result.cluster_centers, dtype=float),
                "vamp2_score": np.asarray([vamp2], dtype=float),
            }
            if getattr(result, "metastable_populations", None) is not None:
                arrays["metastable_populations"] = np.asarray(
                    result.metastable_populations, dtype=float
                )
            if getattr(result, "count_matrix", None) is not None:
                arrays["count_matrix"] = np.asarray(result.count_matrix, dtype=float)
            if getattr(result, "eigenvectors", None) is not None:
                arrays["eigenvectors"] = np.asarray(result.eigenvectors, dtype=float)
            its = getattr(result, "its", None)
            if its is not None:
                arrays["its_lagtimes"] = np.asarray(its.lagtimes, dtype=float)
                arrays["its_timescales"] = np.asarray(its.timescales, dtype=float)
            np.savez_compressed(self.outdir / f"iter_{iteration}" / "msm.npz", **arrays)
        except Exception as exc:  # noqa: BLE001
            logging.debug("Failed to save msm.npz for iteration %d: %s", iteration, exc)

    def _update_resolution_and_convergence(self, occupied_bins: int | None) -> None:
        if occupied_bins is None:
            return

        self._update_convergence_counter(int(occupied_bins))
        if self.converged:
            return

        self.occupancy_history.append(int(occupied_bins))
        patience = self.config.spawning.resolution_check_patience
        if patience <= 0 or len(self.occupancy_history) < patience:
            return

        recent_occupancies = self.occupancy_history[-patience:]
        if len(set(recent_occupancies)) != 1:
            return

        logging.info(
            "Trigger activated: no exploration for %d consecutive iterations. "
            "Fine-tuning bin space resolution.",
            patience,
        )
        self.occupancy_history = []
        changed = self._increase_sampling_resolution()
        if changed:
            self.convergence_stall_count = 0
        else:
            logging.info(
                "Sampling resolution is already at its configured maximum."
            )

    def _update_convergence_counter(self, occupied_bins: int) -> None:
        if not self._resolution_at_max():
            self.convergence_stall_count = 0
            self.last_occupied_bins = occupied_bins
            return

        if self.last_occupied_bins is None or occupied_bins > self.last_occupied_bins:
            self.convergence_stall_count = 0
        else:
            self.convergence_stall_count += 1
        self.last_occupied_bins = occupied_bins

        patience = self.config.spawning.convergence_patience
        if patience > 0 and self.convergence_stall_count >= patience:
            self.converged = True
            self.convergence_reason = (
                "No new occupied bins for "
                f"{self.convergence_stall_count} iteration(s) after reaching "
                "maximum sampling resolution."
            )
            logging.info("Trails-MD converged: %s", self.convergence_reason)

    def _increase_sampling_resolution(self) -> bool:
        if self.config.spawning.spawn_scheme == "voronoi":
            old_clusters = self.config.spawning.voronoi_clusters
            max_clusters = self.config.spawning.voronoi_max_clusters
            if old_clusters >= max_clusters:
                return False
            new_clusters = min(int(old_clusters * 1.5), max_clusters)
            if new_clusters <= old_clusters:
                new_clusters = min(old_clusters + 1, max_clusters)
            self.config.spawning.voronoi_clusters = new_clusters
            if hasattr(self.spawner, "n_clusters"):
                self.spawner.n_clusters = new_clusters
            logging.info(
                "Increased voronoi clusters from %d to %d",
                old_clusters,
                new_clusters,
            )
            return new_clusters != old_clusters

        old_bins = list(self.config.n_bins)
        max_bins = self.config.spawning.resolution_max_bins
        new_bins = [min(int(b * 1.15), max_bins) for b in old_bins]
        new_bins = [
            min(old + 1, max_bins) if new <= old and old < max_bins else new
            for old, new in zip(old_bins, new_bins, strict=False)
        ]
        if new_bins == old_bins:
            return False
        self.config.n_bins = new_bins
        if hasattr(self.spawner, "n_bins"):
            self.spawner.n_bins = new_bins
        logging.info("Increased regular bins from %s to %s", old_bins, new_bins)
        return True

    def _resolution_at_max(self) -> bool:
        if self.config.spawning.spawn_scheme == "voronoi":
            return (
                self.config.spawning.voronoi_clusters
                >= self.config.spawning.voronoi_max_clusters
            )
        return all(
            int(value) >= self.config.spawning.resolution_max_bins
            for value in self.config.n_bins
        )

    @staticmethod
    def _require_file(errors: list[str], label: str, value: str) -> None:
        path = Path(value)
        if not path.exists():
            errors.append(f"{label} does not exist: {path}")
        elif not path.is_file():
            errors.append(f"{label} is not a file: {path}")

    @classmethod
    def _require_optional_file(
        cls, errors: list[str], label: str, value: str | None
    ) -> None:
        if value:
            cls._require_file(errors, label, value)

    def _validate_project_file(self, errors: list[str]) -> None:
        project_file = self.config.system.project_file
        if not project_file:
            return

        path = Path(project_file)
        self._require_file(errors, "system.project_file", project_file)
        if not path.exists() or not path.is_file():
            return

        spec = importlib.util.spec_from_file_location("trails_md_project_check", path)
        if spec is None or spec.loader is None:
            errors.append(f"system.project_file could not be imported: {path}")
            return

        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as exc:
            errors.append(f"system.project_file import failed for {path}: {exc}")
            return

        if not callable(getattr(module, "extract_cvs", None)):
            errors.append(
                "system.project_file must define callable extract_cvs("
                "trajectories, top_file, conf_file)."
            )

    def _validate_engine_preflight(self, errors: list[str]) -> None:
        engine_name = self.config.engine.md_engine
        if engine_name == "amber":
            self._require_executable(
                errors,
                "engine.amber_executable",
                self.config.engine.amber_executable,
                "Activate an Amber environment or set engine.amber_executable "
                "to a full path such as /path/to/pmemd.cuda, pmemd, or sander.",
            )
            self._require_optional_file(
                errors,
                "engine.amber_input_file",
                self.config.engine.amber_input_file,
            )
            self._validate_amber_template(errors)
        elif engine_name == "gromacs":
            self._require_executable(
                errors,
                "engine.gromacs_executable",
                self.config.engine.gromacs_executable,
                "Load GROMACS or set engine.gromacs_executable to the full gmx path.",
            )
            include_dir = self.config.engine.gromacs_include_dir
            if include_dir and not Path(include_dir).is_dir():
                errors.append(
                    f"engine.gromacs_include_dir is not a directory: {include_dir}"
                )
            self._validate_gromacs_mdrun_options(errors)
        elif engine_name != "openmm":
            errors.append(
                "engine.md_engine must be one of: openmm, amber, gromacs. "
                f"Got: {engine_name!r}"
            )

    @staticmethod
    def _require_executable(
        errors: list[str], label: str, executable: str, hint: str
    ) -> None:
        if not shutil.which(executable):
            errors.append(f"{label} not found on PATH: {executable!r}. {hint}")

    def _validate_amber_template(self, errors: list[str]) -> None:
        template = self.config.engine.amber_input_file
        if not template or not Path(template).exists():
            return

        try:
            Path(template).read_text().format(
                steps=self.config.spawning.step,
                dt=self.config.engine.dt,
                temp=self.config.engine.temperature,
                stride=self.config.spawning.stride,
                ntp=1 if self.config.engine.npt else 0,
                ntb=2 if self.config.engine.npt else 1,
            )
        except KeyError as exc:
            errors.append(
                f"engine.amber_input_file has unsupported placeholder {exc!s}: "
                f"{template}"
            )
        except Exception as exc:
            errors.append(f"engine.amber_input_file could not be rendered: {exc}")

    def _validate_gromacs_mdrun_options(self, errors: list[str]) -> None:
        choices = {
            "gromacs_mdrun_nb": {"auto", "cpu", "gpu"},
            "gromacs_mdrun_pme": {"auto", "cpu", "gpu"},
            "gromacs_mdrun_update": {"auto", "cpu", "gpu"},
            "gromacs_mdrun_bonded": {"auto", "cpu", "gpu"},
            "gromacs_mdrun_pin": {"auto", "on", "off"},
        }
        for field, allowed in choices.items():
            value = getattr(self.config.engine, field)
            if value is not None and value not in allowed:
                errors.append(
                    f"engine.{field} must be one of {sorted(allowed)}; got {value!r}."
                )
        if self.config.engine.gromacs_mdrun_ntmpi <= 0:
            errors.append("engine.gromacs_mdrun_ntmpi must be greater than 0.")
        ntomp = self.config.engine.gromacs_mdrun_ntomp
        if ntomp is not None and ntomp <= 0:
            errors.append("engine.gromacs_mdrun_ntomp must be greater than 0.")

    def _validate_sampling_settings(self, errors: list[str]) -> None:
        if self.config.spawning.walker <= 0:
            errors.append("spawning.walker must be greater than 0.")
        if self.config.spawning.step <= 0:
            errors.append("spawning.step must be greater than 0.")
        if self.config.spawning.stride <= 0:
            errors.append("spawning.stride must be greater than 0.")
        if self.config.spawning.step < self.config.spawning.stride:
            errors.append("spawning.step must be greater than or equal to stride.")
        if self.config.spawning.resolution_check_patience < 0:
            errors.append("spawning.resolution_check_patience must be non-negative.")
        if self.config.spawning.convergence_patience < 0:
            errors.append("spawning.convergence_patience must be non-negative.")
        if self.config.spawning.resolution_max_bins <= 0:
            errors.append("spawning.resolution_max_bins must be greater than 0.")
        if self.config.spawning.voronoi_max_clusters <= 0:
            errors.append("spawning.voronoi_max_clusters must be greater than 0.")

        if self.config.space_mode == "fixed":
            dim = len(self.config.n_bins)
            if self.config.min_values is not None and len(self.config.min_values) != dim:
                errors.append("min_values length must match n_bins for fixed space.")
            if self.config.max_values is not None and len(self.config.max_values) != dim:
                errors.append("max_values length must match n_bins for fixed space.")

    def _ensure_output_log_header(self) -> None:
        if self.output_log.exists():
            return

        header = [
            "# Trails-MD run log",
            f"# outdir={self.outdir}",
            f"# md_engine={self.config.engine.md_engine}",
            f"# space_mode={self.config.space_mode}",
            f"# spawn_scheme={self.config.spawning.spawn_scheme}",
            f"# spawn_type={self.config.spawning.spawn_type}",
            f"# walker={self.config.spawning.walker}",
            f"# step={self.config.spawning.step}",
            f"# stride={self.config.spawning.stride}",
            f"# n_bins={json.dumps(self.config.n_bins)}",
            f"# min_values={json.dumps(self.config.min_values)}",
            f"# max_values={json.dumps(self.config.max_values)}",
            (
                "iteration\trunner_s\tanalysis_s\ttotal_s\t"
                "successful_walkers\tfailed_walkers\tframes_this_iteration\t"
                "cumulative_frames\toccupied_bins\ttotal_bins\texploration_fraction\t"
                "spawn_indices\tcvs_file\ttrajectory_dir\tcheckpoint_dir"
            ),
        ]
        self.output_log.write_text("\n".join(header) + "\n", encoding="utf-8")

    def _append_iteration_log(
        self,
        *,
        iteration: int,
        runner_time: float,
        other_time: float,
        success: list[bool],
        points: Any,
        occupied_bins: int | None,
        total_bins: int | None,
        cumulative_frames: int | None,
        spawn_indices: list[int],
        trajectories: list[str],
    ) -> None:
        points_array = np.asarray(points)
        frames_this_iteration = len(points_array)
        successful_walkers = sum(1 for value in success if value)
        failed_walkers = len(success) - successful_walkers
        exploration_fraction = (
            occupied_bins / total_bins
            if occupied_bins is not None and total_bins not in (None, 0)
            else None
        )
        iter_dir = self.outdir / f"iter_{iteration}"
        checkpoint_dir = self.outdir / "checkpoints" / f"iter_{iteration}"
        row = [
            iteration,
            f"{runner_time:.6f}",
            f"{other_time:.6f}",
            f"{runner_time + other_time:.6f}",
            successful_walkers,
            failed_walkers,
            frames_this_iteration,
            _log_value(cumulative_frames),
            _log_value(occupied_bins),
            _log_value(total_bins),
            f"{exploration_fraction:.8f}" if exploration_fraction is not None else "NA",
            json.dumps([int(index) for index in spawn_indices]),
            str(iter_dir / "cvs.npz"),
            str(iter_dir),
            str(checkpoint_dir) if checkpoint_dir.exists() else "NA",
        ]
        with self.output_log.open("a", encoding="utf-8") as handle:
            handle.write("\t".join(str(value) for value in row) + "\n")


def _log_value(value: Any) -> Any:
    return "NA" if value is None else value
