import logging
import warnings
from pathlib import Path
import importlib.util
import json
import shutil
import sys

# Suppress common non-critical warnings from dependencies
warnings.filterwarnings(
    "ignore", message="Non-optimal GB parameters detected for GB model HCT"
)
warnings.filterwarnings("ignore", message="Reload offsets from trajectory")
warnings.filterwarnings("ignore", message=".*Reader has no dt information.*")
from typing import Any, Dict, List

import numpy as np
from pydantic import ValidationError

from autosampler.checkpoints.manager import CheckpointManager
from autosampler.config import AutoSamplerConfig
from autosampler.engines.amber import amber_trajectory_suffix
from autosampler.engines.base import EngineFactory
from autosampler.paths import build_frame_records, map_global_frame
from autosampler.reporting import IterationReporter
from autosampler.spaces import AdaptiveSpaceModel, FeatureExtractor
from autosampler.spaces.registry import is_adaptive_space
from autosampler.spawners.base import SpawnerFactory
from autosampler.utils.seeds import SeedManager
from autosampler.workflows.parallel import run_iteration_parallel


class AutoSamplerCore:
    """Main orchestrator for the AutoSampler framework."""

    def __init__(self, config_dict: Dict[str, Any]):
        try:
            self.config = AutoSamplerConfig(**config_dict)
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
        )

        # State variables
        self.iteration = 0
        self.history = {}
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
            from autosampler.spaces.feature_selection import FeatureSelector

            self.feature_selector = FeatureSelector(
                lagtime=fs_cfg.lagtime,
                method=fs_cfg.method,
                max_features=fs_cfg.max_features,
                dim=fs_cfg.dim,
                min_gain=fs_cfg.min_gain,
            )

        # Adaptive CV-retraining policy (VAMP-2 driven when configured).
        from autosampler.spaces.retraining import RetrainController

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
            from autosampler.msm import MSMEstimator, build_monitor_from_config

            self.msm_estimator = MSMEstimator(
                lagtime=msm_cfg.lagtime,
                n_microstates=msm_cfg.n_microstates,
                cluster_method=msm_cfg.cluster_method,
                estimator=msm_cfg.estimator,
                n_metastable=msm_cfg.n_metastable,
                n_timescales=msm_cfg.n_timescales,
                lagtimes=msm_cfg.lagtimes,
                n_bayesian_samples=msm_cfg.n_bayesian_samples,
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
        return kwargs

    def restore_checkpoint(self, iteration: int):
        """Restore sampler state from a saved checkpoint and resume at the next iteration."""
        restored_model = self.space_model
        if restored_model is None and is_adaptive_space(self.config.space_mode):
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

    def resume_walkers(self) -> List[Any]:
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

        trajectories = self._sampling_trajectories([])
        if not trajectories:
            raise RuntimeError(
                f"Checkpoint history entry {latest_iteration} has no trajectories."
            )

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

    def _traj_suffix(self) -> str:
        if self.config.engine.md_engine == "amber":
            return amber_trajectory_suffix(
                self.config.engine.amber_trajectory_format,
                self.config.engine.amber_executable,
            )
        return "xtc"

    def run_iteration(self, walkers: List[Any]):
        """Run a single adaptive sampling iteration."""

        # 1. Run production MD
        import time

        runner_start_time = time.time()

        engine_kwargs = {
            k: v for k, v in self.config.engine.model_dump().items() if k != "md_engine"
        }
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
        if not all(results):
            failed = results.count(False)
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
            raise RuntimeError(
                f"{failed} walker(s) failed during iteration {self.iteration}; "
                "stopping before CV extraction." + detail
            )
        other_start_time = time.time()
        if len(self.walker_parents) != len(walkers):
            self.walker_parents = [None for _ in walkers]

        # 2. Extract and project coordinates
        trajectories = [
            str(
                self.outdir
                / f"iter_{self.iteration}"
                / f"iteration_{self.iteration}_{idx}.{self._traj_suffix()}"
            )
            for idx in range(len(walkers))
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
                    # Optimize: Cap memory to prevent O(N) deep learning slowdown
                    max_frames = self.config.max_adaptive_memory_frames
                    frames_per_iter = len(features)
                    max_iters = max(5, max_frames // frames_per_iter)

                    if len(self.feature_memory) > max_iters:
                        import random

                        # Keep initial state to anchor global space, random sample the rest
                        sampled_memory = [self.feature_memory[0]]
                        sampled_memory.extend(
                            random.sample(self.feature_memory[1:], max_iters - 1)
                        )
                        fit_features = np.vstack(sampled_memory)
                        total_walkers = len(sampled_memory) * len(walkers)
                    else:
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
        spawn_indices = self.spawner.sample(
            points, self.config.spawning.walker, history=self.history
        )
        sampling_trajectories = self._sampling_trajectories(trajectories)
        sampling_frame_records = self._sampling_frame_records(current_frame_records)
        next_walker_parents = [
            map_global_frame(sampling_frame_records, index)["key"]
            for index in spawn_indices
        ]
        next_walkers = feature_extractor.extract_positions_by_indices(
            sampling_trajectories, spawn_indices
        )

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

        # 3. Save checkpoint (if frequency matches)
        if (
            self.config.checkpoint_freq > 0
            and self.iteration % self.config.checkpoint_freq == 0
        ):
            self.checkpoint_manager.save(
                iteration=self.iteration,
                space_model=self.space_model,
                scaler=self.scaler,
                bin_state=self.bin_state,
                history=self.history,
                sampler_state=self._checkpoint_state(),
            )

        self.iteration += 1
        self.walker_parents = next_walker_parents
        other_time = time.time() - other_start_time

        # Compute bin occupancy for logging
        occupied_bins = None
        total_bins = None
        cumulative_frames = None
        try:
            from autosampler.binning.spatial import RegularBinner

            is_adaptive = is_adaptive_space(self.config.space_mode)
            binner = RegularBinner(
                n_bins=self.config.n_bins,
                min_values=None if is_adaptive else self.config.min_values,
                max_values=None if is_adaptive else self.config.max_values,
            )
            # Gather all historical projections
            cumulative_points = []
            for entry in self.history.values():
                if isinstance(entry, dict) and entry.get("projection") is not None:
                    cumulative_points.append(np.asarray(entry["projection"]))
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

            # Adaptive resolution check
            self._update_resolution_and_convergence(occupied_bins)

        except Exception as e:
            bin_occupancy_str = "N/A"
            logging.debug(f"Failed to compute bin occupancy: {e}")

        # MSM estimation + MSM-based convergence (opt-in).
        self._maybe_build_msm()

        current_iteration = self.iteration - 1
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

    def _sampling_trajectories(self, current_trajectories: List[str]) -> List[str]:
        trajectories: List[str] = []
        for iteration in sorted(self.history):
            entry = self.history[iteration]
            if not isinstance(entry, dict) or entry.get("projection") is None:
                continue
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
        self, current_frame_records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for iteration in sorted(self.history):
            entry = self.history[iteration]
            if not isinstance(entry, dict) or entry.get("projection") is None:
                continue
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
    ) -> List[str]:
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

    def _checkpoint_state(self) -> Dict[str, Any]:
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
        }

    def _restore_sampler_state(self, state: Dict[str, Any] | None) -> None:
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

    @staticmethod
    def _validate_trajectory_files(trajectories: List[str]) -> None:
        """Ensure each expected trajectory exists and is non-empty before reading.

        A walker can report success yet leave a missing or truncated file (disk
        full, killed writer); catching it here gives a clear error instead of an
        opaque downstream parse failure.
        """
        bad: list[str] = []
        for path in trajectories:
            p = Path(path)
            if not p.is_file():
                bad.append(f"missing: {path}")
            elif p.stat().st_size == 0:
                bad.append(f"empty: {path}")
        if bad:
            joined = "\n  - ".join(bad)
            raise RuntimeError(
                "Trajectory files are not usable for CV extraction:\n  - " + joined
            )

    def _fs_apply(self, features: np.ndarray) -> np.ndarray:
        """Restrict features to the VAMP-selected columns (no-op if disabled)."""
        if self.feature_selection_indices is None:
            return features
        return np.asarray(features)[:, self.feature_selection_indices]

    def _extract_feature_type(
        self, feature_extractor: FeatureExtractor, trajectories: List[str], ftype: str
    ) -> np.ndarray:
        if ftype == "phi_psi":
            return feature_extractor.extract_aib9_phi_psi(trajectories)
        if ftype == "fitted_coords":
            return feature_extractor.extract_fitted_coords(trajectories)
        return feature_extractor.extract_pairwise_distances(trajectories)

    def _extract_adaptive_features(
        self, feature_extractor: FeatureExtractor, trajectories: List[str]
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
        from autosampler.spaces.feature_selection import rank_candidates

        n_frames = self.config.spawning.step // self.config.spawning.stride
        extracted: Dict[str, np.ndarray] = {}
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
        from autosampler.spaces.feature_selection import vamp2_score

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

    def _extract_physical_cvs(self, trajectories: List[str]) -> np.ndarray:
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

    def _collect_msm_trajectories(self) -> List[Any]:
        """Split cumulative history projections into continuous per-walker trajectories.

        Each short walker is one continuous trajectory; transition counts are
        pooled across all of them by the estimator. Single-frame segments are
        dropped because they carry no transitions.
        """
        frames_per_walker = self.config.spawning.step // self.config.spawning.stride
        trajs: List[Any] = []
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
            if frames_per_walker <= 0:
                if len(projection) > 1:
                    trajs.append(projection)
                continue
            n_walkers = int(np.ceil(len(projection) / frames_per_walker))
            for walker in range(n_walkers):
                segment = projection[
                    walker * frames_per_walker : (walker + 1) * frames_per_walker
                ]
                if len(segment) > 1:
                    trajs.append(segment)
        return trajs

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
        if converged and not self.converged:
            self.converged = True
            self.convergence_reason = self.msm_monitor.reason

    def _save_msm_result(self, iteration: int, result: Any) -> None:
        try:
            vamp2 = np.nan if result.vamp2_score is None else result.vamp2_score
            np.savez_compressed(
                self.outdir / f"iter_{iteration}" / "msm.npz",
                lagtime=np.asarray(result.lagtime),
                timescales=np.asarray(result.timescales, dtype=float),
                stationary_distribution=np.asarray(
                    result.stationary_distribution, dtype=float
                ),
                transition_matrix=np.asarray(result.transition_matrix, dtype=float),
                cluster_centers=np.asarray(result.cluster_centers, dtype=float),
                vamp2_score=np.asarray([vamp2], dtype=float),
            )
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
            logging.info("AutoSampler converged: %s", self.convergence_reason)

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
            for old, new in zip(old_bins, new_bins)
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

        spec = importlib.util.spec_from_file_location("autosampler_project_check", path)
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
            "# AutoSampler run log",
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
        success: List[bool],
        points: Any,
        occupied_bins: int | None,
        total_bins: int | None,
        cumulative_frames: int | None,
        spawn_indices: List[int],
        trajectories: List[str],
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
