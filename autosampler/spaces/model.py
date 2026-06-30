import torch
import numpy as np
from typing import Any

from deeptime.decomposition.deep import TVAE
from deeptime.util.data import TrajectoryDataset
from torch.utils.data import DataLoader
from .scalers import TrajectoryScaler
from .tvae import TVAEBottleneckEncoder, TVAEBottleneckDecoder


class AdaptiveSpaceModel:
    """Manages the training and projection of the ML dimensionality reduction space."""

    _DEFAULTS = {
        "lagtime": 5,
        "latent_dim": 2,
        "epochs": 50,
        "batch_size": "auto",
        "learning_rate": 5e-4,
        "encoder_hidden_dims": [256, 128],
        "decoder_hidden_dims": [128, 256],
        "dropout_rate": 0.1,
        "deep_tica_hidden_dims": [256, 128],
        "spib_n_states": 10,
        "spib_beta": 1e-3,
        "seed": 0,
    }

    def __init__(
        self,
        space_mode: str = "tvae",
        lagtime: int = 5,
        latent_dim: int = 2,
        epochs: int = 50,
        batch_size: int | str = "auto",
        learning_rate: float = 5e-4,
        encoder_hidden_dims: list[int] | None = None,
        decoder_hidden_dims: list[int] | None = None,
        dropout_rate: float = 0.1,
        deep_tica_hidden_dims: list[int] | None = None,
        spib_n_states: int = 10,
        spib_beta: float = 1e-3,
        seed: int = 0,
        **_: Any,
    ):
        self.type = space_mode
        self.seed = int(seed)
        self.lagtime = int(lagtime)
        self.latent_dim = int(latent_dim)
        self.epochs = int(epochs)
        self.batch_size = batch_size
        self.learning_rate = float(learning_rate)
        self.encoder_hidden_dims = list(encoder_hidden_dims or [256, 128])
        self.decoder_hidden_dims = list(decoder_hidden_dims or [128, 256])
        self.dropout_rate = float(dropout_rate)
        self.deep_tica_hidden_dims = list(deep_tica_hidden_dims or [256, 128])
        self.spib_n_states = int(spib_n_states)
        self.spib_beta = float(spib_beta)
        self.scaler = TrajectoryScaler("minmax")
        self.model = None
        self.fitted = None  # PyTorch model used for projection
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __setstate__(self, state: dict) -> None:
        state = dict(state)
        # Backwards compatibility: pre-2.x checkpoints stored the projection
        # network under the misspelled attribute ``fited``.
        if "fited" in state and "fitted" not in state:
            state["fitted"] = state.pop("fited")
        self.__dict__.update(state)
        self.ensure_config_defaults()

    def ensure_config_defaults(self, **overrides: Any) -> None:
        defaults = dict(self._DEFAULTS)
        defaults.update(
            {key: value for key, value in overrides.items() if key in defaults}
        )
        if not hasattr(self, "type"):
            self.type = overrides.get("space_mode", "tvae")
        for key, value in defaults.items():
            if not hasattr(self, key):
                setattr(self, key, list(value) if isinstance(value, list) else value)
        if not hasattr(self, "device"):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _batch_size(self, n_walkers: int, walker_length: int) -> int:
        self.ensure_config_defaults()
        if self.batch_size == "auto":
            return n_walkers * walker_length
        return int(self.batch_size)

    @staticmethod
    def _torch_features(array: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(array, dtype=np.float32)

    def fit(self, features: np.ndarray, walker_length: int, n_walkers: int):
        """Fit the ML model on new trajectory features.
        features: shape (total_frames, n_features)
        """
        self.ensure_config_defaults()
        input_size = features.shape[-1]

        # Reseed the torch RNG from the configured seed before every fit so that a
        # CV retrained at iteration N is reproducible regardless of the RNG draws
        # made in between (network init, DataLoader shuffling, etc.).
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # Fail fast with an actionable message if an optional backend is missing.
        from .registry import ensure_available, is_adaptive_space

        if is_adaptive_space(self.type):
            ensure_available(self.type)

        # We need continuous trajectories for deeptime TVAE, so we split them by walker
        # features array should be ordered by walker, then time

        # Scale the features
        self.scaler.fit(features)
        scaled_features = self.scaler.transform(features)

        if self.type == "tvae":
            scaled_features = self._torch_features(scaled_features)
            # Reshape to a list of separate trajectory arrays for deeptime.
            traj_list = [
                scaled_features[i * walker_length : (i + 1) * walker_length]
                for i in range(n_walkers)
            ]
            encoder = TVAEBottleneckEncoder(
                input_size,
                self.latent_dim,
                hidden_dims=self.encoder_hidden_dims,
                dropout_rate=self.dropout_rate,
            ).to(self.device)
            decoder = TVAEBottleneckDecoder(
                self.latent_dim,
                input_size,
                hidden_dims=self.decoder_hidden_dims,
                dropout_rate=self.dropout_rate,
            ).to(self.device)
            self.model = TVAE(encoder, decoder, learning_rate=self.learning_rate)

            # Prepare data
            dataset = TrajectoryDataset.from_trajectories(self.lagtime, traj_list)

            # Simple dataloader without validation split for brevity
            loader_train = DataLoader(
                dataset,
                batch_size=self._batch_size(n_walkers, walker_length),
                shuffle=False,
            )

            self.model.fit(loader_train, n_epochs=self.epochs)

            fitted = self.model.fetch_model().copy()
            self.fitted = fitted.encoder.eval().to('cpu')

        elif self.type == "tica":
            from deeptime.decomposition import TICA
            traj_list = [
                scaled_features[i * walker_length : (i + 1) * walker_length]
                for i in range(n_walkers)
            ]
            self.model = TICA(dim=self.latent_dim, lagtime=self.lagtime)
            self.model.fit(traj_list)

        elif self.type == "pca":
            from sklearn.decomposition import PCA
            self.model = PCA(n_components=self.latent_dim)
            self.model.fit(scaled_features)

        elif self.type == "deep-tica":
            scaled_features = self._torch_features(scaled_features)
            traj_list = [
                scaled_features[i * walker_length : (i + 1) * walker_length]
                for i in range(n_walkers)
            ]
            import lightning as pl
            from mlcolvar.cvs import DeepTICA
            from mlcolvar.utils.timelagged import create_timelagged_dataset
            from mlcolvar.data import DictModule, DictDataset

            data_dict = {}
            for traj in traj_list:
                X = torch.tensor(traj, dtype=torch.float32)
                t = torch.arange(len(traj)).float()
                ds = create_timelagged_dataset(X, t=t, lag_time=self.lagtime)

                for k in ds.keys:
                    if k not in data_dict:
                        data_dict[k] = []
                    data_dict[k].append(ds[k])

            dataset = DictDataset({k: torch.cat(v) for k, v in data_dict.items()})
            datamodule = DictModule(
                dataset, batch_size=self._batch_size(n_walkers, walker_length)
            )

            self.model = DeepTICA(
                layers=[input_size, *self.deep_tica_hidden_dims, self.latent_dim]
            )

            import logging
            import warnings

            logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

            trainer = pl.Trainer(
                max_epochs=self.epochs,
                enable_checkpointing=False,
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trainer.fit(self.model, datamodule)

        elif self.type == "vampnet":
            from deeptime.decomposition.deep import VAMPNet
            from deeptime.util.torch import MLP

            scaled_features = self._torch_features(scaled_features)
            traj_list = [
                scaled_features[i * walker_length : (i + 1) * walker_length]
                for i in range(n_walkers)
            ]
            lobe = MLP(
                units=[input_size, *self.encoder_hidden_dims, self.latent_dim],
                nonlinearity=torch.nn.SiLU,
            ).to(self.device)
            vampnet = VAMPNet(
                lobe=lobe, learning_rate=self.learning_rate, device=self.device
            )
            dataset = TrajectoryDataset.from_trajectories(self.lagtime, traj_list)
            loader_train = DataLoader(
                dataset,
                batch_size=self._batch_size(n_walkers, walker_length),
                shuffle=True,
            )
            self.model = vampnet.fit(loader_train, n_epochs=self.epochs).fetch_model()
            self.fitted = lobe.eval().to("cpu")

        elif self.type == "spib":
            from .spib import train_spib

            scaled_features = self._torch_features(scaled_features)
            traj_list = [
                scaled_features[i * walker_length : (i + 1) * walker_length]
                for i in range(n_walkers)
            ]
            self.fitted = train_spib(
                traj_list,
                lagtime=self.lagtime,
                latent_dim=self.latent_dim,
                hidden_dims=self.encoder_hidden_dims,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                batch_size=self._batch_size(n_walkers, walker_length),
                n_states=self.spib_n_states,
                beta=self.spib_beta,
                dropout=self.dropout_rate,
                device=self.device,
                seed=self.seed,
            )

        elif self.type == "deep-lda":
            # Deep-LDA is supervised: it requires per-frame state labels, so it
            # is intended for the targeted/labelled workflow (e.g. known
            # reactant/product basins) rather than fully autonomous exploration.
            raise NotImplementedError(
                "space_mode 'deep-lda' is supervised and needs per-frame state "
                "labels; it is registered for the labelled/targeted workflow but "
                "not wired into autonomous exploration. Use 'deep-tica', "
                "'vampnet', 'spib' or 'tvae' for unsupervised adaptive sampling."
            )

        else:
            raise ValueError(
                f"space_mode {self.type!r} has no training implementation."
            )

    def project(self, features: np.ndarray) -> np.ndarray:
        """Project scaled features into latent space."""
        self.ensure_config_defaults()
        scaled = self.scaler.transform(features)

        if self.type == "tvae":
            device = next(self.fitted.parameters()).device
            tensor = torch.as_tensor(
                self._torch_features(scaled), dtype=torch.float32, device=device
            )
            with torch.no_grad():
                projected = self.fitted(tensor)[0].detach().cpu().numpy()
        elif self.type == "vampnet":
            device = next(self.fitted.parameters()).device
            tensor = torch.as_tensor(
                self._torch_features(scaled), dtype=torch.float32, device=device
            )
            with torch.no_grad():
                projected = self.fitted(tensor).detach().cpu().numpy()
        elif self.type == "spib":
            device = next(self.fitted.parameters()).device
            tensor = torch.as_tensor(
                self._torch_features(scaled), dtype=torch.float32, device=device
            )
            with torch.no_grad():
                mean, _ = self.fitted(tensor)
                projected = mean.detach().cpu().numpy()
        elif self.type == "deep-tica":
            try:
                device = next(self.model.parameters()).device
            except (StopIteration, AttributeError):
                device = torch.device("cpu")
            tensor = torch.as_tensor(
                self._torch_features(scaled), dtype=torch.float32, device=device
            )
            with torch.no_grad():
                projected = self.model(tensor).detach().cpu().numpy()
        elif self.type == "pca":
            projected = self.model.transform(scaled)
        elif self.type == "tica":
            # TICA transform takes a list of trajs
            projected = self.model.transform([scaled])[0]
        else:
            projected = scaled  # fallback

        return projected
