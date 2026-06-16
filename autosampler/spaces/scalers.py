from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import numpy as np

class TrajectoryScaler:
    """Scales pairwise feature vectors while preserving shape mappings."""
    
    def __init__(self, scaler_type: str = "minmax"):
        if scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")

    def fit(self, features: np.ndarray) -> None:
        """Expects shape (n_walkers, walker_length, n_features) or (n_samples, n_features)."""
        if features.ndim > 2:
            flat_feats = features.reshape(-1, features.shape[-1])
        else:
            flat_feats = features
        self.scaler.fit(flat_feats)

    def transform(self, features: np.ndarray) -> np.ndarray:
        orig_shape = features.shape
        if features.ndim > 2:
            flat_feats = features.reshape(-1, orig_shape[-1])
        else:
            flat_feats = features
        scaled = self.scaler.transform(flat_feats)
        return scaled.reshape(orig_shape)
