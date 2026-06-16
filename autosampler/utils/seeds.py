import random
import numpy as np
import torch
import os

class SeedManager:
    """Manages random number generator seeding across all libraries to ensure bitwise reproducibility."""
    
    def __init__(self, seed: int):
        self.seed = seed

    def set_seed(self) -> None:
        """Initialize random seeds globally across all relevant backends."""
        # 1. Standard Python library
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        
        # 2. NumPy backend
        np.random.seed(self.seed)
        
        # 3. PyTorch (CPU & GPU)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
