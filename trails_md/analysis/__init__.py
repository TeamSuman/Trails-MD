"""Post-hoc MSM analysis and plotting utilities.

``data`` holds matplotlib-free numerics (loading msm.npz/cvs.npz, free
energies); ``plots`` holds the matplotlib visualisations; ``riteweight``
reweights an adaptively-sampled ensemble towards its stationary distribution
without a lag-time or Markovianity assumption.
"""

from . import data
from .riteweight import RiteWeightResult, riteweight

__all__ = ["data", "riteweight", "RiteWeightResult"]
