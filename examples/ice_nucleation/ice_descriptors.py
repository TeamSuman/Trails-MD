"""Physics-based ice-nucleation / polymorph-selectivity descriptors.

Implements the two-component fixed CV from the ice-nucleation CV protocol
(docs/ice_nucleation_cv_protocol.md):

  Component 1 (nucleation progress): the size of the largest spatially
  connected cluster of ice-like water molecules, ``n_max``.

  Component 2 (polymorph selectivity): cubicity ``chi = n_Ic / (n_Ic + n_Ih)``
  from a CHILL+-style per-molecule classification (Nguyen & Molinero, J. Phys.
  Chem. B 2015, 119, 9369; DOI 10.1021/jp510289t).

Works on both the coarse-grained mW water model (one bead per molecule) and
all-atom water (oxygen-only selection), matching the auto-detection pattern
used by IceCoder's ``Scripts/soaper.py``.

CHILL+ implementation notes
----------------------------
Bond correlations use the l=3 Steinhardt spherical harmonics (c3 is reported
in the literature as the best discriminator between crystalline water
networks). Per-bond correlation for a neighbor pair (i, j) within the O-O
cutoff:

    c3(i, j) = Re[ sum_m q3m(i) * conj(q3m(j)) ] / (|q3(i)| * |q3(j)|)

Bonds are classified as *staggered* (c3 < -0.8) or *eclipsed*
(-0.35 <= c3 <= 0.25); values in between are ambiguous and not counted.
Per-molecule, with n_bonds the number of neighbors within the cutoff:

  - cubic ice:      n_bonds == 4 and n_staggered == 4
  - hexagonal ice:  n_bonds == 4 and n_staggered == 3 and n_eclipsed == 1
  - clathrate:      n_bonds == 4 and n_eclipsed == 4
  - interfacial ice (ice-like but not a clean 4-bond pattern above): treated
    as "ice-like" for clustering (n_max) but not counted toward chi, since
    chi only needs the two polymorphs of interest (Ih vs Ic).
  - everything else: liquid/interfacial water.

These thresholds are literature defaults. Per the protocol's Phase 0, they
must be (re-)calibrated per water model against reference liquid/Ih/Ic
ensembles before being trusted quantitatively -- this module exposes the raw
per-bond correlations and per-molecule bond counts precisely so that
calibration can be done without re-deriving the neighbor/harmonics machinery.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import freud
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ice_descriptors requires the 'freud-analysis' package: pip install freud-analysis"
    ) from exc


# Literature CHILL+ thresholds (Nguyen & Molinero 2015).
STAGGERED_MAX = -0.8
ECLIPSED_RANGE = (-0.35, 0.25)

# Water O-O first-neighbor-shell cutoff (Angstrom); appropriate for both mW
# and all-atom water where the first coordination shell sits near ~3.5 A.
DEFAULT_R_CUT = 3.5

# Molecule-level CHILL+ labels.
LABEL_LIQUID = 0
LABEL_INTERFACIAL_ICE = 1
LABEL_HEXAGONAL = 2
LABEL_CUBIC = 3
LABEL_CLATHRATE = 4

LABEL_NAMES = {
    LABEL_LIQUID: "liquid",
    LABEL_INTERFACIAL_ICE: "interfacial_ice",
    LABEL_HEXAGONAL: "hexagonal",
    LABEL_CUBIC: "cubic",
    LABEL_CLATHRATE: "clathrate",
}


def _freud_box(box_dimensions: np.ndarray) -> "freud.box.Box":
    """Build a freud Box from MDAnalysis-style [Lx,Ly,Lz,alpha,beta,gamma]."""
    from MDAnalysis.lib.mdamath import triclinic_vectors

    matrix = triclinic_vectors(np.asarray(box_dimensions, dtype=np.float64))
    return freud.box.Box.from_matrix(matrix)


def _bond_neighbor_list(fbox: "freud.box.Box", positions: np.ndarray, r_cut: float):
    aq = freud.locality.AABBQuery(fbox, positions)
    nlist = aq.query(positions, {"r_max": r_cut, "exclude_ii": True}).toNeighborList()
    return nlist


@dataclass
class ChillPlusResult:
    labels: np.ndarray  # (N,) int, one of LABEL_*
    n_bonds: np.ndarray  # (N,) int, neighbors within r_cut
    n_staggered: np.ndarray  # (N,) int
    n_eclipsed: np.ndarray  # (N,) int
    ice_like: np.ndarray  # (N,) bool, used for n_max clustering


def chill_plus_labels(
    positions: np.ndarray,
    box_dimensions: np.ndarray,
    r_cut: float = DEFAULT_R_CUT,
    l: int = 3,
) -> ChillPlusResult:
    """Classify each water molecule's local environment via CHILL+.

    Args:
        positions: (N, 3) oxygen (or mW bead) coordinates, Angstrom.
        box_dimensions: MDAnalysis-style [Lx, Ly, Lz, alpha, beta, gamma].
        r_cut: O-O neighbor cutoff, Angstrom.
        l: spherical harmonic degree (CHILL+ uses l=3).

    Returns:
        ChillPlusResult with per-molecule labels and diagnostic bond counts.
    """
    positions = np.asarray(positions, dtype=np.float64)
    n = positions.shape[0]
    fbox = _freud_box(box_dimensions)

    steinhardt = freud.order.Steinhardt(l=l)
    steinhardt.compute((fbox, positions), neighbors={"r_max": r_cut, "exclude_ii": True})
    qlm = steinhardt.particle_harmonics  # (N, 2l+1), complex
    norm = np.sqrt(np.sum(np.abs(qlm) ** 2, axis=1))  # (N,)

    nlist = _bond_neighbor_list(fbox, positions, r_cut)
    i_idx = nlist.point_indices
    j_idx = nlist.query_point_indices

    denom = norm[i_idx] * norm[j_idx]
    valid = denom > 1e-12
    c3 = np.full(i_idx.shape[0], np.nan, dtype=np.float64)
    c3[valid] = np.real(
        np.sum(qlm[i_idx[valid]] * np.conj(qlm[j_idx[valid]]), axis=1)
    ) / denom[valid]

    staggered_bond = c3 < STAGGERED_MAX
    eclipsed_bond = (c3 >= ECLIPSED_RANGE[0]) & (c3 <= ECLIPSED_RANGE[1])

    n_bonds = np.bincount(i_idx, minlength=n)
    n_staggered = np.bincount(i_idx, weights=staggered_bond.astype(np.float64), minlength=n)
    n_eclipsed = np.bincount(i_idx, weights=eclipsed_bond.astype(np.float64), minlength=n)
    n_staggered = np.round(n_staggered).astype(int)
    n_eclipsed = np.round(n_eclipsed).astype(int)

    labels = np.full(n, LABEL_LIQUID, dtype=np.int64)

    four_bonds = n_bonds == 4
    is_cubic = four_bonds & (n_staggered == 4)
    is_hex = four_bonds & (n_staggered == 3) & (n_eclipsed == 1)
    is_clathrate = four_bonds & (n_eclipsed == 4)
    # Ice-like but not a clean hex/cubic/clathrate pattern: still counts
    # toward the nucleus for n_max, not toward chi. Restricted to molecules
    # with the tetrahedral (4-neighbor) coordination expected of a hydrogen-
    # bonded network -- without this restriction, high-coordination random
    # packings (e.g. dense liquid/gas configurations) produce false-positive
    # "ice-like" bond counts purely from having many neighbors, not from
    # genuine local order (confirmed via a randomized-positions control).
    is_interfacial_ice = four_bonds & ~is_cubic & ~is_hex & ~is_clathrate & (
        (n_staggered + n_eclipsed) >= 3
    )

    labels[is_interfacial_ice] = LABEL_INTERFACIAL_ICE
    labels[is_hex] = LABEL_HEXAGONAL
    labels[is_cubic] = LABEL_CUBIC
    labels[is_clathrate] = LABEL_CLATHRATE

    ice_like = labels != LABEL_LIQUID

    return ChillPlusResult(
        labels=labels,
        n_bonds=n_bonds,
        n_staggered=n_staggered,
        n_eclipsed=n_eclipsed,
        ice_like=ice_like,
    )


def largest_ice_cluster(
    positions: np.ndarray,
    box_dimensions: np.ndarray,
    ice_like: np.ndarray,
    r_cut: float = DEFAULT_R_CUT,
) -> tuple[int, np.ndarray]:
    """Size of the largest spatially connected cluster of ice-like molecules.

    Args:
        positions: (N, 3) coordinates, Angstrom.
        box_dimensions: MDAnalysis-style 6-vector.
        ice_like: (N,) boolean mask (e.g. ``ChillPlusResult.ice_like``).
        r_cut: clustering distance cutoff, Angstrom.

    Returns:
        (n_max, cluster_size_per_molecule) where cluster_size_per_molecule is
        0 for non-ice-like molecules and the size of their cluster otherwise.
    """
    positions = np.asarray(positions, dtype=np.float64)
    ice_like = np.asarray(ice_like, dtype=bool)
    n = positions.shape[0]
    cluster_size = np.zeros(n, dtype=np.int64)

    n_ice = int(ice_like.sum())
    if n_ice == 0:
        return 0, cluster_size

    fbox = _freud_box(box_dimensions)
    sub_positions = positions[ice_like]

    cl = freud.cluster.Cluster()
    cl.compute((fbox, sub_positions), neighbors={"r_max": r_cut})
    idx = cl.cluster_idx
    sizes = np.bincount(idx)
    n_max = int(sizes.max()) if len(sizes) else 0

    sub_cluster_size = sizes[idx]
    cluster_size[ice_like] = sub_cluster_size
    return n_max, cluster_size


def cubicity(labels: np.ndarray) -> tuple[float, int, int]:
    """chi = n_Ic / (n_Ic + n_Ih); NaN if no hex/cubic molecules present."""
    n_ic = int(np.sum(labels == LABEL_CUBIC))
    n_ih = int(np.sum(labels == LABEL_HEXAGONAL))
    total = n_ic + n_ih
    chi = float(n_ic) / total if total > 0 else float("nan")
    return chi, n_ic, n_ih


@dataclass
class IceCVResult:
    n_max: int
    chi: float
    n_ih: int
    n_ic: int
    n_interfacial_ice: int
    n_liquid: int
    labels: np.ndarray
    cluster_size: np.ndarray


def ice_cv(
    positions: np.ndarray,
    box_dimensions: np.ndarray,
    r_cut: float = DEFAULT_R_CUT,
) -> IceCVResult:
    """Compute the full fixed ice CV [n_max, chi] plus diagnostics for one frame."""
    chill = chill_plus_labels(positions, box_dimensions, r_cut=r_cut)
    n_max, cluster_size = largest_ice_cluster(
        positions, box_dimensions, chill.ice_like, r_cut=r_cut
    )
    chi, n_ic, n_ih = cubicity(chill.labels)
    n_interfacial_ice = int(np.sum(chill.labels == LABEL_INTERFACIAL_ICE))
    n_liquid = int(np.sum(chill.labels == LABEL_LIQUID))
    return IceCVResult(
        n_max=n_max,
        chi=chi,
        n_ih=n_ih,
        n_ic=n_ic,
        n_interfacial_ice=n_interfacial_ice,
        n_liquid=n_liquid,
        labels=chill.labels,
        cluster_size=cluster_size,
    )


def select_water_oxygens(universe):
    """Auto-detect and select the water-oxygen atom group (mW or all-atom).

    Mirrors the auto-detection in IceCoder/Scripts/soaper.py: a coarse-grained
    mW system has one bead per molecule (already "oxygens"); an all-atom
    system needs an explicit oxygen selection.
    """
    for selection in ("name OW", "name O and resname SOL HOH TIP3 TIP4 WAT"):
        group = universe.select_atoms(selection)
        if len(group) > 0:
            return group
    # Single-bead water (e.g. mW): every atom is one molecule.
    return universe.atoms
