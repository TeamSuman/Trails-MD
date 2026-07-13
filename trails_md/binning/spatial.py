from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BinTable:
    ids: list[Any]
    centers: np.ndarray
    populations: np.ndarray
    populated_data: list[list[int]]
    area: np.ndarray | None = None
    target_closeness: np.ndarray | None = None

    @property
    def occupied_indices(self) -> np.ndarray:
        return np.flatnonzero(self.populations > 0)


def padded_bounds(points: np.ndarray, padding: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = maxs - mins
    pad = padding * np.where(span > 0, span, 1.0)
    return mins - pad, maxs + pad



def bucket_frames(cell_idx: np.ndarray, nbin) -> tuple[np.ndarray, list]:
    """Group frame indices by grid cell, vectorized.

    ``cell_idx`` is an ``(n_frames, n_axes)`` integer array of per-axis bin indices.
    Returns ``(populations, populated_data)`` in row-major (``np.ndindex``) order.

    The obvious implementation is a Python loop over frames, which is O(n_frames)
    interpreted work re-run on the *whole cumulative history* every iteration --
    the dominant analysis cost in long campaigns. Sorting the flat cell index
    instead makes the per-frame work numpy-level and leaves only a loop over
    (few) bins.
    """
    nbin = [int(b) for b in nbin]
    n_cells = int(np.prod(nbin))
    if len(cell_idx) == 0:
        return np.zeros(n_cells, dtype=int), [[] for _ in range(n_cells)]

    flat = np.ravel_multi_index(tuple(cell_idx.T), tuple(nbin))
    populations = np.bincount(flat, minlength=n_cells)

    order = np.argsort(flat, kind="stable")
    bounds = np.searchsorted(flat[order], np.arange(n_cells + 1))
    populated_data = [order[bounds[i]:bounds[i + 1]] for i in range(n_cells)]
    return populations, populated_data

class RegularBinner:
    """Uniform grid binning for projected CV points."""

    def __init__(
        self,
        n_bins: list[int] | tuple[int, ...],
        min_values: list[float] | None = None,
        max_values: list[float] | None = None,
        target: list[float] | None = None,
        padding: float = 0.1,
    ):
        self.n_bins = np.asarray(n_bins, dtype=int)
        if np.any(self.n_bins <= 0):
            raise ValueError("n_bins values must be positive.")
        self.min_values = (
            None if min_values is None else np.asarray(min_values, dtype=float)
        )
        self.max_values = (
            None if max_values is None else np.asarray(max_values, dtype=float)
        )
        self.target = None if target is None else np.asarray(target, dtype=float)
        self.padding = padding

    def fit(self, points: np.ndarray) -> BinTable:
        points = np.asarray(points, dtype=float)
        if points.ndim != 2:
            raise ValueError("RegularBinner expects a 2D array.")
        if len(self.n_bins) != points.shape[1]:
            raise ValueError("n_bins dimensionality must match projected points.")

        if self.min_values is None or self.max_values is None:
            mins, maxs = padded_bounds(points, self.padding)
            self.min_values = mins if self.min_values is None else self.min_values
            self.max_values = maxs if self.max_values is None else self.max_values

        widths = (self.max_values - self.min_values) / self.n_bins
        widths = np.where(widths > 1e-12, widths, 1e-12)
        self.widths = widths

        ids = list(np.ndindex(*self.n_bins.tolist()))
        centers = np.array(
            [
                self.min_values + (np.asarray(bin_id, dtype=float) + 0.5) * widths
                for bin_id in ids
            ],
            dtype=float,
        )
        id_to_row = {bin_id: i for i, bin_id in enumerate(ids)}

        bin_ids = self.find_bins(points)
        populations, populated_data = bucket_frames(bin_ids, self.n_bins.tolist())

        target_closeness = None
        if self.target is not None:
            target_bin = self.find_bins(self.target.reshape(1, -1))[0]
            target_center = centers[id_to_row[tuple(target_bin.tolist())]]
            distance = np.linalg.norm(centers - target_center, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                closeness = np.where(distance > 0, 1.0 / distance, 0.0)
            denom = closeness.max() - closeness.min()
            target_closeness = (
                (closeness - closeness.min()) / denom if denom > 1e-12 else np.ones_like(closeness)
            )

        return BinTable(
            ids=ids,
            centers=centers,
            populations=populations,
            populated_data=populated_data,
            target_closeness=target_closeness,
        )

    def find_bins(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        raw = np.floor((points - self.min_values) / self.widths).astype(int)
        return np.clip(raw, 0, self.n_bins - 1)


class VoronoiBinner:
    """KMeans-backed Voronoi binning for 2D projected CV points."""

    def __init__(
        self,
        n_clusters: int = 150,
        min_values: list[float] | None = None,
        max_values: list[float] | None = None,
        target: list[float] | None = None,
        periodic: bool = False,
        grid_size: int = 250,
        padding: float = 0.1,
        seed: int = 42,
    ):
        self.n_clusters = int(n_clusters)
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be positive.")
        self.min_values = None if min_values is None else np.asarray(min_values, dtype=float)
        self.max_values = None if max_values is None else np.asarray(max_values, dtype=float)
        self.target = None if target is None else np.asarray(target, dtype=float)
        self.periodic = periodic
        self.grid_size = int(grid_size)
        self.padding = padding
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def fit(self, points: np.ndarray) -> BinTable:
        points = self._as_2d(points)
        if self.min_values is None or self.max_values is None:
            mins, maxs = padded_bounds(points, self.padding)
            mins = mins if self.min_values is None else self.min_values
            maxs = maxs if self.max_values is None else self.max_values
        else:
            mins, maxs = self.min_values, self.max_values
        self.bounding_box = np.array(
            [mins[0], maxs[0], mins[1], maxs[1]], dtype=float
        )

        self.n_clusters = min(self.n_clusters, len(points))
        if self.n_clusters == len(points):
            centers = points.copy()
        else:
            try:
                from sklearn.cluster import MiniBatchKMeans

                # Downsample to a maximum of max(10000, 20*n_clusters) for fast fitting
                max_samples = max(10000, 20 * self.n_clusters)
                if len(points) > max_samples:
                    idx = self.rng.choice(len(points), max_samples, replace=False)
                    fit_points = points[idx]
                else:
                    fit_points = points

                model = MiniBatchKMeans(
                    n_clusters=self.n_clusters,
                    random_state=self.seed,
                    n_init=1,
                    batch_size=max(1024, 3 * self.n_clusters),
                )
                centers = model.fit(fit_points).cluster_centers_
            except ModuleNotFoundError:
                centers = _kmeans_centers(points, self.n_clusters)
        self.centers = self._wrap_points(np.asarray(centers, dtype=float))

        bin_ids = self.find_bins(points)
        populations = np.zeros(self.n_clusters, dtype=int)
        populated_data: list[list[int]] = [[] for _ in range(self.n_clusters)]
        for frame_index, bin_id in enumerate(bin_ids):
            populations[int(bin_id)] += 1
            populated_data[int(bin_id)].append(frame_index)

        target_closeness = None
        if self.target is not None:
            target_bin = int(self.find_bins(self.target.reshape(1, -1))[0])
            target_center = self.centers[target_bin]
            distance = np.sqrt(
                self._distance_squared(
                    self.centers, target_center.reshape(1, -1)
                ).reshape(-1)
            )
            denom = distance.max() - distance.min()
            target_closeness = (
                (distance.max() - distance) / denom
                if denom > 1e-12
                else np.ones_like(distance)
            )

        return BinTable(
            ids=list(range(self.n_clusters)),
            centers=self.centers,
            populations=populations,
            populated_data=populated_data,
            area=self._estimate_areas(),
            target_closeness=target_closeness,
        )

    def find_bins(self, points: np.ndarray) -> np.ndarray:
        points = self._wrap_points(self._as_2d(points))
        return np.argmin(self._distance_squared(points, self.centers), axis=1).astype(int)

    def _estimate_areas(self) -> np.ndarray:
        if self.n_clusters == 1:
            x_min, x_max, y_min, y_max = self.bounding_box
            return np.asarray([float((x_max - x_min) * (y_max - y_min))])

        if self.periodic:
            return self._halfplane_clipped_areas(periodic=True)

        try:
            return self._scipy_clipped_areas()
        except Exception:
            return self._halfplane_clipped_areas(periodic=False)

    def _scipy_clipped_areas(self) -> np.ndarray:
        from scipy.spatial import Voronoi
        from shapely.geometry import Polygon, box

        x_min, x_max, y_min, y_max = self.bounding_box
        clip_box = box(x_min, y_min, x_max, y_max)
        vor = Voronoi(self.centers)
        regions, vertices = _finite_voronoi_regions_2d(
            vor, radius=2.0 * max(x_max - x_min, y_max - y_min)
        )
        areas = np.zeros(self.n_clusters, dtype=float)
        for index, region in enumerate(regions):
            polygon = Polygon(vertices[region])
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            clipped = polygon.intersection(clip_box)
            areas[index] = max(float(clipped.area), 0.0)

        return self._normalize_area_failures(areas)

    def _halfplane_clipped_areas(self, periodic: bool = False) -> np.ndarray:
        from shapely.geometry import Polygon

        x_min, x_max, y_min, y_max = self.bounding_box
        box_vertices = np.asarray(
            [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
            ],
            dtype=float,
        )
        comparison_centers = self._comparison_centers(periodic=periodic)
        areas = np.zeros(self.n_clusters, dtype=float)
        for center_index, center in enumerate(self.centers):
            vertices = box_vertices.copy()
            for other_index, other in comparison_centers:
                if other_index == center_index and np.allclose(other, center):
                    continue
                vertices = _clip_polygon_to_voronoi_halfplane(vertices, center, other)
                if len(vertices) == 0:
                    break
            if len(vertices) >= 3:
                polygon = Polygon(vertices)
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
                areas[center_index] = max(float(polygon.area), 0.0)

        return self._normalize_area_failures(areas)

    def _comparison_centers(self, periodic: bool = False) -> list[tuple[int, np.ndarray]]:
        if not periodic:
            return [(index, center.copy()) for index, center in enumerate(self.centers)]

        x_min, x_max, y_min, y_max = self.bounding_box
        widths = np.asarray([x_max - x_min, y_max - y_min], dtype=float)
        centers: list[tuple[int, np.ndarray]] = []
        for index, center in enumerate(self.centers):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    shift = np.asarray([dx, dy], dtype=float) * widths
                    centers.append((index, center + shift))
        return centers

    def _normalize_area_failures(self, areas: np.ndarray) -> np.ndarray:
        x_min, x_max, y_min, y_max = self.bounding_box
        total_area = float((x_max - x_min) * (y_max - y_min))
        areas = np.asarray(areas, dtype=float)
        areas = np.where(np.isfinite(areas) & (areas > 0.0), areas, 0.0)
        if areas.sum() <= 1e-12:
            return np.ones(self.n_clusters, dtype=float) * (
                total_area / self.n_clusters
            )
        return areas / areas.sum() * total_area

    def _distance_squared(self, points: np.ndarray, centers: np.ndarray) -> np.ndarray:
        if not self.periodic:
            from scipy.spatial.distance import cdist

            return cdist(points, centers, "sqeuclidean")

        delta = points[:, None, :] - centers[None, :, :]
        x_min, x_max, y_min, y_max = self.bounding_box
        widths = np.array([x_max - x_min, y_max - y_min], dtype=float)
        delta = delta - widths * np.round(delta / widths)
        return np.sum(delta * delta, axis=2)

    def _wrap_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float).copy()
        if not hasattr(self, "bounding_box"):
            return points
        x_min, x_max, y_min, y_max = self.bounding_box
        if self.periodic:
            points[:, 0] = ((points[:, 0] - x_min) % (x_max - x_min)) + x_min
            points[:, 1] = ((points[:, 1] - y_min) % (y_max - y_min)) + y_min
        else:
            points[:, 0] = np.clip(points[:, 0], x_min, x_max)
            points[:, 1] = np.clip(points[:, 1], y_min, y_max)
        return points

    @staticmethod
    def _as_2d(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Voronoi binning expects 2D projected points.")
        return points


def _kmeans_centers(points: np.ndarray, n_clusters: int, max_iter: int = 100) -> np.ndarray:
    """Small deterministic KMeans fallback used when scikit-learn is unavailable."""
    centers = _farthest_initial_centers(points, n_clusters)
    for _ in range(max_iter):
        distances = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for cluster_id in range(n_clusters):
            members = points[labels == cluster_id]
            if len(members):
                new_centers[cluster_id] = members.mean(axis=0)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return centers


def _farthest_initial_centers(points: np.ndarray, n_clusters: int) -> np.ndarray:
    centers = [points[0]]
    min_dist = np.sum((points - centers[0]) ** 2, axis=1)
    for _ in range(1, n_clusters):
        next_index = int(np.argmax(min_dist))
        centers.append(points[next_index])
        dist = np.sum((points - points[next_index]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)
    return np.asarray(centers, dtype=float)


def _finite_voronoi_regions_2d(
    vor: Any, radius: float | None = None
) -> tuple[list[list[int]], np.ndarray]:
    if vor.points.shape[1] != 2:
        raise ValueError("Only 2D Voronoi diagrams are supported.")

    new_regions: list[list[int]] = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = float(np.ptp(vor.points, axis=0).max() * 2.0)

    all_ridges: dict[int, list[tuple[int, int, int]]] = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices, strict=False):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for point_index, region_index in enumerate(vor.point_region):
        vertices = vor.regions[region_index]
        if all(vertex >= 0 for vertex in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[point_index]
        new_region = [vertex for vertex in vertices if vertex >= 0]
        for point_2, vertex_1, vertex_2 in ridges:
            if vertex_2 < 0:
                vertex_1, vertex_2 = vertex_2, vertex_1
            if vertex_1 >= 0:
                continue

            tangent = vor.points[point_2] - vor.points[point_index]
            norm = np.linalg.norm(tangent)
            if norm <= 1e-12:
                continue
            tangent /= norm
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = vor.points[[point_index, point_2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[vertex_2] + direction * radius
            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        vs = np.asarray([new_vertices[vertex] for vertex in new_region])
        centroid = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - centroid[1], vs[:, 0] - centroid[0])
        new_region = [vertex for _, vertex in sorted(zip(angles, new_region, strict=False))]
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices, dtype=float)


def _clip_polygon_to_voronoi_halfplane(
    vertices: np.ndarray,
    center: np.ndarray,
    other: np.ndarray,
    tolerance: float = 1e-12,
) -> np.ndarray:
    if len(vertices) == 0:
        return vertices

    normal = 2.0 * (other - center)
    offset = float(np.dot(other, other) - np.dot(center, center))

    def inside(point: np.ndarray) -> bool:
        return float(np.dot(normal, point) - offset) <= tolerance

    def intersection(start: np.ndarray, end: np.ndarray) -> np.ndarray:
        direction = end - start
        denominator = float(np.dot(normal, direction))
        if abs(denominator) <= tolerance:
            return end
        t = (offset - float(np.dot(normal, start))) / denominator
        t = min(max(t, 0.0), 1.0)
        return start + t * direction

    clipped: list[np.ndarray] = []
    previous = vertices[-1]
    previous_inside = inside(previous)
    for current in vertices:
        current_inside = inside(current)
        if current_inside:
            if not previous_inside:
                clipped.append(intersection(previous, current))
            clipped.append(current)
        elif previous_inside:
            clipped.append(intersection(previous, current))
        previous = current
        previous_inside = current_inside

    if not clipped:
        return np.empty((0, 2), dtype=float)
    return np.asarray(clipped, dtype=float)
