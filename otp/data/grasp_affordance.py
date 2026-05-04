"""
otp/data/grasp_affordance.py

Pre-compute task-irrelevant grasp affordances for OTP-Soft.

Two-stage sampler:
  1. Surface-aware antipodal grasps (GraspNet §3.2 style):
     uniformly sample contact points on the mesh SURFACE (not vertices),
     ray-cast through the mesh to find antipodal contact partners.  This
     correctly handles thin-walled containers (bowl, cup) where the
     two sides of the wall are valid antipodal contacts but never appear
     as a vertex pair.
  2. PCA-aligned fallback grasps:
     when antipodal sampler finds < num_grasps valid grasps, generate
     additional grasps whose approach axis = mesh's shortest PCA axis.
     The shortest principal axis is intrinsic geometry, so this fallback
     remains task-irrelevant (no language / instruction prior introduced).

Both stages are pose-only (4x4 SE(3) → 7-vec [pos | quat(wxyz)]); they
encode WHERE to grasp, not WHEN.  Decoder selects among the candidates
at run time.

Output npz layout (one file per UNIQUE object across all BDDLs):
    grasps:        (num_grasps, 7) float32   pose [pos | quat(wxyz)]
    mesh_vertices: (num_points, 3) float32   FPS sample for OTP-Soft input
    valid_mask:    (num_grasps,)   bool
    source_mask:   (num_grasps,)   |S6 byte string in {antipodal, pca, invalid}

Required packages: trimesh.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API: vertex-based antipodal (legacy, retained for tests)
# ---------------------------------------------------------------------------

def sample_antipodal_grasps(
    mesh_vertices: np.ndarray,
    mesh_normals: np.ndarray,
    num_grasps: int = 8,
    gripper_width: float = 0.08,
    friction_coef: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """
    Vertex-based antipodal grasp sampler (legacy).

    NOTE: Production pipelines should call ``sample_surface_antipodal_grasps``
    on a ``trimesh.Trimesh`` object instead; the vertex-based sampler fails
    on thin-walled and z-symmetric containers because the inner and outer
    walls of a bowl are seldom captured as a vertex pair.

    Args:
        mesh_vertices: (N_pts, 3) float — point cloud surface samples.
        mesh_normals:  (N_pts, 3) float — outward unit normals at each point.
        num_grasps:    Target number of grasps.
        gripper_width: Maximum allowed contact-pair distance [m].
        friction_coef: Coulomb friction coefficient.
        seed:          RNG seed for reproducibility.

    Returns:
        (num_grasps, 7) float32 — each row is [x, y, z, qw, qx, qy, qz].
        Invalid (un-found) rows are all zeros.
    """
    out = _sample_antipodal_grasps_internal(
        mesh_vertices, mesh_normals,
        num_grasps=num_grasps,
        gripper_width=gripper_width,
        friction_coef=friction_coef,
        seed=seed,
    )
    return out["grasps"]


def _sample_antipodal_grasps_internal(
    mesh_vertices: np.ndarray,
    mesh_normals: np.ndarray,
    num_grasps: int = 8,
    gripper_width: float = 0.08,
    friction_coef: float = 0.5,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Vertex-based sampler (legacy).  Returns dict with grasps + contact info.
    """
    if mesh_vertices.ndim != 2 or mesh_vertices.shape[1] != 3:
        raise ValueError(
            f"mesh_vertices must be (N, 3); got {mesh_vertices.shape}"
        )
    if mesh_normals.shape != mesh_vertices.shape:
        raise ValueError(
            f"mesh_normals shape {mesh_normals.shape} ≠ "
            f"mesh_vertices shape {mesh_vertices.shape}"
        )

    rng = np.random.default_rng(seed)
    N = len(mesh_vertices)
    friction_angle = float(np.arctan(friction_coef))
    cos_friction = float(np.cos(friction_angle))

    grasps = np.zeros((num_grasps, 7), dtype=np.float32)
    valid_mask = np.zeros(num_grasps, dtype=bool)
    contact_a = np.zeros((num_grasps, 3), dtype=np.float32)
    contact_b = np.zeros((num_grasps, 3), dtype=np.float32)
    normal_a = np.zeros((num_grasps, 3), dtype=np.float32)
    normal_b = np.zeros((num_grasps, 3), dtype=np.float32)

    found = 0
    max_attempts = 100 * num_grasps

    for _ in range(max_attempts):
        if found >= num_grasps or N < 2:
            break
        i = int(rng.integers(N))
        j = int(rng.integers(N))
        if i == j:
            continue
        p_a = mesh_vertices[i]
        n_a = mesh_normals[i]
        p_b = mesh_vertices[j]
        n_b = mesh_normals[j]

        diff = p_b - p_a
        dist = float(np.linalg.norm(diff))
        if dist < 1e-6 or dist >= gripper_width:
            continue

        if float(np.dot(n_a, n_b)) > -cos_friction:
            continue

        line_dir = diff / dist
        if float(np.dot(n_a, line_dir)) > -cos_friction:
            continue
        if float(np.dot(n_b, line_dir)) < cos_friction:
            continue

        # Build grasp pose.
        pos = 0.5 * (p_a + p_b)
        closing_axis = line_dir
        # approach: prefer world-up direction projected onto plane ⊥ closing.
        z_world = np.array([0, 0, 1], dtype=np.float64)
        approach = z_world - np.dot(z_world, closing_axis) * closing_axis
        if np.linalg.norm(approach) < 1e-3:
            # Closing axis is itself ~vertical; pick world-x as approach base.
            x_world = np.array([1, 0, 0], dtype=np.float64)
            approach = x_world - np.dot(x_world, closing_axis) * closing_axis
        approach /= np.linalg.norm(approach)
        transverse = np.cross(approach, closing_axis)

        R = np.column_stack([closing_axis, transverse, approach]).astype(
            np.float32
        )
        q = _rotation_matrix_to_quat_wxyz(R)

        grasps[found, :3] = pos.astype(np.float32)
        grasps[found, 3:] = q
        valid_mask[found] = True
        contact_a[found] = p_a
        contact_b[found] = p_b
        normal_a[found] = n_a
        normal_b[found] = n_b
        found += 1

    return {
        "grasps": grasps,
        "valid_mask": valid_mask,
        "contact_a": contact_a,
        "contact_b": contact_b,
        "normal_a": normal_a,
        "normal_b": normal_b,
    }


# ---------------------------------------------------------------------------
# Public API: surface-aware antipodal (production)
# ---------------------------------------------------------------------------

def sample_surface_antipodal_grasps(
    mesh,
    num_grasps: int = 8,
    gripper_width: float = 0.08,
    friction_coef: float = 0.5,
    finger_thickness: float = 0.004,
    finger_length: float = 0.04,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Surface-aware antipodal grasp sampler (GraspNet §3.2 style).

    Args:
        mesh:             trimesh.Trimesh — full triangulated mesh.
        num_grasps:       Target number of grasps.
        gripper_width:    Maximum gripper opening [m].
        friction_coef:    Coulomb friction coefficient.
        finger_thickness: Gripper-finger half-thickness [m]; used for
                          penetration check.
        finger_length:    Gripper-finger length [m]; used for
                          penetration check (sampling along finger axis).
        seed:             RNG seed.

    Returns:
        dict with:
          grasps:     (num_grasps, 7) float32  [pos | quat(wxyz)]
          valid_mask: (num_grasps,)   bool
          source:     (num_grasps,)   |S10 in {b'antipodal', b'invalid'}
    """
    import trimesh

    rng = np.random.default_rng(seed)
    friction_angle = float(np.arctan(friction_coef))
    cos_friction = float(np.cos(friction_angle))

    grasps = np.zeros((num_grasps, 7), dtype=np.float32)
    valid_mask = np.zeros(num_grasps, dtype=bool)
    source = np.array([b"invalid"] * num_grasps, dtype="|S10")

    found = 0
    # Surface sampling has higher hit rate than vertex sampling, so 30 rounds
    # of N_a candidate starts is usually sufficient.
    max_rounds = 30
    n_starts_per_round = max(num_grasps * 4, 32)
    eps = 1e-4

    for _round in range(max_rounds):
        if found >= num_grasps:
            break

        # 1. Sample N_a points on the mesh surface (face-area weighted).
        try:
            points, face_idx = trimesh.sample.sample_surface(
                mesh, n_starts_per_round, seed=int(rng.integers(2**31)),
            )
        except Exception as e:
            logger.debug("sample_surface failed: %s", e)
            break
        normals = mesh.face_normals[face_idx]

        # 2. For each starting contact, ray-cast through the mesh along -n_a
        #    to find antipodal partners.  We push the ray origin slightly
        #    inward to avoid hitting p_a itself due to numerical precision.
        for k in range(len(points)):
            if found >= num_grasps:
                break
            p_a = points[k]
            n_a = normals[k]
            if not np.isfinite(p_a).all() or not np.isfinite(n_a).all():
                continue
            n_a_norm = float(np.linalg.norm(n_a))
            if n_a_norm < 1e-6:
                continue
            n_a = n_a / n_a_norm

            # Ray from slightly inside p_a, going INTO the object (= -n_a).
            ray_origin = p_a + (-n_a) * eps
            ray_direction = -n_a

            try:
                hits, _, hit_face_idx = mesh.ray.intersects_location(
                    ray_origins=ray_origin[None, :],
                    ray_directions=ray_direction[None, :],
                    multiple_hits=True,
                )
            except Exception as e:
                logger.debug("ray intersect failed: %s", e)
                continue

            if len(hits) == 0:
                continue

            # 3. Pick the FIRST exit hit beyond the starting wall.
            #    For a thin shell the first hit may be the inner side of
            #    the same wall (very small distance); skip while distance
            #    is below a tiny threshold or while the hit normal is
            #    same-direction as n_a.
            best_pb = None
            best_nb = None
            best_dist = None
            dists = np.linalg.norm(hits - p_a[None, :], axis=1)
            order = np.argsort(dists)
            for ord_idx in order:
                pb = hits[ord_idx]
                nb = mesh.face_normals[hit_face_idx[ord_idx]]
                nb_norm = float(np.linalg.norm(nb))
                if nb_norm < 1e-6:
                    continue
                nb = nb / nb_norm
                d = float(dists[ord_idx])
                if d < 5 * eps:
                    continue
                if d >= gripper_width:
                    break
                # The hit normal should oppose the ray direction
                # (= align with n_a) for a true antipodal contact:
                #   ray_direction = -n_a, so nb · ray_direction < 0
                #   which means nb · n_a > 0 ... but we need the OUTWARD
                #   normal at the antipodal wall.  The face normal as
                #   reported by trimesh is the outward normal; for an
                #   antipodal contact on the opposite wall, n_b should
                #   point AWAY from p_a, i.e. nb · n_a < 0 within
                #   friction cone.
                if float(np.dot(nb, n_a)) > -cos_friction:
                    continue
                best_pb, best_nb, best_dist = pb, nb, d
                break

            if best_pb is None:
                continue

            # 4. Penetration check: build a virtual gripper at the midpoint
            #    and verify finger volumes don't intersect the mesh.
            pos = 0.5 * (p_a + best_pb)
            closing_axis = (best_pb - p_a) / best_dist
            # Approach axis: prefer world-Z projected onto plane perpendicular
            # to closing, falling back to world-X if degenerate.
            z_world = np.array([0, 0, 1], dtype=np.float64)
            approach = z_world - np.dot(z_world, closing_axis) * closing_axis
            if np.linalg.norm(approach) < 1e-3:
                x_world = np.array([1, 0, 0], dtype=np.float64)
                approach = x_world - np.dot(x_world, closing_axis) * closing_axis
            approach /= np.linalg.norm(approach)
            transverse = np.cross(approach, closing_axis)

            if not _gripper_clear_of_mesh(
                mesh, pos, closing_axis, approach, transverse,
                gripper_width=gripper_width,
                finger_thickness=finger_thickness,
                finger_length=finger_length,
            ):
                continue

            R = np.column_stack([closing_axis, transverse, approach]).astype(
                np.float32
            )
            q = _rotation_matrix_to_quat_wxyz(R)

            grasps[found, :3] = pos.astype(np.float32)
            grasps[found, 3:] = q
            valid_mask[found] = True
            source[found] = b"antipodal"
            found += 1

    return {
        "grasps": grasps,
        "valid_mask": valid_mask,
        "source": source,
    }


def _gripper_clear_of_mesh(
    mesh,
    pos: np.ndarray,
    closing_axis: np.ndarray,
    approach: np.ndarray,
    transverse: np.ndarray,
    gripper_width: float,
    finger_thickness: float,
    finger_length: float,
) -> bool:
    """
    Sanity check: when the gripper closes around `pos` with the given
    orientation, neither finger volume intersects the mesh interior.

    Approximation: for each finger, sample a 3x3 grid of points along
    finger length × finger thickness and check that none lie inside the
    mesh.  Cheap and conservative.
    """
    import trimesh  # noqa: F401

    # Finger center along closing axis: ±gripper_width/2 from grasp center,
    # backed off by finger_thickness so the finger touches but doesn't
    # collide-from-inside.
    half_w = gripper_width / 2.0 + finger_thickness
    samples_per_finger = []
    n_along = 3       # along finger length (approach direction).
    n_thick = 2       # along finger thickness (transverse direction).
    for sign in (-1, +1):
        center = pos + sign * closing_axis * half_w
        for ai in range(n_along):
            a = -finger_length * 0.4 + ai * (finger_length * 0.4)
            for ti in range(n_thick):
                t = (ti - (n_thick - 1) / 2.0) * finger_thickness
                pt = center + a * approach + t * transverse
                samples_per_finger.append(pt)

    pts = np.array(samples_per_finger, dtype=np.float64)
    try:
        inside = mesh.contains(pts)
    except Exception:
        # If the mesh is not watertight, contains() is unreliable.
        # Fall back to "no penetration" optimistically — surface antipodal
        # geometry by itself gives strong contact constraints.
        return True

    # Tolerate at most 1 sample inside (face boundary noise).
    return int(inside.sum()) <= 1


# ---------------------------------------------------------------------------
# Public API: PCA fallback grasp generator
# ---------------------------------------------------------------------------

def sample_pca_aligned_grasps(
    mesh,
    num_grasps: int = 8,
    gripper_width: float = 0.08,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    PCA-aligned grasp generator — task-irrelevant fallback.

    Algorithm:
      1. Compute PCA on mesh.vertices; let v1, v2, v3 be principal axes
         in DESCENDING eigenvalue order (longest, mid, shortest).
      2. approach axis  = v3        (shortest principal axis)
         closing axis   = v2        (mid principal axis)
         transverse     = v1        (longest principal axis)
      3. Generate `num_grasps` poses by rotating around the approach axis
         in equal yaw increments; positions are mesh AABB center, lifted
         along approach by the AABB half-extent along that axis.
      4. Validate that the AABB extent along the closing axis is < gripper_width;
         if not, the object is wider than the gripper along that axis and
         no PCA-aligned grasp can close — return all-invalid.

    The fallback introduces no language- or task-dependent prior; it relies
    only on the intrinsic geometry of the object.
    """
    grasps = np.zeros((num_grasps, 7), dtype=np.float32)
    valid_mask = np.zeros(num_grasps, dtype=bool)
    source = np.array([b"invalid"] * num_grasps, dtype="|S10")

    if not hasattr(mesh, "vertices") or len(mesh.vertices) < 4:
        return {"grasps": grasps, "valid_mask": valid_mask, "source": source}

    V = np.asarray(mesh.vertices, dtype=np.float64)
    centroid = V.mean(axis=0)
    Vc = V - centroid
    cov = (Vc.T @ Vc) / max(len(Vc) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)         # ascending
    # Reorder descending.
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    v1 = eigvecs[:, 0]      # longest axis
    v2 = eigvecs[:, 1]      # middle axis
    v3 = eigvecs[:, 2]      # shortest axis (approach direction)

    # Project AABB extents onto each axis.
    proj_v2 = Vc @ v2
    proj_v3 = Vc @ v3
    extent_v2 = float(proj_v2.max() - proj_v2.min())
    extent_v3 = float(proj_v3.max() - proj_v3.min())

    if extent_v2 >= gripper_width:
        # Cannot close along closing axis — give up.
        return {"grasps": grasps, "valid_mask": valid_mask, "source": source}

    approach_axis = v3 / max(np.linalg.norm(v3), 1e-12)
    # Position: mesh centroid lifted along +approach by half the v3 extent.
    pos = centroid + approach_axis * (extent_v3 / 2.0 + 0.005)

    rng = np.random.default_rng(seed)
    yaw_offsets = rng.uniform(0, 2 * np.pi / num_grasps, size=1)[0]

    for i in range(num_grasps):
        yaw = yaw_offsets + 2 * np.pi * i / num_grasps
        # Rotate closing/transverse around approach axis by yaw.
        c = np.cos(yaw)
        s = np.sin(yaw)
        closing = c * v2 + s * v1
        transverse = -s * v2 + c * v1
        # Re-orthonormalize for numerical safety.
        closing /= max(np.linalg.norm(closing), 1e-12)
        transverse = np.cross(approach_axis, closing)
        transverse /= max(np.linalg.norm(transverse), 1e-12)

        R = np.column_stack([closing, transverse, approach_axis]).astype(
            np.float32
        )
        q = _rotation_matrix_to_quat_wxyz(R)

        grasps[i, :3] = pos.astype(np.float32)
        grasps[i, 3:] = q
        valid_mask[i] = True
        source[i] = b"pca"

    return {"grasps": grasps, "valid_mask": valid_mask, "source": source}


# ---------------------------------------------------------------------------
# Quaternion helper
# ---------------------------------------------------------------------------

def _rotation_matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to (w, x, y, z) quaternion."""
    t = float(R[0, 0] + R[1, 1] + R[2, 2])
    if t > 0:
        s = float(np.sqrt(t + 1.0)) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = float(np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = float(np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = float(np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float32)
    n = float(np.linalg.norm(q))
    if n > 0:
        q /= n
    return q


# ---------------------------------------------------------------------------
# Mesh extraction (BDDL → trimesh.Trimesh)
# ---------------------------------------------------------------------------

def extract_object_meshes(
    libero_bddl_path: Union[str, Path],
    output_dir: Union[str, Path],
    num_points: int = 256,
    asset_root: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Parse a LIBERO BDDL file, load object meshes, save:
      - downsampled FPS point cloud (256 pts) for OTP-Soft input
      - scaled mesh path for surface-aware sampler

    Output per object: <output_dir>/<obj>.mesh.npz with keys:
      vertices: (num_points, 3)  FPS-downsampled (in metres, scaled)
      normals:  (num_points, 3)
      mesh_path: str            absolute path to the original .obj/.stl
      mesh_scale: (3,) float64  scale factor applied to convert raw mesh
                                units into metres (read from fixture XML)

    Note on scale: LIBERO's fixture XMLs encode the mesh-to-world scale via
    `<mesh scale="sx sy sz">` in the .xml.  Without applying it, raw mesh
    coordinates can be off by 1-2 orders of magnitude (e.g. alphabet_soup
    raw is ~6.6 m, scaled by 0.01 → 6.6 cm, which is the actual size).
    """
    try:
        import trimesh
    except ImportError as e:
        raise ImportError(
            "extract_object_meshes requires trimesh. "
            "Install with: pip install trimesh"
        ) from e

    bddl_path = Path(libero_bddl_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    object_to_mesh = _parse_bddl_object_meshes(bddl_path, asset_root)
    if not object_to_mesh:
        raise KeyError(f"No object meshes parsed from {bddl_path}")

    written: Dict[str, Path] = {}
    for obj_name, (mesh_path, scale) in object_to_mesh.items():
        if not mesh_path.exists():
            logger.warning("[%s] mesh missing: %s", obj_name, mesh_path)
            continue

        try:
            mesh = trimesh.load(str(mesh_path), force="mesh")
        except Exception as e:
            logger.warning("[%s] trimesh load failed: %s", obj_name, e)
            continue

        # Apply LIBERO fixture-XML scale to convert raw mesh units → metres.
        if not np.allclose(scale, 1.0):
            # apply_scale takes a single scalar or per-axis list.
            mesh.apply_scale(scale.tolist())

        verts = np.asarray(mesh.vertices, dtype=np.float32)
        norms = np.asarray(mesh.vertex_normals, dtype=np.float32)
        idx = _farthest_point_sample(verts, num_points)
        v_ds = verts[idx]
        n_ds = norms[idx]

        out_path = out_dir / f"{obj_name}.mesh.npz"
        np.savez(
            out_path,
            vertices=v_ds,
            normals=n_ds,
            mesh_path=str(mesh_path.resolve()),
            mesh_scale=scale.astype(np.float64),
        )
        written[obj_name] = out_path

    return written


# ---------------------------------------------------------------------------
# Top-level pipeline: BDDL → per-object affordance npz files
# ---------------------------------------------------------------------------

def precompute_all_affordances(
    libero_object_dir: Union[str, Path],
    grasp_output_dir: Union[str, Path],
    num_grasps: int = 8,
    gripper_width: float = 0.08,
    friction_coef: float = 0.5,
    num_points: int = 256,
    asset_root: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    For every BDDL file under ``libero_object_dir``, extract object meshes,
    sample antipodal grasps via the surface-aware sampler, fall back to
    PCA-aligned grasps if antipodal returned < num_grasps, and write
    per-object affordance npz files under ``grasp_output_dir``.

    Output npz layout (one file per UNIQUE object):
        grasps:        (num_grasps, 7) float32   pose [pos | quat(wxyz)]
        mesh_vertices: (num_points, 3) float32   FPS-downsampled
        valid_mask:    (num_grasps,)   bool
        source_mask:   (num_grasps,)   |S10 in {b'antipodal', b'pca', b'invalid'}

    Returns:
        dict[object_name, Path] — written affordance files.
    """
    try:
        import trimesh
    except ImportError as e:
        raise ImportError(
            "precompute_all_affordances requires trimesh. "
            "Install with: pip install trimesh"
        ) from e

    src_dir = Path(libero_object_dir)
    out_dir = Path(grasp_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bddl_paths = sorted(src_dir.rglob("*.bddl"))
    if not bddl_paths:
        raise FileNotFoundError(
            f"No .bddl files under {src_dir}. "
            f"Pass --libero-asset-dir pointing at LIBERO's "
            f"libero/libero/bddl_files (or assets) directory."
        )

    written: Dict[str, Path] = {}
    seen_objects: Dict[str, Path] = {}

    for bddl in bddl_paths:
        try:
            mesh_files = extract_object_meshes(
                bddl, out_dir / "_meshes", num_points=num_points,
                asset_root=asset_root,
            )
        except Exception as e:
            logger.warning("[%s] mesh extraction failed: %s", bddl.name, e)
            continue

        for obj_name, mesh_npz_path in mesh_files.items():
            if obj_name in seen_objects:
                continue
            seen_objects[obj_name] = mesh_npz_path

            mesh_npz = np.load(mesh_npz_path)
            v_ds = mesh_npz["vertices"]              # (num_points, 3) — already scaled
            mesh_path_str = str(mesh_npz["mesh_path"])
            # Mesh scale read from fixture XML; needs to be re-applied here
            # because we re-load the raw .obj for surface antipodal sampling.
            mesh_scale = mesh_npz["mesh_scale"] if "mesh_scale" in mesh_npz.files \
                         else np.array([1.0, 1.0, 1.0])

            # Load full mesh for surface-aware sampler.
            try:
                full_mesh = trimesh.load(mesh_path_str, force="mesh")
                if not np.allclose(mesh_scale, 1.0):
                    full_mesh.apply_scale(mesh_scale.tolist())
            except Exception as e:
                logger.warning(
                    "[%s] trimesh load failed at sampling stage: %s",
                    obj_name, e,
                )
                continue

            obj_seed = hash(obj_name) & 0x7FFF_FFFF

            # Stage 1: surface-aware antipodal.
            sa_out = sample_surface_antipodal_grasps(
                full_mesh,
                num_grasps=num_grasps,
                gripper_width=gripper_width,
                friction_coef=friction_coef,
                seed=obj_seed,
            )
            grasps = sa_out["grasps"].copy()
            valid_mask = sa_out["valid_mask"].copy()
            source_mask = sa_out["source"].copy()

            n_antipodal = int(valid_mask.sum())
            if n_antipodal < num_grasps:
                # Stage 2: PCA-aligned fallback for the remaining slots.
                pca_out = sample_pca_aligned_grasps(
                    full_mesh,
                    num_grasps=num_grasps - n_antipodal,
                    gripper_width=gripper_width,
                    seed=obj_seed + 1,
                )
                pca_grasps = pca_out["grasps"]
                pca_valid = pca_out["valid_mask"]
                pca_source = pca_out["source"]

                # Place PCA grasps into the empty slots.
                empty_slots = np.where(~valid_mask)[0]
                n_to_fill = min(len(empty_slots), int(pca_valid.sum()))
                if n_to_fill > 0:
                    pca_valid_idx = np.where(pca_valid)[0][:n_to_fill]
                    target_slots = empty_slots[:n_to_fill]
                    grasps[target_slots] = pca_grasps[pca_valid_idx]
                    valid_mask[target_slots] = True
                    source_mask[target_slots] = pca_source[pca_valid_idx]

            out_path = out_dir / f"{obj_name}.npz"
            np.savez(
                out_path,
                grasps=grasps,
                mesh_vertices=v_ds,
                valid_mask=valid_mask,
                source_mask=source_mask,
            )
            written[obj_name] = out_path

            logger.info(
                "[%s] %d antipodal + %d pca = %d valid / %d total",
                obj_name, n_antipodal,
                int((source_mask == b"pca").sum()),
                int(valid_mask.sum()), num_grasps,
            )

    # Manifest.
    summary = {
        "num_objects": len(written),
        "objects": sorted(written.keys()),
        "num_grasps_per_object": num_grasps,
        "gripper_width": gripper_width,
        "friction_coef": friction_coef,
        "num_points": num_points,
    }
    with open(out_dir / "affordance_manifest.json", "w") as f:
        json.dump(summary, f, indent=2)

    return written


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _farthest_point_sample(points: np.ndarray, n: int) -> np.ndarray:
    """Return indices of n farthest-point-sampled rows of `points`."""
    N = len(points)
    if N <= n:
        idx = np.arange(N)
        if N < n:
            idx = np.concatenate([idx, np.zeros(n - N, dtype=int)])
        return idx
    selected = np.zeros(n, dtype=int)
    dists = np.full(N, np.inf, dtype=np.float64)
    selected[0] = 0
    last = points[0]
    for i in range(1, n):
        d = np.linalg.norm(points - last, axis=1)
        dists = np.minimum(dists, d)
        nxt = int(np.argmax(dists))
        selected[i] = nxt
        last = points[nxt]
    return selected


def _parse_bddl_object_meshes(
    bddl_path: Path,
    asset_root: Optional[Path] = None,
) -> Dict[str, "tuple"]:
    """
    Best-effort BDDL parser with .msh -> .obj/.stl fallback.

    LIBERO fixture XMLs typically reference *_vis.msh (MuJoCo binary format)
    which trimesh cannot read.  Same directory usually contains a canonical
    <obj_class>.obj — we prefer that, then any .obj/.stl in the XML's tree.

    Returns:
        dict[obj_class, (mesh_path, scale)] where scale is a (3,) float
        ndarray applied to mesh.vertices to convert raw OBJ units into
        the simulator's metre-scale frame (read from the fixture XML's
        <mesh scale="..."> attribute; defaults to (1,1,1) if absent).
    """
    import re as _re
    import xml.etree.ElementTree as ET

    text = bddl_path.read_text()

    if asset_root is None:
        cur = bddl_path.parent
        while cur != cur.parent and not (cur / "assets").exists():
            cur = cur.parent
        asset_root = cur / "assets" if (cur / "assets").exists() else bddl_path.parent

    object_to_mesh: Dict[str, tuple] = {}

    for m in _re.finditer(r"([a-zA-Z0-9_]+)\s*-\s*([a-zA-Z0-9_]+)", text):
        obj_class = m.group(2)
        candidates = list(Path(asset_root).rglob(f"{obj_class}.xml"))
        if not candidates:
            continue
        xml_path = candidates[0]
        xml_dir = xml_path.parent

        chosen: Optional[Path] = None
        scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        # Read scale from the FIRST <mesh scale="..."> entry in the XML.
        # All meshes in a given fixture share the same scale in LIBERO.
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for mesh_elem in root.findall(".//mesh"):
                if "scale" in mesh_elem.attrib:
                    parts = mesh_elem.attrib["scale"].split()
                    if len(parts) == 3:
                        scale = np.array([float(p) for p in parts],
                                         dtype=np.float64)
                        break
        except Exception as e:
            logger.debug("XML scale parse failed for %s: %s", xml_path, e)

        # 1) Canonical <obj_class>.obj next to fixture XML.
        canonical_obj = xml_dir / f"{obj_class}.obj"
        if canonical_obj.exists():
            chosen = canonical_obj.resolve()

        # 2) Mesh entries in the XML, preferring .obj/.stl over .msh.
        if chosen is None:
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                mesh_entries = []
                for mesh_elem in root.findall(".//mesh"):
                    if "file" in mesh_elem.attrib:
                        mesh_entries.append(mesh_elem.attrib["file"])
                priority = {".obj": 0, ".stl": 1}
                mesh_entries.sort(
                    key=lambda f: priority.get(Path(f).suffix.lower(), 99)
                )
                for mesh_file in mesh_entries:
                    full = (xml_dir / mesh_file).resolve()
                    if full.exists() and full.suffix.lower() in (".obj", ".stl"):
                        chosen = full
                        break
            except Exception as e:
                logger.debug("XML parse failed for %s: %s", xml_path, e)

        # 3) Fallback: any .obj or .stl under xml_dir tree (skip *_col.obj).
        if chosen is None:
            for ext in (".obj", ".stl"):
                hits = sorted(xml_dir.rglob(f"*{ext}"))
                non_col = [p for p in hits if not p.stem.endswith("_col")]
                if non_col:
                    chosen = non_col[0].resolve()
                    break
                if hits:
                    chosen = hits[0].resolve()
                    break

        if chosen is not None:
            object_to_mesh[obj_class] = (chosen, scale)

    return object_to_mesh
