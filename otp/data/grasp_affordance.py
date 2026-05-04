"""
Grasp affordance pre-computation for OTP-Soft.

Three public entry points:

  sample_antipodal_grasps(mesh_vertices, mesh_normals, ...) -> np.ndarray (N, 7)
      GraspNet-baseline antipodal sampler over a vertex point cloud.
      Returns N grasp poses [pos(3) | quat(w,x,y,z)].  Invalid rows are
      all-zero pads (caller derives valid_mask from
      ~np.all(grasps == 0, axis=1)).

  extract_object_meshes(libero_bddl_path, output_dir)
      Parse a LIBERO BDDL file, locate object .obj/.stl mesh files via
      the LIBERO assets directory, downsample to 256 vertices via FPS,
      compute vertex normals, write npz with {vertices, normals}.

  precompute_all_affordances(libero_object_dir, grasp_output_dir, num_grasps=8)
      Walk the LIBERO assets dir, run sample_antipodal_grasps on every
      object mesh, and write npz {grasps, mesh_vertices, valid_mask}.

The first function is pure-numpy and is exercised in unit tests.
The second and third call into trimesh / LIBERO and are exercised by
scripts/03c_extract_grasp_affordances.py at run time.

Calibration contract (§5.1): output filenames are <object_name>.npz where
<object_name> matches LIBERO's BDDL fixture name verbatim.  LiberoOTPDataset
will FileNotFoundError if it cannot map an object_indices entry to a file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public: antipodal grasp sampler
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
    GraspNet-baseline antipodal grasp sampler.

    Args:
        mesh_vertices: (N_pts, 3) float — point cloud surface samples.
        mesh_normals:  (N_pts, 3) float — outward unit normals at each point.
        num_grasps:    Target number of grasps to return.
        gripper_width: Maximum allowed contact-pair distance [m].
        friction_coef: Coulomb friction coefficient.  Defines the friction
                       cone half-angle = arctan(friction_coef).  Two
                       contacts form a force-closure pair iff each contact
                       normal lies within the friction cone of the
                       opposite contact's negative normal.
        seed:          RNG seed for reproducibility.

    Returns:
        (num_grasps, 7) float32 — each row is [x, y, z, qw, qx, qy, qz].
        Invalid (un-found) rows are all zeros; callers compute a
        valid mask via ``~np.all(grasps == 0, axis=1)``.

    Algorithm:
      1. Repeat up to 100 * num_grasps attempts:
         a. Sample two random vertices (p_a, n_a) and (p_b, n_b).
         b. Reject if |p_b - p_a| ≥ gripper_width or distance ≈ 0.
         c. Reject if n_a · n_b > -cos(friction_angle)
            (i.e. normals are not anti-aligned within friction cone).
         d. Reject if (p_b - p_a) is not aligned with n_a within
            friction cone (n_a · line_dir < cos(friction_angle)).
         e. Build grasp pose:
              position    = (p_a + p_b) / 2
              closing axis = (p_b - p_a) / ||p_b - p_a||
              approach axis ⊥ closing, world-Z preferred
              transverse  = approach × closing
            Convert rotation matrix → quaternion via so3_to_quat.
      2. Stop early once num_grasps valid samples are accumulated.
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
    Internal sampler that also returns contact pairs, used by tests.

    Returns dict with:
      grasps:     (num_grasps, 7) float32  pose [pos | quat(wxyz)]
      valid_mask: (num_grasps,)   bool
      contact_a:  (num_grasps, 3) float32  first contact point
      contact_b:  (num_grasps, 3) float32  second contact point
      normal_a:   (num_grasps, 3) float32
      normal_b:   (num_grasps, 3) float32
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
        if found >= num_grasps:
            break
        if N < 2:
            break

        i = int(rng.integers(N))
        j = int(rng.integers(N))
        if i == j:
            continue

        p_a = mesh_vertices[i]
        n_a = mesh_normals[i]
        p_b = mesh_vertices[j]
        n_b = mesh_normals[j]

        # 1) Distance test.
        diff = p_b - p_a
        dist = float(np.linalg.norm(diff))
        if dist < 1e-6 or dist >= gripper_width:
            continue

        # 2) Normals anti-aligned within friction cone.
        if float(np.dot(n_a, n_b)) > -cos_friction:
            continue

        # 3) Line p_a→p_b enters the object at p_a, so it should be
        #    anti-aligned with the OUTWARD normal n_a (i.e. aligned with
        #    -n_a) to within the friction cone.
        line_dir = diff / dist
        if float(np.dot(n_a, line_dir)) > -cos_friction:
            continue

        # 4) Symmetric at p_b: line exits the object at p_b, so it should
        #    be aligned with the OUTWARD normal n_b within the friction cone.
        if float(np.dot(n_b, line_dir)) < cos_friction:
            continue

        # ---------- Build grasp pose ----------
        position = 0.5 * (p_a + p_b)

        # Closing axis (X).
        x_axis = line_dir

        # Approach axis (Z): preferred world up, falling back to world X if
        # x_axis is too aligned with up.
        z_pref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(x_axis, z_pref))) > 0.95:
            z_pref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        z_proj = z_pref - float(np.dot(z_pref, x_axis)) * x_axis
        z_norm = float(np.linalg.norm(z_proj))
        if z_norm < 1e-6:
            continue
        z_axis = z_proj / z_norm

        # Transverse axis (Y).
        y_axis = np.cross(z_axis, x_axis)

        R = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float64)
        # Convert to quaternion (wxyz) via the project's lie_algebra helper.
        quat = _rotation_matrix_to_quat_wxyz(R)

        grasps[found, :3] = position.astype(np.float32)
        grasps[found, 3:] = quat.astype(np.float32)
        valid_mask[found] = True
        contact_a[found] = p_a
        contact_b[found] = p_b
        normal_a[found] = n_a
        normal_b[found] = n_b
        found += 1

    return {
        "grasps":     grasps,
        "valid_mask": valid_mask,
        "contact_a":  contact_a,
        "contact_b":  contact_b,
        "normal_a":   normal_a,
        "normal_b":   normal_b,
    }


def _rotation_matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3×3 rotation matrix to quaternion (wxyz) using the
    project's ``so3_to_quat`` helper for consistency with downstream code.
    """
    import torch

    from otp.utils.lie_algebra import so3_to_quat

    R_t = torch.from_numpy(R.astype(np.float32)).unsqueeze(0)
    q = so3_to_quat(R_t).squeeze(0).numpy()       # (4,) wxyz, w >= 0
    return q


# ---------------------------------------------------------------------------
# Public: mesh extraction (LIBERO + trimesh)
# ---------------------------------------------------------------------------

def extract_object_meshes(
    libero_bddl_path: Union[str, Path],
    output_dir: Union[str, Path],
    num_points: int = 256,
    asset_root: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Parse a LIBERO BDDL file, load object meshes, FPS-downsample, save npz.

    Args:
        libero_bddl_path: Path to a *.bddl file.
        output_dir:       Directory where per-object npz files are written.
        num_points:       FPS downsample target.
        asset_root:       Override LIBERO asset root for mesh resolution.

    Returns:
        dict[object_name, Path] of written npz files.
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
    for obj_name, mesh_path in object_to_mesh.items():
        if not mesh_path.exists():
            logger.warning("[%s] mesh missing: %s", obj_name, mesh_path)
            continue
        mesh = trimesh.load(str(mesh_path), force="mesh")
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        norms = np.asarray(mesh.vertex_normals, dtype=np.float32)
        idx = _farthest_point_sample(verts, num_points)
        v_ds = verts[idx]
        n_ds = norms[idx]
        out_path = out_dir / f"{obj_name}.mesh.npz"
        np.savez(out_path, vertices=v_ds, normals=n_ds)
        written[obj_name] = out_path
    return written


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
    For every BDDL file under ``libero_object_dir``, extract its object
    meshes, sample antipodal grasps, and write per-object affordance npz
    files under ``grasp_output_dir``.

    Output npz layout (one file per UNIQUE object across all BDDLs):
        grasps:        (num_grasps, 7) float32   pose [pos | quat(wxyz)]
        mesh_vertices: (num_points, 3) float32
        valid_mask:    (num_grasps,)   bool

    Returns:
        dict[object_name, Path] — written affordance files.
    """
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

        for obj_name, mesh_path in mesh_files.items():
            if obj_name in seen_objects:
                continue                          # already processed
            seen_objects[obj_name] = mesh_path

            mesh = np.load(mesh_path)
            v = mesh["vertices"]
            n = mesh["normals"]
            grasps = sample_antipodal_grasps(
                v, n,
                num_grasps=num_grasps,
                gripper_width=gripper_width,
                friction_coef=friction_coef,
                seed=hash(obj_name) & 0x7FFF_FFFF,    # deterministic per object
            )
            valid_mask = ~np.all(grasps == 0, axis=1)

            out_path = out_dir / f"{obj_name}.npz"
            np.savez(
                out_path,
                grasps=grasps,
                mesh_vertices=v,
                valid_mask=valid_mask,
            )
            written[obj_name] = out_path

    # Summary manifest for downstream consumers.
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
    """Return n FPS indices into points (deterministic for fixed seed=0)."""
    N = len(points)
    if N <= n:
        idx = np.arange(N)
        # Pad by repeating the last index if mesh smaller than n_points.
        if N < n:
            pad = np.full(n - N, N - 1, dtype=np.int64)
            idx = np.concatenate([idx, pad])
        return idx
    rng = np.random.default_rng(0)
    sampled = [int(rng.integers(N))]
    distances = np.full(N, np.inf, dtype=np.float64)
    for _ in range(n - 1):
        last = points[sampled[-1]]
        d = np.linalg.norm(points - last, axis=1)
        distances = np.minimum(distances, d)
        sampled.append(int(np.argmax(distances)))
    return np.array(sampled, dtype=np.int64)


def _parse_bddl_object_meshes(
    bddl_path: Path,
    asset_root: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Best-effort BDDL parser.

    LIBERO BDDL files use a Lisp-like syntax referencing fixture XML files.
    The fixture XML in turn references mesh assets (.obj/.stl).  This helper:
      1. opens the BDDL,
      2. greps for "fixtures:" block lines of form
            (<obj_name> <fixture_xml_filename>)
      3. resolves <fixture_xml_filename> under the LIBERO assets dir,
      4. parses the XML with ElementTree to find <mesh file="..."/>.

    A best-effort fallback returns the raw fixture filenames if the XML
    cannot be located.
    """
    import re
    import xml.etree.ElementTree as ET

    text = bddl_path.read_text()

    # Resolve LIBERO asset root: argument > sibling-of-BDDL guess.
    if asset_root is None:
        # LIBERO layout: bddl_files/<suite>/foo.bddl  +  assets/.../*.xml
        # Walk up until we find a directory containing 'assets'.
        cur = bddl_path.parent
        while cur != cur.parent and not (cur / "assets").exists():
            cur = cur.parent
        asset_root = cur / "assets" if (cur / "assets").exists() else bddl_path.parent

    object_to_mesh: Dict[str, Path] = {}
    # Match patterns like "akita_black_bowl_1 - akita_black_bowl"
    for m in re.finditer(r"([a-zA-Z0-9_]+)\s*-\s*([a-zA-Z0-9_]+)", text):
        obj_inst, obj_class = m.group(1), m.group(2)
        # Search for class XML under asset_root.
        candidates = list(Path(asset_root).rglob(f"{obj_class}.xml"))
        if not candidates:
            continue
        try:
            tree = ET.parse(candidates[0])
            root = tree.getroot()
            mesh_elem = root.find(".//mesh")
            if mesh_elem is not None and "file" in mesh_elem.attrib:
                mesh_file = mesh_elem.attrib["file"]
                full = (candidates[0].parent / mesh_file).resolve()
                object_to_mesh[obj_class] = full
        except Exception as e:
            logger.debug("XML parse failed for %s: %s", candidates[0], e)
    return object_to_mesh
