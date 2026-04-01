import io
from typing import Literal

import numpy as np
import trimesh

SUPPORTED_EXTENSIONS = {".glb", ".ply", ".obj"}
_EXT_TO_TYPE = {".glb": "glb", ".ply": "ply", ".obj": "obj"}


def load_mesh(data: bytes, extension: str) -> trimesh.Trimesh:
    """Load a mesh from raw bytes. Normalizes to unit cube."""
    ext = extension.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported format '{ext}', accept: .glb, .ply, .obj")

    try:
        raw = data.encode() if isinstance(data, str) else data
        mesh = trimesh.load(
            io.BytesIO(raw), file_type=_EXT_TO_TYPE[ext], force="mesh"
        )
    except Exception as e:
        raise ValueError(f"Failed to parse mesh file: {e}")

    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("Failed to parse mesh file: no valid geometry found")

    # Normalize to unit cube (same as ValDataset)
    vertices = mesh.vertices
    bbmin, bbmax = vertices.min(0), vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * 0.9 / (bbmax - bbmin).max()
    mesh.vertices = (vertices - center) * scale

    return mesh


def to_multipart_glb(
    mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
    mode: Literal["auto", "prompt"] = "auto",
) -> bytes:
    """Convert a labeled mesh into a multipart GLB scene."""
    scene = trimesh.Scene()

    if mode == "prompt":
        selected_mask = face_labels == 1
        remainder_mask = ~selected_mask

        if selected_mask.any():
            submesh = mesh.submesh([np.where(selected_mask)[0]], append=True)
            scene.add_geometry(submesh, node_name="selected_part", geom_name="selected_part")

        if remainder_mask.any():
            submesh = mesh.submesh([np.where(remainder_mask)[0]], append=True)
            scene.add_geometry(submesh, node_name="remainder", geom_name="remainder")
    else:
        unique_labels = np.unique(face_labels)
        for label_id in unique_labels:
            face_mask = face_labels == label_id
            submesh = mesh.submesh([np.where(face_mask)[0]], append=True)
            name = f"part_{label_id}"
            scene.add_geometry(submesh, node_name=name, geom_name=name)

    return scene.export(file_type="glb")
