import numpy as np
import torch
import trimesh

from utils.aug import CenterShift, NormalizeMy, NormalizeColor, ToTensor


def _sample_surface(mesh, count, sample_color=False, seed=147):
    """Sample points on the mesh surface. Extracted from utils/point.py to avoid open3d dependency."""
    face_weight = mesh.area_faces
    weight_cum = np.cumsum(face_weight)
    random = np.random.default_rng(seed).random
    face_pick = random(count) * weight_cum[-1]
    face_index = np.searchsorted(weight_cum, face_pick)

    tri_origins = mesh.vertices[mesh.faces[:, 0]]
    tri_vectors = mesh.vertices[mesh.faces[:, 1:]].copy()
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]

    if sample_color and hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
        uv_origins = mesh.visual.uv[mesh.faces[:, 0]]
        uv_vectors = mesh.visual.uv[mesh.faces[:, 1:]].copy()
        uv_origins_tile = np.tile(uv_origins, (1, 2)).reshape((-1, 2, 2))
        uv_vectors -= uv_origins_tile
        uv_origins = uv_origins[face_index]
        uv_vectors = uv_vectors[face_index]

    random_lengths = random((len(tri_vectors), 2, 1))
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)
    samples = sample_vector + tri_origins

    if sample_color:
        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
            sample_uv_vector = (uv_vectors * random_lengths).sum(axis=1)
            uv_samples = sample_uv_vector + uv_origins
            try:
                texture = mesh.visual.material.baseColorTexture
            except Exception:
                texture = mesh.visual.material.image
            colors = trimesh.visual.color.uv_to_interpolated_color(uv_samples, texture)
        else:
            try:
                colors = mesh.visual.face_colors[face_index]
            except Exception:
                colors = None
        return samples, face_index, colors

    return samples, face_index


def prepare(mesh: trimesh.Trimesh, num_points: int = 100000, seed: int = 666) -> dict:
    """Sample points from mesh and prepare tensors for model input.

    Replicates the pipeline from ValDataset.__getitem__ + collate_fn_eval + prep_points_train(eval=True).
    """
    # Sample surface points with colors and normals
    points, point_to_face, colors = _sample_surface(
        mesh, count=num_points, sample_color=True, seed=seed
    )

    if colors is None:
        colors = np.full((points.shape[0], 3), 192, dtype=np.uint8)
    colors = colors[:, :3]

    if hasattr(mesh, "face_normals") and mesh.face_normals is not None:
        normals = mesh.face_normals[point_to_face]
    else:
        normals = np.ones((points.shape[0], 3), dtype=np.float32)

    # Apply eval transforms (same as prep_points_train with eval=True)
    data_dict = {
        "coord": points,
        "color": colors,
        "normal": normals,
        "vertices": mesh.vertices,
    }
    data_dict = CenterShift(apply_z=True)(data_dict)
    data_dict = NormalizeMy()(data_dict)
    data_dict = NormalizeColor()(data_dict)
    data_dict = ToTensor()(data_dict)

    # Shape into batch dimension [1, N, D]
    return {
        "coords": data_dict["coord"].unsqueeze(0),
        "color": data_dict["color"].unsqueeze(0),
        "normal": data_dict["normal"].unsqueeze(0),
        "point_to_face": torch.from_numpy(point_to_face).unsqueeze(0),
        "vertices": data_dict["vertices"].unsqueeze(0),
        "faces": torch.from_numpy(mesh.faces).unsqueeze(0),
    }
