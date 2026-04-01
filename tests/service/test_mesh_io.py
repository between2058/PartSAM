import io
import numpy as np
import pytest
import trimesh


def _make_box_bytes(file_type: str) -> bytes:
    """Create a simple box mesh and export to bytes."""
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    return mesh.export(file_type=file_type)


def test_load_glb():
    from service.mesh_io import load_mesh

    data = _make_box_bytes("glb")
    mesh = load_mesh(data, ".glb")
    assert isinstance(mesh, trimesh.Trimesh)
    assert len(mesh.faces) == 12  # box has 12 triangles


def test_load_ply():
    from service.mesh_io import load_mesh

    data = _make_box_bytes("ply")
    mesh = load_mesh(data, ".ply")
    assert isinstance(mesh, trimesh.Trimesh)
    assert len(mesh.faces) == 12


def test_load_obj():
    from service.mesh_io import load_mesh

    data = _make_box_bytes("obj")
    mesh = load_mesh(data, ".obj")
    assert isinstance(mesh, trimesh.Trimesh)
    assert len(mesh.faces) == 12


def test_load_unsupported_format():
    from service.mesh_io import load_mesh

    with pytest.raises(ValueError, match="Unsupported format"):
        load_mesh(b"fake", ".stl")


def test_load_invalid_data():
    from service.mesh_io import load_mesh

    with pytest.raises(ValueError, match="Failed to parse"):
        load_mesh(b"not a mesh", ".glb")


def test_to_multipart_glb_auto():
    from service.mesh_io import to_multipart_glb

    mesh = trimesh.creation.box(extents=[1, 1, 1])
    # Assign half the faces to label 0, half to label 1
    face_labels = np.array([0] * 6 + [1] * 6)
    glb_bytes = to_multipart_glb(mesh, face_labels, mode="auto")
    assert isinstance(glb_bytes, bytes)
    assert len(glb_bytes) > 0
    # Reload and verify it's a scene with 2 geometries
    scene = trimesh.load(io.BytesIO(glb_bytes), file_type="glb")
    assert isinstance(scene, trimesh.Scene)
    assert len(scene.geometry) == 2


def test_to_multipart_glb_prompt():
    from service.mesh_io import to_multipart_glb

    mesh = trimesh.creation.box(extents=[1, 1, 1])
    # Binary labels: 0=remainder, 1=selected
    face_labels = np.array([0] * 6 + [1] * 6)
    glb_bytes = to_multipart_glb(mesh, face_labels, mode="prompt")
    assert isinstance(glb_bytes, bytes)
    scene = trimesh.load(io.BytesIO(glb_bytes), file_type="glb")
    assert isinstance(scene, trimesh.Scene)
    geom_names = set(scene.geometry.keys())
    assert "selected_part" in geom_names
    assert "remainder" in geom_names
