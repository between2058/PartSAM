import numpy as np
import torch
import trimesh
import pytest


def test_prepare_returns_expected_keys():
    from service.preprocessing import prepare

    mesh = trimesh.creation.box(extents=[1, 1, 1])
    result = prepare(mesh, num_points=1000)
    expected_keys = {"coords", "color", "normal", "point_to_face", "vertices", "faces"}
    assert set(result.keys()) == expected_keys


def test_prepare_tensor_shapes():
    from service.preprocessing import prepare

    mesh = trimesh.creation.box(extents=[1, 1, 1])
    num_points = 1000
    result = prepare(mesh, num_points=num_points)
    assert result["coords"].shape == (1, num_points, 3)
    assert result["color"].shape == (1, num_points, 3)
    assert result["normal"].shape == (1, num_points, 3)
    assert result["point_to_face"].shape == (1, num_points)
    assert result["vertices"].shape[0] == 1
    assert result["vertices"].shape[2] == 3
    assert result["faces"].shape[0] == 1
    assert result["faces"].shape[2] == 3


def test_prepare_tensor_dtypes():
    from service.preprocessing import prepare

    mesh = trimesh.creation.box(extents=[1, 1, 1])
    result = prepare(mesh, num_points=1000)
    assert result["coords"].dtype == torch.float32
    assert result["color"].dtype == torch.float32
    assert result["normal"].dtype == torch.float32
