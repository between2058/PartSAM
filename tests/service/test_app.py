import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import trimesh
from httpx import ASGITransport, AsyncClient


def _make_box_glb() -> bytes:
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    return mesh.export(file_type="glb")


@pytest.fixture
def mock_model_manager():
    """Patch model_manager so no GPU is needed."""
    with patch("service.app.model_manager") as mm:
        mm.settings = MagicMock()
        mm.settings.max_faces = 200000
        mm.settings.device = "cpu"

        # acquire() is an async context manager yielding a mock model
        mock_model = MagicMock()
        acm = AsyncMock()
        acm.__aenter__ = AsyncMock(return_value=mock_model)
        acm.__aexit__ = AsyncMock(return_value=False)
        mm.acquire.return_value = acm

        yield mm


@pytest.fixture
def mock_auto_segment():
    with patch("service.app.auto_segment") as mock:
        mock.return_value = np.array([0] * 6 + [1] * 6)
        yield mock


@pytest.fixture
def mock_prompt_segment():
    with patch("service.app.prompt_segment") as mock:
        mock.return_value = np.array([0] * 6 + [1] * 6)
        yield mock


@pytest.mark.asyncio
async def test_health(mock_model_manager):
    from service.app import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body


@pytest.mark.asyncio
async def test_segment_auto_returns_glb(
    mock_model_manager, mock_auto_segment
):
    from service.app import app

    glb_data = _make_box_glb()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/segment/auto",
            files={"file": ("test.glb", glb_data, "application/octet-stream")},
        )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/octet-stream"
    assert len(resp.content) > 0


@pytest.mark.asyncio
async def test_segment_auto_unsupported_format(mock_model_manager):
    from service.app import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/segment/auto",
            files={"file": ("test.stl", b"fake", "application/octet-stream")},
        )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_segment_prompt_returns_glb(
    mock_model_manager, mock_prompt_segment
):
    from service.app import app

    glb_data = _make_box_glb()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/segment/prompt",
            files={"file": ("test.glb", glb_data, "application/octet-stream")},
            data={
                "points": json.dumps([[0.0, 0.0, 0.0]]),
                "labels": json.dumps([1]),
            },
        )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/octet-stream"


@pytest.mark.asyncio
async def test_segment_prompt_missing_points(mock_model_manager):
    from service.app import app

    glb_data = _make_box_glb()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/segment/prompt",
            files={"file": ("test.glb", glb_data, "application/octet-stream")},
            data={"labels": json.dumps([1])},
        )
    assert resp.status_code == 422
