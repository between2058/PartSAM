# PartSAM FastAPI Microservice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a synchronous FastAPI microservice that exposes PartSAM's auto and prompt-based 3D part segmentation as HTTP endpoints, accepting mesh files and returning multipart GLB.

**Architecture:** Single-process FastAPI with asyncio.Semaphore for GPU concurrency control. Model loaded once at startup. Inference logic extracted from existing `eval_everypart.py`. Horizontal scaling via multiple instances behind a load balancer.

**Tech Stack:** FastAPI, uvicorn, pydantic-settings, python-multipart, trimesh, torch, pointops, hydra

**Spec:** `docs/superpowers/specs/2026-04-01-fastapi-microservice-design.md`

---

## File Structure

```
service/
├── __init__.py
├── app.py              # FastAPI app, routes, lifespan
├── config.py           # pydantic-settings Settings class
├── model_manager.py    # Model loading + Semaphore concurrency
├── inference.py        # auto_segment() + prompt_segment()
├── mesh_io.py          # Load mesh from bytes, export multipart GLB
└── preprocessing.py    # Point sampling + normalization pipeline
tests/
└── service/
    ├── __init__.py
    ├── test_config.py
    ├── test_mesh_io.py
    ├── test_preprocessing.py
    └── test_app.py
```

---

### Task 1: Project scaffolding and dependencies

**Files:**
- Create: `service/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/service/__init__.py`
- Create: `requirements-service.txt`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p service tests/service
```

- [ ] **Step 2: Create empty init files**

Create `service/__init__.py`:
```python
```

Create `tests/__init__.py`:
```python
```

Create `tests/service/__init__.py`:
```python
```

- [ ] **Step 3: Create requirements-service.txt**

```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
pydantic-settings>=2.0.0
python-multipart>=0.0.9
pytest>=8.0.0
httpx>=0.27.0
```

- [ ] **Step 4: Install dependencies**

Run: `pip install -r requirements-service.txt`

- [ ] **Step 5: Commit**

```bash
git add service/ tests/ requirements-service.txt
git commit -m "feat: scaffold FastAPI service directory and dependencies"
```

---

### Task 2: Configuration module

**Files:**
- Create: `service/config.py`
- Create: `tests/service/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/service/test_config.py`:
```python
import os
import pytest


def test_default_settings():
    from service.config import Settings

    s = Settings()
    assert s.ckpt_path == "./pretrained/model.safetensors"
    assert s.device == "cuda:0"
    assert s.max_concurrent == 1
    assert s.semaphore_timeout == 300.0
    assert s.fps_points == 512
    assert s.batch_size == 32
    assert s.max_faces == 200000
    assert s.iou_threshold == 0.65
    assert s.nms_threshold == 0.3
    assert s.use_graph_cut is True
    assert s.threshold_percentage_size == 0.01
    assert s.threshold_percentage_area == 0.01


def test_settings_from_env(monkeypatch):
    from service.config import Settings

    monkeypatch.setenv("PARTSAM_DEVICE", "cuda:1")
    monkeypatch.setenv("PARTSAM_MAX_CONCURRENT", "4")
    monkeypatch.setenv("PARTSAM_SEMAPHORE_TIMEOUT", "60")
    s = Settings()
    assert s.device == "cuda:1"
    assert s.max_concurrent == 4
    assert s.semaphore_timeout == 60.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/between2058/Documents/code/PartSAM && python -m pytest tests/service/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'service.config'`

- [ ] **Step 3: Write minimal implementation**

Create `service/config.py`:
```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "PARTSAM_"}

    ckpt_path: str = "./pretrained/model.safetensors"
    device: str = "cuda:0"
    max_concurrent: int = 1
    semaphore_timeout: float = 300.0
    fps_points: int = 512
    batch_size: int = 32
    max_faces: int = 200000
    iou_threshold: float = 0.65
    nms_threshold: float = 0.3
    use_graph_cut: bool = True
    threshold_percentage_size: float = 0.01
    threshold_percentage_area: float = 0.01
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/between2058/Documents/code/PartSAM && python -m pytest tests/service/test_config.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add service/config.py tests/service/test_config.py
git commit -m "feat: add config module with pydantic-settings"
```

---

### Task 3: Mesh I/O module

**Files:**
- Create: `service/mesh_io.py`
- Create: `tests/service/test_mesh_io.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/service/test_mesh_io.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/between2058/Documents/code/PartSAM && python -m pytest tests/service/test_mesh_io.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `service/mesh_io.py`:
```python
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
        mesh = trimesh.load(
            io.BytesIO(data), file_type=_EXT_TO_TYPE[ext], force="mesh"
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/between2058/Documents/code/PartSAM && python -m pytest tests/service/test_mesh_io.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add service/mesh_io.py tests/service/test_mesh_io.py
git commit -m "feat: add mesh I/O module with load and multipart GLB export"
```

---

### Task 4: Preprocessing module

**Files:**
- Create: `service/preprocessing.py`
- Create: `tests/service/test_preprocessing.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/service/test_preprocessing.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/between2058/Documents/code/PartSAM && python -m pytest tests/service/test_preprocessing.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `service/preprocessing.py`:
```python
import numpy as np
import torch
import trimesh

from utils.point import sample_surface
from utils.aug import CenterShift, NormalizeMy, NormalizeColor, ToTensor


def prepare(mesh: trimesh.Trimesh, num_points: int = 100000, seed: int = 666) -> dict:
    """Sample points from mesh and prepare tensors for model input.

    Replicates the pipeline from ValDataset.__getitem__ + collate_fn_eval + prep_points_train(eval=True).
    """
    # Sample surface points with colors and normals
    points, point_to_face, colors = sample_surface(
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/between2058/Documents/code/PartSAM && python -m pytest tests/service/test_preprocessing.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add service/preprocessing.py tests/service/test_preprocessing.py
git commit -m "feat: add preprocessing module for mesh-to-tensor pipeline"
```

---

### Task 5: Model manager module

**Files:**
- Create: `service/model_manager.py`

- [ ] **Step 1: Write the implementation**

Create `service/model_manager.py`:
```python
import asyncio
from contextlib import asynccontextmanager

import hydra
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_model

from PartSAM.utils.torch_utils import replace_with_fused_layernorm
from service.config import Settings


class ModelManager:
    def __init__(self):
        self.model = None
        self._semaphore = None
        self.settings = None

    def load(self, settings: Settings):
        self.settings = settings

        # Load model config via Hydra
        with hydra.initialize("../configs", version_base=None):
            cfg = hydra.compose(config_name="partsam")
            OmegaConf.resolve(cfg)

        self.model = hydra.utils.instantiate(cfg.model)
        self.model.apply(replace_with_fused_layernorm)
        load_model(self.model, settings.ckpt_path)
        self.model.eval()
        self.model.to(settings.device)

        self._semaphore = asyncio.Semaphore(settings.max_concurrent)

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()

    def reset_cache(self):
        """Reset cached embeddings between requests."""
        self.model.labels = None
        self.model.pc_embeddings = None
        self.model.patches = None
        self.model.pf_feat = None
        self.model.part_planes = None

    @asynccontextmanager
    async def acquire(self):
        """Acquire GPU access with timeout."""
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.settings.semaphore_timeout,
            )
        except asyncio.TimeoutError:
            raise TimeoutError("Server busy, retry later")

        try:
            self.reset_cache()
            yield self.model
        finally:
            self._semaphore.release()


model_manager = ModelManager()
```

- [ ] **Step 2: Verify syntax**

Run: `cd /Users/between2058/Documents/code/PartSAM && python -c "import ast; ast.parse(open('service/model_manager.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add service/model_manager.py
git commit -m "feat: add model manager with semaphore concurrency control"
```

---

### Task 6: Inference module

**Files:**
- Create: `service/inference.py`

- [ ] **Step 1: Write the implementation**

Create `service/inference.py`:
```python
import gc

import numpy as np
import pointops
import torch

from utils.infer_utils import nms, sort_masks_by_area, post_processing
from service.config import Settings


def auto_segment(
    model: torch.nn.Module,
    data: dict,
    settings: Settings,
) -> np.ndarray:
    """Run full auto-segmentation pipeline. Returns per-face label array.

    Extracted from evaluation/eval_everypart.py.
    """
    device = settings.device

    # Move data to device
    data_gpu = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data_gpu[k] = v.to(device, non_blocking=True)
        else:
            data_gpu[k] = v

    coord = data_gpu["coords"][0]
    coord_offset = torch.tensor([coord.shape[0]], device=device)
    new_coord_offset = torch.tensor([settings.fps_points], device=device)
    fps_idx = pointops.farthest_point_sampling(coord, coord_offset, new_coord_offset)
    prompt_labels = torch.tensor([1], dtype=torch.long, device=device).unsqueeze(0)

    masks = []
    scores = []
    batch_size = settings.batch_size

    with torch.no_grad():
        for batch_start in range(0, fps_idx.size(0), batch_size):
            batch_end = min(batch_start + batch_size, fps_idx.size(0))
            batch_indices = range(batch_start, batch_end)
            selected_indices = fps_idx[batch_start:batch_end]
            prompt_points = coord[selected_indices].unsqueeze(1)

            data_in = {
                "coords": data_gpu["coords"][0].repeat(len(batch_indices), 1, 1),
                "color": data_gpu["color"][0].repeat(len(batch_indices), 1, 1),
                "normal": data_gpu["normal"][0].repeat(len(batch_indices), 1, 1),
                "point_to_face": data_gpu["point_to_face"][0].repeat(len(batch_indices), 1),
                "vertices": data_gpu["vertices"][0].repeat(len(batch_indices), 1, 1),
                "faces": data_gpu["faces"][0].repeat(len(batch_indices), 1, 1),
                "prompt_coords": prompt_points,
                "selected_indices": selected_indices,
                "prompt_labels": prompt_labels.expand(len(batch_indices), -1),
            }

            batch_masks, batch_scores = model.predict_masks(**data_in)
            masks.append(batch_masks.cpu())
            scores.append(batch_scores.cpu())
            del batch_masks, batch_scores
            torch.cuda.empty_cache()
            gc.collect()

    masks = torch.cat(masks, dim=0)
    scores = torch.cat(scores, dim=0)
    masks = masks.reshape(-1, masks.size(2)) > 0
    scores = scores.reshape(scores.size(0) * scores.size(1), -1)

    # Filter by IoU threshold
    top_indices = (scores > settings.iou_threshold).squeeze()
    masks = masks[top_indices]
    scores = scores[top_indices]

    # Reset model cache
    model.labels = None
    model.pc_embeddings = None

    # NMS
    nms_indices = nms(masks, scores, threshold=settings.nms_threshold)
    filtered_masks = masks[nms_indices]

    # Sort and assign labels
    sorted_masks = sort_masks_by_area(filtered_masks)
    labels = torch.full((sorted_masks.size(1),), -1)
    for i in range(len(filtered_masks)):
        labels[sorted_masks[i]] = i

    # Map point labels to face labels via voting
    face_labels = _point_labels_to_face_labels(
        labels,
        data_gpu["point_to_face"][0].cpu().numpy(),
        data_gpu["vertices"][0].squeeze(0).cpu().numpy(),
        data_gpu["faces"][0].squeeze(0).cpu().numpy(),
        device,
    )

    # Post-processing
    import trimesh

    mesh = trimesh.Trimesh(
        vertices=data_gpu["vertices"][0].squeeze(0).cpu().numpy(),
        faces=data_gpu["faces"][0].squeeze(0).cpu().numpy(),
    )
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)
    mesh = post_processing(face_labels, mesh, settings)

    # Extract final face labels from colored mesh
    return _extract_labels_from_colored_mesh(mesh)


def prompt_segment(
    model: torch.nn.Module,
    data: dict,
    points: list[list[float]],
    labels: list[int],
    device: str,
) -> np.ndarray:
    """Run prompt-based segmentation. Returns binary per-face label array (0=remainder, 1=selected)."""
    # Move data to device
    data_gpu = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data_gpu[k] = v.to(device, non_blocking=True)
        else:
            data_gpu[k] = v

    prompt_coords = torch.tensor(points, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
    prompt_labels_t = torch.tensor(labels, dtype=torch.long, device=device).unsqueeze(0)

    # If multiple prompt points, reshape appropriately
    if len(points) > 1:
        prompt_coords = torch.tensor(points, dtype=torch.float32, device=device).unsqueeze(0)
        prompt_labels_t = torch.tensor(labels, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        data_in = {
            "coords": data_gpu["coords"],
            "color": data_gpu["color"],
            "normal": data_gpu["normal"],
            "point_to_face": data_gpu["point_to_face"],
            "vertices": data_gpu["vertices"],
            "faces": data_gpu["faces"],
            "prompt_coords": prompt_coords,
            "selected_indices": torch.tensor([0], device=device),
            "prompt_labels": prompt_labels_t,
        }
        masks, iou_preds = model.predict_masks(**data_in)

    # Select the mask with highest IoU prediction
    masks = masks.squeeze(0)  # [num_outputs, N]
    iou_preds = iou_preds.squeeze(0)  # [num_outputs]
    best_idx = iou_preds.argmax()
    best_mask = (masks[best_idx] > 0).cpu().numpy()

    # Map point mask to face labels via voting
    point_to_face = data_gpu["point_to_face"][0].cpu().numpy()
    num_faces = data_gpu["faces"][0].squeeze(0).shape[0]
    face_votes = np.zeros(num_faces, dtype=np.int32)
    face_counts = np.zeros(num_faces, dtype=np.int32)

    for i, face_idx in enumerate(point_to_face):
        face_counts[face_idx] += 1
        if best_mask[i]:
            face_votes[face_idx] += 1

    # Face is selected if majority of its points are in the mask
    face_labels = (face_votes > face_counts / 2).astype(np.int32)

    # Reset model cache
    model.labels = None
    model.pc_embeddings = None

    return face_labels


def _point_labels_to_face_labels(
    labels: torch.Tensor,
    face_index: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    device: str,
) -> np.ndarray:
    """Map point-level labels to face-level labels via voting + KNN fill."""
    num_faces = len(faces)
    num_labels = int(labels.max()) + 1
    votes = np.zeros((num_faces, num_labels), dtype=np.int32)
    np.add.at(votes, (face_index, labels.numpy()), 1)
    max_votes_labels = np.argmax(votes, axis=1)
    max_votes_labels[np.all(votes == 0, axis=1)] = -1

    # KNN fill for unlabeled faces
    valid_mask = max_votes_labels != -1
    import trimesh as _trimesh

    mesh_temp = _trimesh.Trimesh(vertices=vertices, faces=faces)
    face_centroids = mesh_temp.triangles_center
    coord = torch.tensor(face_centroids, device=device).contiguous().float()
    valid_coord = coord[valid_mask]
    valid_offset = torch.tensor([valid_coord.shape[0]], device=device)
    invalid_coord = coord[~valid_mask]
    invalid_offset = torch.tensor([invalid_coord.shape[0]], device=device)

    if invalid_coord.shape[0] > 0:
        indices, _ = pointops.knn_query(
            1, valid_coord, valid_offset, invalid_coord, invalid_offset
        )
        indices = indices[:, 0].cpu().numpy()
        max_votes_labels[~valid_mask] = max_votes_labels[valid_mask][indices]

    return max_votes_labels


def _extract_labels_from_colored_mesh(mesh) -> np.ndarray:
    """Extract per-face labels from a post-processed colored mesh.

    After post_processing, each face has a unique color per label.
    We convert face colors back to integer labels.
    """
    face_colors = mesh.visual.face_colors[:, :3]  # drop alpha
    unique_colors, inverse = np.unique(face_colors, axis=0, return_inverse=True)
    return inverse
```

- [ ] **Step 2: Verify syntax**

Run: `cd /Users/between2058/Documents/code/PartSAM && python -c "import ast; ast.parse(open('service/inference.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add service/inference.py
git commit -m "feat: add inference module with auto and prompt segmentation"
```

---

### Task 7: FastAPI application

**Files:**
- Create: `service/app.py`
- Create: `tests/service/test_app.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/service/test_app.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/between2058/Documents/code/PartSAM && python -m pytest tests/service/test_app.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `service/app.py`:
```python
import asyncio
import json
import os
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import Response

from service.config import Settings
from service.inference import auto_segment, prompt_segment
from service.mesh_io import SUPPORTED_EXTENSIONS, load_mesh, to_multipart_glb
from service.model_manager import model_manager
from service.preprocessing import prepare


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    model_manager.load(settings)
    yield
    model_manager.unload()


app = FastAPI(title="PartSAM", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model_manager.model is not None,
        "gpu_available": torch.cuda.is_available(),
        "queue_size": (
            model_manager.settings.max_concurrent - model_manager._semaphore._value
            if model_manager._semaphore
            else 0
        ),
        "max_concurrent": (
            model_manager.settings.max_concurrent if model_manager.settings else 0
        ),
    }


def _get_extension(filename: str) -> str:
    _, ext = os.path.splitext(filename)
    return ext.lower()


def _validate_file(filename: str, file_bytes: bytes) -> None:
    ext = _get_extension(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}', accept: .glb, .ply, .obj",
        )


@app.post("/segment/auto")
async def segment_auto(
    file: UploadFile = File(...),
    iou_threshold: float = Query(0.65, ge=0.0, le=1.0),
    nms_threshold: float = Query(0.3, ge=0.0, le=1.0),
    use_graph_cut: bool = Query(True),
):
    file_bytes = await file.read()
    ext = _get_extension(file.filename)
    _validate_file(file.filename, file_bytes)

    try:
        mesh = load_mesh(file_bytes, ext)
    except ValueError as e:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail=str(e))

    if len(mesh.faces) > model_manager.settings.max_faces:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=413,
            detail=f"Mesh too large, max {model_manager.settings.max_faces} faces",
        )

    # Override per-request settings
    settings = model_manager.settings.model_copy()
    settings.iou_threshold = iou_threshold
    settings.nms_threshold = nms_threshold
    settings.use_graph_cut = use_graph_cut

    try:
        async with model_manager.acquire() as model:
            data = await asyncio.to_thread(prepare, mesh)
            face_labels = await asyncio.to_thread(auto_segment, model, data, settings)
    except TimeoutError:
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="Server busy, retry later")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        from fastapi import HTTPException

        raise HTTPException(
            status_code=503,
            detail="GPU out of memory, try smaller mesh or retry later",
        )

    glb_bytes = await asyncio.to_thread(to_multipart_glb, mesh, face_labels, "auto")
    return Response(content=glb_bytes, media_type="application/octet-stream")


@app.post("/segment/prompt")
async def segment_prompt(
    file: UploadFile = File(...),
    points: str = Form(...),
    labels: str = Form(...),
):
    file_bytes = await file.read()
    ext = _get_extension(file.filename)
    _validate_file(file.filename, file_bytes)

    # Parse JSON fields
    try:
        points_list = json.loads(points)
        labels_list = json.loads(labels)
    except json.JSONDecodeError:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=422, detail="points and labels must be valid JSON arrays"
        )

    if len(points_list) != len(labels_list):
        from fastapi import HTTPException

        raise HTTPException(
            status_code=422, detail="points and labels must have the same length"
        )

    try:
        mesh = load_mesh(file_bytes, ext)
    except ValueError as e:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail=str(e))

    if len(mesh.faces) > model_manager.settings.max_faces:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=413,
            detail=f"Mesh too large, max {model_manager.settings.max_faces} faces",
        )

    try:
        async with model_manager.acquire() as model:
            data = await asyncio.to_thread(prepare, mesh)
            face_labels = await asyncio.to_thread(
                prompt_segment,
                model,
                data,
                points_list,
                labels_list,
                model_manager.settings.device,
            )
    except TimeoutError:
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="Server busy, retry later")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        from fastapi import HTTPException

        raise HTTPException(
            status_code=503,
            detail="GPU out of memory, try smaller mesh or retry later",
        )

    glb_bytes = await asyncio.to_thread(to_multipart_glb, mesh, face_labels, "prompt")
    return Response(content=glb_bytes, media_type="application/octet-stream")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/between2058/Documents/code/PartSAM && python -m pytest tests/service/test_app.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add service/app.py tests/service/test_app.py
git commit -m "feat: add FastAPI app with /segment/auto, /segment/prompt, /health endpoints"
```

---

### Task 8: Run entry point and smoke test

**Files:**
- No new files; verify everything works end-to-end

- [ ] **Step 1: Run all unit tests**

Run: `cd /Users/between2058/Documents/code/PartSAM && python -m pytest tests/service/ -v`
Expected: all tests pass

- [ ] **Step 2: Verify server starts (syntax/import check)**

Run: `cd /Users/between2058/Documents/code/PartSAM && python -c "from service.app import app; print('Import OK')"`
Expected: `Import OK`

- [ ] **Step 3: Document how to run**

Add a comment at the top of `service/app.py` after the module docstring:

Update `service/app.py` — add at line 1:
```python
"""PartSAM FastAPI Microservice.

Run with:
    uvicorn service.app:app --host 0.0.0.0 --port 8000

Environment variables:
    PARTSAM_CKPT_PATH       Model checkpoint path (default: ./pretrained/model.safetensors)
    PARTSAM_DEVICE           GPU device (default: cuda:0)
    PARTSAM_MAX_CONCURRENT   Max concurrent inferences (default: 1)
    PARTSAM_SEMAPHORE_TIMEOUT  Queue timeout in seconds (default: 300)
    PARTSAM_MAX_FACES        Max mesh face count (default: 200000)
"""
```

- [ ] **Step 4: Commit**

```bash
git add service/app.py
git commit -m "feat: add run instructions and finalize service"
```

---

## Execution Notes

- **Tasks 1-4** are independent of GPU — tests run on CPU with synthetic meshes.
- **Task 5-6** (model_manager, inference) contain GPU-dependent code extracted from `eval_everypart.py`. They are verified via syntax check + the mocked integration tests in Task 7.
- **Task 7** tests use mocked model/inference, so they run without GPU.
- **Task 8** is the final integration verification.
