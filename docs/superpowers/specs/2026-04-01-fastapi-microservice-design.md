# PartSAM FastAPI Microservice Design

## Overview

A synchronous FastAPI microservice wrapping the PartSAM 3D part segmentation model. Internal service supporting high concurrency via GPU semaphore queuing. Accepts 3D mesh files (.glb, .ply, .obj), returns multipart .glb with each segmented part as an independent scene node.

## API Endpoints

### `POST /segment/auto`

Fully automatic part segmentation. Uses FPS to generate prompt points, runs batch inference, NMS, and post-processing.

**Request:** `multipart/form-data`
- `file` (required): 3D mesh file (.glb, .ply, .obj)

**Query parameters:**
| Parameter | Type | Default | Description |
|---|---|---|---|
| `iou_threshold` | float | 0.65 | Confidence threshold for mask filtering |
| `nms_threshold` | float | 0.3 | NMS overlap threshold |
| `use_graph_cut` | bool | true | Enable graph-cut smoothing (slower but cleaner) |

**Response:** `application/octet-stream` — multipart .glb file. Each part is a separate node named `part_0`, `part_1`, etc.

### `POST /segment/prompt`

Interactive prompt-based segmentation. User specifies 3D point(s) on the mesh, model returns the corresponding part mask.

**Request:** `multipart/form-data`
- `file` (required): 3D mesh file (.glb, .ply, .obj)
- `points` (required): JSON string, array of [x, y, z] coordinates, e.g. `[[0.1, 0.2, 0.3]]`
- `labels` (required): JSON string, array of ints (1=positive, 0=negative), e.g. `[1]`

**Response:** `application/octet-stream` — .glb file with 2 nodes: `selected_part` (the prompted segment) and `remainder` (everything else).

### `GET /health`

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "gpu_available": true,
  "queue_size": 0,
  "max_concurrent": 1
}
```

## Architecture

### Approach

Single-process FastAPI with `asyncio.Semaphore` controlling GPU access. Model loaded once at startup, held in memory. Scale horizontally by running multiple instances behind a load balancer, each bound to a different GPU.

### Module Structure

```
service/
├── app.py              # FastAPI app, route definitions, lifespan (model load/shutdown)
├── config.py           # Settings via pydantic-settings (env vars + Hydra config defaults)
├── model_manager.py    # Model lifecycle: load at startup, Semaphore for concurrency control
├── inference.py        # Core inference logic extracted from eval_everypart.py
│   ├── auto_segment()     — full auto pipeline (FPS → batch predict → NMS → post-process)
│   └── prompt_segment()   — prompt-based single prediction
├── mesh_io.py          # Mesh I/O: load any format → trimesh, build multipart GLB output
└── preprocessing.py    # Point sampling, normalization (extracted from ValDataset)
```

### Data Flow

```
Request (file + params)
  → mesh_io.load(file_bytes, extension) → trimesh.Trimesh
  → preprocessing.prepare(mesh) → coords, colors, normals, point_to_face, vertices, faces
  → model_manager.acquire() → wait on Semaphore for GPU access
  → inference.auto_segment() or inference.prompt_segment()
  → face_labels (numpy array)
  → mesh_io.to_multipart_glb(mesh, face_labels) → bytes
  → Response
```

### Module Responsibilities

**app.py**
- FastAPI app with lifespan handler for model loading/cleanup
- Route handlers: validate input, call inference, stream response
- No business logic — delegates everything to other modules

**config.py**
- `Settings` class using pydantic-settings
- Reads from environment variables with sensible defaults:
  - `PARTSAM_CKPT_PATH` (default: `./pretrained/model.safetensors`)
  - `PARTSAM_DEVICE` (default: `cuda:0`)
  - `PARTSAM_MAX_CONCURRENT` (default: `1`)
  - `PARTSAM_SEMAPHORE_TIMEOUT` (default: `300` seconds)
  - `PARTSAM_FPS_POINTS` (default: `512`)
  - `PARTSAM_BATCH_SIZE` (default: `32`)
  - `PARTSAM_MAX_FACES` (default: `200000`)

**model_manager.py**
- Loads PartSAM model + weights at startup using Hydra config + safetensors
- Applies `replace_with_fused_layernorm`
- Holds model in eval mode on configured device
- Provides `acquire()` context manager wrapping `asyncio.Semaphore` with timeout
- Resets model cache (`pc_embeddings`, `labels`, etc.) before each request

**inference.py**
- `auto_segment(model, data, config) -> numpy.ndarray`:
  1. FPS to select prompt points
  2. Batch predict masks (loop over batches, same logic as eval_everypart.py)
  3. IoU threshold + NMS filtering
  4. Sort masks by area, assign labels
  5. Face voting (point labels → mesh face labels)
  6. KNN fill for unlabeled faces
  7. Post-processing (smooth, split, optional graph-cut)
  8. Returns per-face label array

- `prompt_segment(model, data, points, labels) -> numpy.ndarray`:
  1. Encode PC once
  2. Run predict_masks with user-provided prompt_coords and prompt_labels
  3. Select best mask (highest IoU score)
  4. Threshold mask → binary
  5. Map to face labels via voting
  6. Returns binary per-face label array (0=remainder, 1=selected)

**mesh_io.py**
- `load(file_bytes, extension) -> trimesh.Trimesh`: Load mesh from bytes, normalize to unit cube (center + scale, same as ValDataset)
- `to_multipart_glb(mesh, face_labels) -> bytes`:
  - For auto: iterate unique labels, extract submesh per label, build trimesh.Scene with `part_N` nodes
  - For prompt: extract selected faces as `selected_part`, rest as `remainder`
  - Export scene as GLB bytes

**preprocessing.py**
- `prepare(mesh) -> dict`: Sample 100K surface points with colors and normals (reuse `sample_surface`), apply normalization transforms (CenterShift, NormalizeMy, NormalizeColor, ToTensor), return tensors ready for model input
- Extracted from `ValDataset.__getitem__` + `collate_fn_eval` + `prep_points_train(eval=True)`

### Concurrency Model

- `asyncio.Semaphore(N)` where N = `PARTSAM_MAX_CONCURRENT` (default 1)
- Route handlers are `async def`, inference runs in thread via `asyncio.to_thread()` to avoid blocking the event loop
- Semaphore timeout: requests waiting longer than `PARTSAM_SEMAPHORE_TIMEOUT` get 503
- Model state (`pc_embeddings`, `labels`, etc.) reset before each inference call to prevent cross-request contamination

### Horizontal Scaling

No code changes needed. Deploy multiple instances:
```
Instance 0: PARTSAM_DEVICE=cuda:0, port=8000
Instance 1: PARTSAM_DEVICE=cuda:1, port=8001
...
Load balancer (nginx/k8s) → round-robin across instances
```

## Error Handling

| Condition | HTTP Status | Message |
|---|---|---|
| Unsupported file format | 400 | `Unsupported format, accept: .glb, .ply, .obj` |
| Mesh parse failure | 400 | `Failed to parse mesh file` |
| Mesh faces > max limit | 413 | `Mesh too large, max {N} faces` |
| Prompt points out of bounds | 400 | `Prompt points out of mesh bounding box` |
| Missing/invalid points or labels | 422 | Standard FastAPI validation error |
| GPU OOM | 503 | `GPU out of memory, try smaller mesh or retry later` |
| Semaphore timeout | 503 | `Server busy, retry later` |

GPU OOM is caught by wrapping inference in try/except for `torch.cuda.OutOfMemoryError`, followed by `torch.cuda.empty_cache()`.

## Dependencies (additions to existing project)

- `fastapi`
- `uvicorn`
- `pydantic-settings`
- `python-multipart` (for file uploads)

All other dependencies (torch, trimesh, hydra, pointops, etc.) are already in the project.

## Output Format: Multipart GLB

```python
scene = trimesh.Scene()
for label_id in unique_labels:
    face_mask = (face_labels == label_id)
    submesh = mesh.submesh([face_mask], append=True)
    scene.add_geometry(submesh, node_name=f"part_{label_id}")
return scene.export(file_type="glb")
```

- Each part preserves original vertex positions, normals, and vertex colors
- Node naming: `part_0`, `part_1`, ... for auto mode; `selected_part`, `remainder` for prompt mode
- Downstream consumers can iterate GLB nodes to access individual parts
