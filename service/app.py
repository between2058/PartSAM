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

import asyncio
import json
import os
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, File, Form, Query, UploadFile, HTTPException
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


def _get_extension(filename: str) -> str:
    _, ext = os.path.splitext(filename)
    return ext.lower()


def _validate_file(filename: str) -> None:
    ext = _get_extension(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}', accept: .glb, .ply, .obj",
        )


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


@app.post("/segment/auto")
async def segment_auto(
    file: UploadFile = File(...),
    iou_threshold: float = Query(0.65, ge=0.0, le=1.0),
    nms_threshold: float = Query(0.3, ge=0.0, le=1.0),
    use_graph_cut: bool = Query(True),
):
    file_bytes = await file.read()
    _validate_file(file.filename)
    ext = _get_extension(file.filename)

    try:
        mesh = load_mesh(file_bytes, ext)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if len(mesh.faces) > model_manager.settings.max_faces:
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
        raise HTTPException(status_code=503, detail="Server busy, retry later")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
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
    _validate_file(file.filename)
    ext = _get_extension(file.filename)

    # Parse JSON fields
    try:
        points_list = json.loads(points)
        labels_list = json.loads(labels)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=422, detail="points and labels must be valid JSON arrays"
        )

    if len(points_list) != len(labels_list):
        raise HTTPException(
            status_code=422, detail="points and labels must have the same length"
        )

    try:
        mesh = load_mesh(file_bytes, ext)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if len(mesh.faces) > model_manager.settings.max_faces:
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
        raise HTTPException(status_code=503, detail="Server busy, retry later")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(
            status_code=503,
            detail="GPU out of memory, try smaller mesh or retry later",
        )

    glb_bytes = await asyncio.to_thread(to_multipart_glb, mesh, face_labels, "prompt")
    return Response(content=glb_bytes, media_type="application/octet-stream")
