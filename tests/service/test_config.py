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
