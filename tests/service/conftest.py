"""Stub unavailable native/optional modules so service modules can be imported in tests."""
import sys
import types
from unittest.mock import MagicMock


def _make_mock_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__spec__ = None
    return mod


# Stub pointops (C extension not available in test env)
_pointops = _make_mock_module("pointops")
_pointops.farthest_point_sampling = MagicMock()
_pointops.knn_query = MagicMock()
sys.modules.setdefault("pointops", _pointops)

# Stub hydra
_hydra = _make_mock_module("hydra")
_hydra.initialize = MagicMock()
_hydra.compose = MagicMock()
_hydra.utils = MagicMock()
sys.modules.setdefault("hydra", _hydra)

# Stub omegaconf
_omegaconf = _make_mock_module("omegaconf")
_omegaconf.OmegaConf = MagicMock()
sys.modules.setdefault("omegaconf", _omegaconf)

# Stub safetensors and sub-modules
_safetensors = _make_mock_module("safetensors")
_safetensors_torch = _make_mock_module("safetensors.torch")
_safetensors_torch.load_model = MagicMock()
_safetensors.torch = _safetensors_torch
sys.modules.setdefault("safetensors", _safetensors)
sys.modules.setdefault("safetensors.torch", _safetensors_torch)

# Stub utils.infer_utils (depends on hydra at import time)
# NOTE: do NOT stub "utils" itself — it's a real package on disk (utils/aug.py etc.)
_infer_utils = _make_mock_module("utils.infer_utils")
_infer_utils.nms = MagicMock()
_infer_utils.sort_masks_by_area = MagicMock()
_infer_utils.post_processing = MagicMock()
sys.modules.setdefault("utils.infer_utils", _infer_utils)

# Stub PartSAM.utils.torch_utils (depends on fused ops)
_partsam_utils = _make_mock_module("PartSAM.utils.torch_utils")
_partsam_utils.replace_with_fused_layernorm = MagicMock()
sys.modules.setdefault("PartSAM.utils.torch_utils", _partsam_utils)
