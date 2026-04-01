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
