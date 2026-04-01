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
