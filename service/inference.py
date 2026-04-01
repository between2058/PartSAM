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
