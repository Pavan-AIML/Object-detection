from typing import Tuple
import torch


def extract_bbox_from_heatmap(
    heatmap: torch.Tensor, threshold: float, patch_size: Tuple[int, int], scale: int, stride: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = heatmap > threshold

    indices = mask.nonzero()

    if indices.shape[0] == 0:
        return None, None

    bboxes = []
    scores = []
    for idx in indices:
        bbox = torch.tensor(
            [
                idx[1] * stride * scale,
                idx[0] * stride * scale,
                (idx[1] * stride + patch_size[1]) * scale,
                (idx[0] * stride + patch_size[0]) * scale,
            ]
        )
        bboxes.append(bbox)
        scores.append(heatmap[idx[0], idx[1]])

    return torch.stack(bboxes), torch.stack(scores)
