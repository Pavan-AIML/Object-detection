import torch

from exercise_code.model.compute_image_gradient import compute_image_gradient
from exercise_code.model.fill_hog_bins import fill_hog_bins
from exercise_code.model.utils import blockify_tensor


def HoG(images: torch.Tensor, block_size: int = 8, num_bins: int = 9) -> torch.Tensor:
    gradient_norm, gradient_angle = compute_image_gradient(images)  # B, H, W
    gradient_angle[gradient_angle >= 180.0] = gradient_angle[gradient_angle >= 180.0] - 180.0
    B, H_, W_ = blockify_tensor(gradient_norm, block_size).shape[:3] # B, H_, W_, block_size, block_size
    hog_bins = fill_hog_bins(
        blockify_tensor(gradient_norm, block_size).flatten(0, -3).flatten(-2, -1), # B*H_*W_ , N
        blockify_tensor(gradient_angle, block_size).flatten(0, -3).flatten(-2, -1),  # B*H_*W_ , N
        num_bins,
    ) # B*H_*W_ , num_bins
    hog_bins = hog_bins.reshape(B, H_, W_, -1) # B, H_, W_ , num_bins

    hog_bins = torch.cat(
        [hog_bins[:, :-1, :-1], hog_bins[:, 1:, :-1], hog_bins[:, :-1, 1:], hog_bins[:, 1:, 1:]], dim=-1
    ) # B, H_-1, W_-1 , num_bins*4

    hog_bins = (hog_bins / (torch.linalg.norm(hog_bins, dim=-1, keepdim=True) + 1.0e-5)).flatten(1, -1) # B, H_-1 * W_-1 * num_bins*4


    return hog_bins # B, F
