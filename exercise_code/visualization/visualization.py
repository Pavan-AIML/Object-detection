# __all__ = ("gradients", "bboxes")

from math import cos, sin

import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# from skimage.feature import hog
import torch

dpi = 96


def gradients(image: torch.Tensor, gradient_norm: torch.Tensor, gradient_orientation: torch.Tensor):
    fig, axs = plt.subplots(1, 3, dpi=dpi, figsize=(10, 15))

    hsvimg = torch.ones([gradient_norm.shape[0], gradient_norm.shape[1], 3])
    hsvimg[:, :, 0] = gradient_orientation
    hsvimg[:, :, 1] = torch.clip(gradient_norm, 0.0, 1.0)
    rgbimg = cv2.cvtColor(np.float32(hsvimg.cpu().numpy()), cv2.COLOR_HSV2RGB)

    axs[0].imshow(image.mul(255).permute(1, 2, 0).byte().cpu().numpy(), cmap="gray", interpolation="none")
    axs[0].set_title("Image", fontsize=20)
    axs[0].axis("off")

    axs[1].imshow(gradient_norm.mul(255).byte().cpu().numpy(), cmap="gray", interpolation="none")
    axs[1].set_title("Gradient Norm", fontsize=20)
    axs[1].axis("off")

    # axs[2].imshow(gradient_orientation.div(180).mul(255).byte().cpu().numpy(), cmap="hsv", interpolation="none")
    axs[2].imshow(rgbimg, cmap="hsv", interpolation="none")
    axs[2].set_title("Gradient Orientation", fontsize=20)
    axs[2].axis("off")

    plt.show()


def history_of_oriented_gradients(image: torch.Tensor, hog_bins: torch.Tensor):
    # image 3, H, W
    # hog_bins H/8, W/8, 9
    hog_bin_max = torch.max(hog_bins).item()
    scaling = image.shape[-2] // hog_bins.shape[-3]
    radius = scaling / 2.0

    fig, axs = plt.subplots(1, 2, dpi=dpi, figsize=(10, 10))

    axs[0].imshow(image.mul(255).permute(1, 2, 0).byte().cpu().numpy(), cmap="gray", interpolation="none")
    axs[0].set_title("Image", fontsize=20)
    axs[0].axis("off")

    for i, hog_bins_col in enumerate(hog_bins):
        for j, hog_bin in enumerate(hog_bins_col):
            center = (j * scaling + scaling // 2 - 0.5, i * scaling + scaling // 2 - 0.5)
            # hog_bin_max = torch.max(hog_bin).item()
            # print(center)
            for orientation, h_bin in enumerate(hog_bin):
                orientation_angle = orientation * torch.pi / hog_bins.shape[-1]
                start = (center[0] - cos(orientation_angle) * radius, center[1] - sin(orientation_angle) * radius)
                end = (center[0] + cos(orientation_angle) * radius, center[1] + sin(orientation_angle) * radius)
                # axs[1].plot(start, end, color="white", linewidth=2.0, alpha=(h_bin / hog_bin_max).item())
                axs[1].plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    color="white",
                    linewidth=2.0,
                    alpha=(h_bin / hog_bin_max).item(),
                )
    axs[1].imshow(torch.zeros_like(image).permute(1, 2, 0).byte().cpu().numpy(), cmap="gray", interpolation="none")
    axs[1].set_title("Histogram of Oriented Gradients", fontsize=20)
    axs[1].axis("off")

    plt.show()

def heatmap_objectness(image: torch.Tensor, heatmap: torch.Tensor):
    # image 3, H, W
    # heatmap H_, W_

    fig, axs = plt.subplots(1, 2, dpi=dpi, figsize=(10, 10))

    axs[0].imshow(image.mul(255).permute(1, 2, 0).byte().cpu().numpy(), cmap="gray", interpolation="none")
    axs[0].set_title("Image", fontsize=20)
    axs[0].axis("off")

    axs[1].imshow(heatmap.mul(255).byte().cpu().numpy(), cmap="gray", interpolation="none")
    axs[1].set_title("Heatmap", fontsize=20)
    axs[1].axis("off")

    plt.show()

def dataset(patches: torch.Tensor, labels: torch.Tensor, rows: int, cols: int):
    linewidth = 3
    assert patches.shape[0] == rows * cols
    assert labels.shape[0] == rows * cols
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 10 * rows))

    for ax, patch, label in zip(axs.reshape(-1), patches, labels):
        color = "green" if label else "red"
        ax.imshow(patch.mul(255).permute(1, 2, 0).byte().cpu().numpy(), cmap="gray", interpolation="none")
        ax.axis("off")
        rect = Rectangle(
            (0, 0),
            patch.shape[2],
            patch.shape[1],
            linewidth=10,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)


def bboxes(image: torch.Tensor, bboxes: torch.Tensor) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.imshow(image.mul(255).permute(1, 2, 0).byte().cpu().numpy(), cmap="gray")
    if bboxes is not None:
        for bbox in bboxes:
            # ltrb
            rect = Rectangle(
                (bbox[0].item(), bbox[1].item()),
                (bbox[2] - bbox[0]).item(),
                (bbox[3] - bbox[1]).item(),
                linewidth=2,
                edgecolor="black",
                facecolor="none",
            )
            ax.add_patch(rect)
        ax.set_title("Image with predictions", fontsize=20)
        ax.axis("off")
