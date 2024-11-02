from typing import Tuple

from PIL import Image
import torch
from torchvision.transforms import functional as F

from exercise_code.data.mot_obj_detect import MOT16ObjDetect
from exercise_code.model.compute_iou import compute_iou


class MOT16HoG(MOT16ObjDetect):
    """Classification dataset class for the Multiple Object Tracking Dataset

    Args:
        root (string): Root directory of dataset
        num_patches (int): The total number of patches per idx, sum of positive and negative samples
        patch_size (Tuple[int, int]): 
        random_offset (bool, optional): A random offset to bounding boxes of positive samples
        transforms (callable, optional): A function/transform that takes in the target and transforms it.
        vis_threshold (float, optional): threshhold for labeled visibility value, only values higher than threshold are positive samples

    """
    def __init__(
        self,
        root,
        num_patches: int = 50,
        patch_size: Tuple[int, int] = (128, 64),
        random_offset: bool = True,
        transforms = None,
        vis_threshold = 0.5,
    ):
        super().__init__(root, transforms, vis_threshold)
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.random_offset = random_offset

    def get_whole_image(self, idx) -> torch.Tensor:
        # load images ad masks
        img_path = self._img_paths[idx]
        # mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = F.to_tensor(Image.open(img_path).convert("RGB"))

        return img

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self._img_paths[idx]
        # mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")

        target = self._get_annotation(idx)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # get all positive bboxes of idx
        bboxes = target["boxes"][target["visibilities"].gt(self._vis_threshold)].int()
        pos_patches = []
        for bbox in bboxes[: min(self.num_patches, bboxes.shape[0])]:
            ratio = self.patch_size[0] / self.patch_size[1]
            bbox_height = bbox[3] - bbox[1]
            bbox_width = bbox[2] - bbox[0]
            h_padding = torch.clip(bbox_width * ratio - bbox_height, 0)
            w_padding = torch.clip(bbox_height / ratio - bbox_width, 0)
            top = bbox[1] - int(h_padding / 2.0)
            left = bbox[0] - int(w_padding / 2.0)
            if self.random_offset:
                top = top + torch.randint(-(bbox_height).item() // 20, (bbox_height).item() // 20, (1, 1)).item()
                left = left + torch.randint(-(bbox_width).item() // 20, (bbox_width).item() // 20, (1, 1)).item()
            top = min(max(0, top), img.shape[-2] - (bbox_height + int(h_padding)))
            left = min(max(0, left), img.shape[-1] - (bbox_width + int(w_padding)))

            pos_patches.append(
                F.resized_crop(
                    img,
                    top,
                    left,
                    bbox_height + int(h_padding),
                    bbox_width + int(w_padding),
                    [self.patch_size[0], self.patch_size[1]],
                )
            )

        # get random negative bboxes of idx, their overlap with any labeled bbox should be smaller than neg_th
        neg_th = 0.25
        neg_patches = []
        for _ in range(self.num_patches - len(pos_patches)):
            while True:
                scale = torch.rand(1).item() * 2.0 + 1.0

                top = torch.randint(0, img.shape[-2] - int(self.patch_size[0] * scale), (1,)).item()
                left = torch.randint(0, img.shape[-1] - int(self.patch_size[1] * scale), (1,)).item()

                bbox = torch.Tensor(
                    [left, top, left + int(self.patch_size[1] * scale), top + int(self.patch_size[0] * scale)]
                )
                if torch.any(compute_iou(target["boxes"], bbox[None].expand(target["boxes"].shape[0], -1)) > neg_th):
                    continue

                neg_patches.append(
                    F.resized_crop(
                        img,
                        top,
                        left,
                        int(self.patch_size[0] * scale),
                        int(self.patch_size[1] * scale),
                        [self.patch_size[0], self.patch_size[1]],
                    )
                )
                break

        patches = torch.stack(pos_patches + neg_patches, dim=0)
        labels = torch.zeros(patches.shape[0])
        labels[: len(pos_patches)] = 1.0

        output = {"patches": patches, "patch_labels": labels}
        # target["patches"] = patches
        # target["patch_labels"] = labels
        # return img, output
        return torch.Tensor([0.0]), output

    def __len__(self):
        return len(self._img_paths)
