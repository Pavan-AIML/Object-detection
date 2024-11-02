import configparser
import csv
import os
import os.path as osp

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from torchvision.transforms import functional as F


class MOT16ObjDetect(Dataset):
    """Object Detection and Tracking class for the Multiple Object Tracking Dataset"""

    def __init__(self, root, transforms=None, vis_threshold=0.25):
        self.root = root
        self.transforms = transforms
        self._vis_threshold = vis_threshold
        self._classes = ("background", "pedestrian")
        self._img_paths = []

        def listdir_nohidden(path):
            for f in os.listdir(path):
                if not f.startswith("."):
                    yield f

        for f in listdir_nohidden(root):
            path = os.path.join(root, f)
            config_file = os.path.join(path, "seqinfo.ini")

            assert os.path.exists(config_file), "Path does not exist: {}".format(config_file)

            config = configparser.ConfigParser()
            config.read(config_file)
            seq_len = int(config["Sequence"]["seqLength"])
            im_ext = config["Sequence"]["imExt"]
            im_dir = config["Sequence"]["imDir"]

            _imDir = os.path.join(path, im_dir)

            for i in range(1, seq_len + 1):
                img_path = os.path.join(_imDir, f"{i:06d}{im_ext}")
                assert os.path.exists(img_path), "Path does not exist: {img_path}"
                # self._img_paths.append((img_path, im_width, im_height))
                self._img_paths.append(img_path)

    @property
    def num_classes(self):
        return len(self._classes)

    def _get_annotation(self, idx):
        """ """

        # if 'test' in str(self.root):

        #     num_objs = 0
        #     boxes = torch.zeros((num_objs, 4), dtype=torch.float32)

        #     return {'boxes': boxes,
        #         'labels': torch.ones((num_objs,), dtype=torch.int64),
        #         'image_id': torch.tensor([idx]),
        #         'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
        #         'iscrowd': torch.zeros((num_objs,), dtype=torch.int64),}

        img_path = self._img_paths[idx]
        file_index = int(os.path.basename(img_path).split(".")[0])

        gt_file = os.path.join(os.path.dirname(os.path.dirname(img_path)), "gt", "gt.txt")

        assert os.path.exists(gt_file), "GT file does not exist: {}".format(gt_file)

        bounding_boxes = []

        with open(gt_file, "r") as inf:
            reader = csv.reader(inf, delimiter=",")
            for row in reader:
                visibility = float(row[8])
                if (
                    int(row[0]) == file_index
                    and int(row[6]) == 1
                    and int(row[7]) == 1
                    and visibility >= self._vis_threshold
                ):
                    bb = {}
                    bb["bb_left"] = int(row[2])
                    bb["bb_top"] = int(row[3])
                    bb["bb_width"] = int(row[4])
                    bb["bb_height"] = int(row[5])
                    bb["visibility"] = float(row[8])

                    bounding_boxes.append(bb)

        num_objs = len(bounding_boxes)

        boxes = torch.zeros((num_objs, 4), dtype=torch.float32)
        visibilities = torch.zeros((num_objs), dtype=torch.float32)

        for i, bb in enumerate(bounding_boxes):
            # Make pixel indexes 0-based, should already be 0-based (or not)
            x1 = bb["bb_left"] - 1
            y1 = bb["bb_top"] - 1
            # This -1 accounts for the width (width of 1 x1=x2)
            x2 = x1 + bb["bb_width"] - 1
            y2 = y1 + bb["bb_height"] - 1

            boxes[i, 0] = x1
            boxes[i, 1] = y1
            boxes[i, 2] = x2
            boxes[i, 3] = y2
            visibilities[i] = bb["visibility"]

        return {
            "boxes": boxes,
            "labels": torch.ones((num_objs,), dtype=torch.int64),
            "image_id": torch.tensor(idx),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((num_objs,), dtype=torch.int64),
            "visibilities": visibilities,
        }

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self._img_paths[idx]
        # mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")

        target = self._get_annotation(idx)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self._img_paths)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


def obj_detect_transforms(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data
