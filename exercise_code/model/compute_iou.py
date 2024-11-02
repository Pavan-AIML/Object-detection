import torch


def compute_iou(bbox_1: torch.Tensor, bbox_2: torch.Tensor) -> torch.Tensor:
    assert bbox_1.shape == bbox_2.shape

    ########################################################################
    # TODO:                                                                #
    # Compute the intersection over union (IoU) for two batches of         #
    # bounding boxes, each of shape (B, 4). The result should be a tensor  #
    # of shape (B,).                                                       #
    # NOTE: the format of the bounding boxes is (ltrb), meaning            #
    # (left edge, top edge, right edge, bottom edge). Remember the         #
    # orientation of the image coordinates.                                #
    # NOTE: First calculate the intersection and use this to compute the   #
    # union                                                                #
    # iou = ...                                                            #
    ########################################################################


    pass

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return iou
