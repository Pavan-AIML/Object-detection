import torch

from exercise_code.model.compute_iou import compute_iou


def non_maximum_suppression(bboxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
    ########################################################################
    # TODO:                                                                #
    # Compute the non maximum suppression                                  #
    # Input:                                                               #
    # bounding boxes of shape B,4                                          #
    # scores of shape B                                                    #
    # threshold for iou: if the overlap is bigger, only keep one of the    #
    # bboxes                                                               #
    # Output:                                                              #
    # bounding boxes of shape B_,4                                         #
    ########################################################################

    pass

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return bboxes_nms
