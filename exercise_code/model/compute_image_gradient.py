from typing import Tuple
import torch.nn as nn
import torch


def compute_image_gradient(images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # images B x H x W

    ########################################################################
    # TODO:                                                                #
    # Compute the 2-dimenational gradient for a given grey image of size   #
    # B x H x W. The return values of this function should be the norm and #
    # the angle of this gradient vector.                                   #
    # NOTE: first, calculate the gradient in x and y direction             #
    # (you will need add padding to the image boundaries),                 #
    # then, compute the vector norm and angle.                             #
    # The angle of a given gradient angle is defined                       #
    # in degrees (range=0.,,,.360).                                        #
    # NOTE: The angle is defined counter-clockwise angle between the       #
    # gradient and the unit vector along the x-axis received from atan2.   #
    ########################################################################


    pass

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return norm, angle
