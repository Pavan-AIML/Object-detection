import torch


def fill_hog_bins(gradient_norm: torch.Tensor, gradient_angle: torch.Tensor, num_bins: int) -> torch.Tensor:
    assert gradient_norm.shape == gradient_angle.shape
    device = gradient_norm.device

    ########################################################################
    # TODO:                                                                #
    # Based on the given gradient norm and angle, fill the Histogram of    #
    # Orientatied Gradients bins.                                          #
    # For this, first determine the two bins a gradient should be part of  #
    # based on the gradient angle. Then, based on the distance to the bins #
    # fill the bins with a weighting of the gradient norm.                 #
    # Input:                                                               #
    # Both gradient_norm and gradient_angle have the shape (B, N), where   #
    # N is the flatten cell.                                               #
    # The angle is given in degrees with values in the range [0.0, 180.0). #
    # (the angles 0.0 and 180.0 are equivalent.)                           #
    # Output:                                                              #
    # The output is a histogram over the flattened cell with num_bins      #
    # quantized values and should have the shape (B, num_bins)             #
    #                                                                      #
    # NOTE: Keep in mind the cyclical nature of the gradient angle and     #
    # its effects on the bins.                                             #
    # NOTE: make sure, the histogram_of_oriented_gradients are on the same #
    # device as the gradient inputs. In general be mindful of the device   #
    # of the tensors.                                                      #
    # histogram_of_oriented_gradients = ...                                #
    ########################################################################


    pass

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return histogram_of_oriented_gradients
