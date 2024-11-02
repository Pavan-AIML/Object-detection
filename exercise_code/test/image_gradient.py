import torch
from .base_tests import (
    UnitTest,
    MethodTest,
    CompositeTest,
    ClassTest,
    test_results_to_score,
)

from ..model.compute_image_gradient import compute_image_gradient

########################################################################
# TODO:                                                                #
# Nothing to do here                                                   #
########################################################################

pass

########################################################################
#                           END OF YOUR CODE                           #
########################################################################


class GradientShape(UnitTest):
    def __init__(self):
        self.num_images = 2
        self.H, self.W = 128, 64
        self.images = torch.zeros(self.num_images, self.H, self.W)
        self.gradient_norm = None
        self.gradient_angle = None

    def test(self):
        self.gradient_norm, self.gradient_angle = compute_image_gradient(self.images)
        return self.gradient_norm.shape == (self.num_images, self.H, self.W) and self.gradient_angle.shape == (
            self.num_images,
            self.H,
            self.W,
        )

    def define_success_message(self):
        return f"Congratulations: The output shape of compute_image_gradient is correct"

    def define_failure_message(self):
        return f"The output shape of compute_image_gradient is incorrect (expected {(self.num_images, self.H, self.W)}, got {self.output.shape})."


class GradientOutput(UnitTest):
    def __init__(self):
        self.images = torch.load("exercise_code/test/images.pth")

        self.gradient_norm_results = torch.load("exercise_code/test/images_gradient_norm.pth")
        self.gradient_angle_results = torch.load("exercise_code/test/images_gradient_angle.pth")
        self.gradient_norm = None
        self.gradient_angle = None

    def test(self):
        self.gradient_norm, self.gradient_angle = compute_image_gradient(self.images)
        print((self.gradient_norm_results / self.gradient_norm).shape)
        scale_factor = torch.mean(self.gradient_norm_results / self.gradient_norm)

        return torch.allclose(self.gradient_norm_results, scale_factor * self.gradient_norm) and torch.allclose(
            self.gradient_angle_results, self.gradient_angle
        )

    def define_success_message(self):
        return f"Congratulations: Gradient computation is correct"

    def define_failure_message(self):
        return f"The gradient computation does not return the expected value. This does not necessarily implicate wrong implementation. There are different ways to define the gradient and we do not check every possibillity. Please refer to the visualization to get a hint to whether your implementation produces a reasonable result."


class ImageGradientTest(CompositeTest):
    def define_tests(self):
        return [
            GradientShape(),
            GradientOutput(),
        ]


def test_compute_image_gradient():
    test = ImageGradientTest()
    return test_results_to_score(test())
