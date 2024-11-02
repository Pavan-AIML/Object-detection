from math import floor
import torch
from .base_tests import (
    UnitTest,
    MethodTest,
    CompositeTest,
    ClassTest,
    test_results_to_score,
)

from ..model.sliding_window_detection import sliding_window_detection
from ..model.network import Net


class SlidingWindowShapeTest1(UnitTest):
    def __init__(self) -> None:
        self.H, self.W = 129, 65
        self.patch_size = (128, 64)
        self.stride = 1
        self.net = Net()

        self.image = torch.rand(3, self.H, self.W)
        self.output_shape = (
            floor((self.image.shape[-2] - (self.patch_size[-2] - 1) - 1) / self.stride + 1),
            floor((self.image.shape[-1] - (self.patch_size[-1] - 1) - 1) / self.stride + 1),
        )
        self.output = None

    def test(self):
        self.output = sliding_window_detection(self.image, self.net, self.patch_size, self.stride)
        return self.output.shape == self.output_shape

    def define_success_message(self):
        return f"Congratulations: The output shape of sliding_window_detection with a stride of 1 is correct"

    def define_failure_message(self):
        return f"The output shape of sliding_window_detection with a stride of 1 is incorrect (expected {self.output_shape}, got {self.output.shape})."


class SlidingWindowShapeTest2(UnitTest):
    def __init__(self) -> None:
        self.H, self.W = 136, 71
        self.patch_size = (128, 64)
        self.stride = 2
        self.net = Net()

        self.image = torch.rand(3, self.H, self.W)
        self.output_shape = (
            floor((self.image.shape[-2] - (self.patch_size[-2] - 1) - 1) / self.stride + 1),
            floor((self.image.shape[-1] - (self.patch_size[-1] - 1) - 1) / self.stride + 1),
        )
        self.output = None

    def test(self):
        self.output = sliding_window_detection(self.image, self.net, self.patch_size, self.stride)
        return self.output.shape == self.output_shape

    def define_success_message(self):
        return f"Congratulations: The output shape of sliding_window_detection with a stride of 2 is correct"

    def define_failure_message(self):
        return f"The output shape of sliding_window_detection with a stride of 2 is incorrect (expected {self.output_shape}, got {self.output.shape})."


class SlidingWindowShapeRange(UnitTest):
    def __init__(self) -> None:
        self.H, self.W = 136, 71
        self.patch_size = (128, 64)
        self.stride = 2
        self.net = Net()

        self.image = torch.rand(3, self.H, self.W)
        self.output = None

    def test(self):
        self.output = sliding_window_detection(self.image, self.net, self.patch_size, self.stride)
        return torch.all(self.output >= 0.0) and torch.all(self.output <= 1.0)

    def define_success_message(self):
        return f"Congratulations: The output range of sliding_window_detection is correct"

    def define_failure_message(self):
        smaller_zero = torch.any(self.output < 0.0)
        greater_one = torch.any(self.output > 1.0)
        return f"The output shape range of sliding_window_detection is incorrect (expected all to be in the range [0.0, 1.0], got a value {'less than 0.0' if smaller_zero else ''}{' and ' if smaller_zero and greater_one else ''}{'greater than 1.0' if greater_one else ''})."


class SlidingWindowDetectionTest(CompositeTest):
    def define_tests(self):
        return [
            SlidingWindowShapeTest1(),
            SlidingWindowShapeTest2(),
            SlidingWindowShapeRange(),
        ]


def test_sliding_window_detection():
    test = SlidingWindowDetectionTest()
    return test_results_to_score(test())
