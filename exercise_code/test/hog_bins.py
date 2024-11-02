import torch
from .base_tests import (
    UnitTest,
    MethodTest,
    CompositeTest,
    ClassTest,
    test_results_to_score,
)

from ..model.fill_hog_bins import fill_hog_bins


########################################################################
# TODO:                                                                #
# Nothing to do here                                                   #
########################################################################

pass

########################################################################
#                           END OF YOUR CODE                           #
########################################################################


class BinsShapeTest(UnitTest):
    def __init__(self) -> None:
        self.num_cells = 5
        self.cell_size = 64
        self.num_bins = 9
        self.gradient_magnitude = torch.rand(self.num_cells, self.cell_size)
        self.gradient_angle = torch.rand(self.num_cells, self.cell_size) * 180.0

        self.output = None

    def test(self):
        self.output = fill_hog_bins(self.gradient_magnitude, self.gradient_angle, self.num_bins)
        return self.output.shape == (self.num_cells, self.num_bins)

    def define_success_message(self):
        return f"Congratulations: The output shape of fill_hog_bins is correct"

    def define_failure_message(self):
        return f"The output shape of fill_hog_bins is incorrect (expected {(self.num_cells, self.num_bins)}, got {self.output.shape})."


class BinsRangeTest(UnitTest):
    def __init__(self) -> None:
        self.num_cells = 5
        self.cell_size = 64
        self.num_bins = 9
        self.gradient_magnitude = torch.rand(self.num_cells, self.cell_size)
        self.gradient_angle = torch.rand(self.num_cells, self.cell_size) * 180.0

        self.output = None

    def test(self):
        self.output = fill_hog_bins(self.gradient_magnitude, self.gradient_angle, self.num_bins)
        return torch.all(self.output >= 0.0)

    def define_success_message(self):
        return f"Congratulations: The output range of compute_Bins is correct"

    def define_failure_message(self):
        return f"The output shape range of compute_Bins is incorrect (expected all to be in the range [0.0, 1.0], got a value less than 0.0"


# TODO:
class BinsTest1(UnitTest):
    def __init__(self):
        self.gradient_norm = torch.load("exercise_code/test/blockified_gradient_norm.pth")
        self.gradient_angle = torch.load("exercise_code/test/blockified_gradient_angle.pth")

        self.bins = torch.load("exercise_code/test/hog_bins.pth")
        self.output = None

    def test(self):
        self.output = fill_hog_bins(self.gradient_norm, self.gradient_angle, self.bins.shape[-1])
        return torch.allclose(self.output, self.bins)

    def define_success_message(self):
        return f"Congratulations: fill_hog_bins was implemented correctly"

    def define_failure_message(self):
        return f"The output of fill_hog_bins is incorrect (expected {self.bins}, got {self.output})."


class BinsTest(CompositeTest):
    def define_tests(self):
        return [
            BinsShapeTest(),
            BinsRangeTest(),
            BinsTest1(),
        ]


def test_fill_hog_bins():
    test = BinsTest()
    return test_results_to_score(test())
