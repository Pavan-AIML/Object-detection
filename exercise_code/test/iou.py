import torch
from .base_tests import UnitTest, MethodTest, CompositeTest, ClassTest, test_results_to_score

from ..model.compute_iou import compute_iou


class IoUShapeTest(UnitTest):
    def __init__(self) -> None:
        self.num_bboxes = 5
        self.bbox_1 = torch.rand(self.num_bboxes, 4)
        self.bbox_1[:, 2] = self.bbox_1[:, 0] + self.bbox_1[:, 2] # right bigger than left
        self.bbox_1[:, 3] = self.bbox_1[:, 1] + self.bbox_1[:, 3] # top bigger than bottom

        self.bbox_2 = torch.rand(self.num_bboxes, 4)
        self.bbox_2[:, 2] = self.bbox_2[:, 0] + self.bbox_2[:, 2] # right bigger than left
        self.bbox_2[:, 3] = self.bbox_2[:, 1] + self.bbox_2[:, 3] # top bigger than bottom

        self.output = None

    def test(self):
        self.output = compute_iou(self.bbox_1, self.bbox_2)
        return self.output.shape == (self.num_bboxes,)

    def define_success_message(self):
        return f"Congratulations: The output shape of compute_iou is correct"

    def define_failure_message(self):
        return f"The output shape of compute_iou is incorrect (expected {(self.num_bboxes,)}, got {self.output.shape})."


class IoUSymmetryTest(UnitTest):
    def __init__(self) -> None:
        self.num_bboxes = 5
        self.bbox_1 = torch.rand(self.num_bboxes, 4)
        self.bbox_1[:, 2] = self.bbox_1[:, 0] + self.bbox_1[:, 2]
        self.bbox_1[:, 3] = self.bbox_1[:, 1] + self.bbox_1[:, 3]

        self.bbox_2 = torch.rand(self.num_bboxes, 4)
        self.bbox_2[:, 2] = self.bbox_2[:, 0] + self.bbox_2[:, 2]
        self.bbox_2[:, 3] = self.bbox_2[:, 1] + self.bbox_2[:, 3]

        self.output1 = None

    def test(self):
        self.output1 = compute_iou(self.bbox_1, self.bbox_2)
        self.output2 = compute_iou(self.bbox_2, self.bbox_1)
        return torch.allclose(self.output1, self.output2)

    def define_success_message(self):
        return f"Congratulations: The output shape of compute_iou is symmetrical"

    def define_failure_message(self):
        return f"The output shape of compute_iou is not symmetrical."


class IoURangeTest(UnitTest):
    def __init__(self) -> None:
        self.num_bboxes = 300
        self.bbox_1 = torch.rand(self.num_bboxes, 4)
        self.bbox_1[:, 2] = self.bbox_1[:, 0] + self.bbox_1[:, 2]
        self.bbox_1[:, 3] = self.bbox_1[:, 1] + self.bbox_1[:, 3]

        self.bbox_2 = torch.rand(self.num_bboxes, 4)
        self.bbox_2[:, 2] = self.bbox_2[:, 0] + self.bbox_2[:, 2]
        self.bbox_2[:, 3] = self.bbox_2[:, 1] + self.bbox_2[:, 3]

        self.output = None

    def test(self):
        self.output = compute_iou(self.bbox_1, self.bbox_2)
        return torch.all(self.output >= 0.0) and torch.all(self.output <= 1.0)

    def define_success_message(self):
        return f"Congratulations: The output range of compute_iou is correct"

    def define_failure_message(self):
        smaller_zero = torch.any(self.output < 0.0)
        greater_one = torch.any(self.output > 1.0)
        return f"The output shape range of compute_iou is incorrect (expected all to be in the range [0.0, 1.0], got a value {'less than 0.0' if smaller_zero else ''}{' and ' if smaller_zero and greater_one else ''}{'greater than 1.0' if greater_one else ''})."


class IoUTest1(UnitTest):
    def __init__(self):
        self.bbox_1 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        self.bbox_2 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        self.result = 1.0
        self.output = None

    def test(self):
        self.output = compute_iou(self.bbox_1, self.bbox_2)
        return self.output[0].item() == self.result

    def define_success_message(self):
        return f"Congratulations: Test case 1 ran successfully"

    def define_failure_message(self):
        return f"The output of test case 1 is incorrect (expected {self.result}, got {self.output[0].item()})."


class IoUTest2(UnitTest):
    def __init__(self):
        self.bbox_1 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        self.bbox_2 = torch.tensor([[0.0, 0.0, 0.5, 1.0]])
        self.result = 0.5
        self.output = None

    def test(self):
        self.output = compute_iou(self.bbox_1, self.bbox_2)
        return self.output[0].item() == self.result

    def define_success_message(self):
        return f"Congratulations: Test case 1 ran successfully"

    def define_failure_message(self):
        return f"The output of test case 1 is incorrect (expected {self.result}, got {self.output[0].item()})."


class IoUTest3(UnitTest):
    def __init__(self):
        self.bbox_1 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        self.bbox_2 = torch.tensor([[0.0, 0.0, 2.0, 1.0]])
        self.result = 0.5
        self.output = None

    def test(self):
        self.output = compute_iou(self.bbox_1, self.bbox_2)
        return self.output[0].item() == self.result

    def define_success_message(self):
        return f"Congratulations: Test case 1 ran successfully"

    def define_failure_message(self):
        return f"The output of test case 1 is incorrect (expected {self.result}, got {self.output[0].item()})."


class IoUTest4(UnitTest):
    def __init__(self):
        self.bbox_1 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        self.bbox_2 = torch.tensor([[1.0, 1.0, 2.0, 2.0]])
        self.result = 0.0
        self.output = None

    def test(self):
        self.output = compute_iou(self.bbox_1, self.bbox_2)
        return self.output[0].item() == self.result

    def define_success_message(self):
        return f"Congratulations: Test case 1 ran successfully"

    def define_failure_message(self):
        return f"The output of test case 1 is incorrect (expected {self.result}, got {self.output[0].item()})."


class IoUTest(CompositeTest):
    def define_tests(self):
        return [
            IoUShapeTest(),
            IoURangeTest(),
            IoUSymmetryTest(),
            IoUTest1(),
            IoUTest2(),
            IoUTest3(),
            IoUTest4(),
        ]

    # def define_class_name(self):
    #     return "inactive_tracks"


def test_compute_iou():
    test = IoUTest()
    return test_results_to_score(test())
