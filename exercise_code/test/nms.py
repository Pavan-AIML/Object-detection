import torch
from .base_tests import UnitTest, MethodTest, CompositeTest, ClassTest, test_results_to_score

from ..model.nms import non_maximum_suppression

########################################################################
# TODO:                                                                #
# Nothing to do here                                                   #
########################################################################

pass

########################################################################
#                           END OF YOUR CODE                           #
########################################################################


class NMSTest1(UnitTest):
    def __init__(self):
        self.bboxes = torch.load("exercise_code/test/nms_bboxes.pth")
        self.scores = torch.load("exercise_code/test/nms_scores.pth")
        self.threshold = 0.1
        self.nms_output = torch.load("exercise_code/test/nms_threshold_0_1.pth")
        self.output = None

    def test(self):
        self.output = non_maximum_suppression(self.bboxes, self.scores, self.threshold)
        if self.output.shape == self.nms_output.shape:
            if torch.allclose(self.output, self.nms_output):
                return True
        return False

    def define_success_message(self):
        return f"Congratulations: Test case 1 ran successfully"

    def define_failure_message(self):
        if self.output.shape == self.nms_output.shape:
            return f"The output of test case 1 is incorrect (expected {self.nms_output}, got {self.output})."
        return (
            f"The output shape of test case 1 is incorrect (expected {self.nms_output.shape}, got {self.output.shape})."
        )


class NMSTest2(UnitTest):
    def __init__(self):
        self.bboxes = torch.load("exercise_code/test/nms_bboxes.pth")
        self.scores = torch.load("exercise_code/test/nms_scores.pth")
        self.threshold = 0.2
        self.nms_output = torch.load("exercise_code/test/nms_threshold_0_2.pth")
        self.output = None

    def test(self):
        self.output = non_maximum_suppression(self.bboxes, self.scores, self.threshold)
        if self.output.shape == self.nms_output.shape:
            if torch.allclose(self.output, self.nms_output):
                return True
        return False

    def define_success_message(self):
        return f"Congratulations: Test case 1 ran successfully"

    def define_failure_message(self):
        if self.output.shape == self.nms_output.shape:
            return f"The output of test case 1 is incorrect (expected {self.nms_output}, got {self.output})."
        return (
            f"The output shape of test case 1 is incorrect (expected {self.nms_output.shape}, got {self.output.shape})."
        )


class NMSTest3(UnitTest):
    def __init__(self):
        self.bboxes = torch.load("exercise_code/test/nms_bboxes.pth")
        self.scores = torch.load("exercise_code/test/nms_scores.pth")
        self.threshold = 0.5
        self.nms_output = torch.load("exercise_code/test/nms_threshold_0_5.pth")
        self.output = None

    def test(self):
        self.output = non_maximum_suppression(self.bboxes, self.scores, self.threshold)
        if self.output.shape == self.nms_output.shape:
            if torch.allclose(self.output, self.nms_output):
                return True
        return False

    def define_success_message(self):
        return f"Congratulations: Test case 1 ran successfully"

    def define_failure_message(self):
        if self.output.shape == self.nms_output.shape:
            return f"The output of test case 1 is incorrect (expected {self.nms_output}, got {self.output})."
        return (
            f"The output shape of test case 1 is incorrect (expected {self.nms_output.shape}, got {self.output.shape})."
        )


class NMSTest4(UnitTest):
    def __init__(self):
        self.bboxes = torch.load("exercise_code/test/nms_bboxes.pth")
        self.scores = torch.load("exercise_code/test/nms_scores.pth")
        self.threshold = 0.9
        self.nms_output = torch.load("exercise_code/test/nms_threshold_0_9.pth")
        self.output = None

    def test(self):
        self.output = non_maximum_suppression(self.bboxes, self.scores, self.threshold)
        if self.output.shape == self.nms_output.shape:
            if torch.allclose(self.output, self.nms_output):
                return True
        return False

    def define_success_message(self):
        return f"Congratulations: Test case 1 ran successfully"

    def define_failure_message(self):
        if self.output.shape == self.nms_output.shape:
            return f"The output of test case 1 is incorrect (expected {self.nms_output}, got {self.output})."
        return (
            f"The output shape of test case 1 is incorrect (expected {self.nms_output.shape}, got {self.output.shape})."
        )


class NMSTest(CompositeTest):
    def define_tests(self):
        return [
            NMSTest1(),
            NMSTest2(),
            NMSTest3(),
            NMSTest4(),
        ]

    # def define_class_name(self):
    #     return "inactive_tracks"


def test_non_maximum_suppression():
    test = NMSTest()
    return test_results_to_score(test())
