import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class Net(nn.Module):
    def __init__(self, activation: nn.Module = nn.Sigmoid(), input_size=3780, hidden_size=128, classes=1):
        super().__init__()
        self.input_size = input_size

        # Here we initialize our activation and set up our two linear layers
        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, classes)

    def forward(self, x):
        # x = x.view(-1, self.input_size)  # flatten
        x = self.fc1(x)
        x = self.activation(x)
        # x = self.fc2(x)
        # x = self.activation(x)
        # x = self.fc3(x)
        # x = self.activation(x)
        x = self.fc4(x)
        x = self.sigmoid(x)

        return x[:, 0]
        

class FRCNN_FPN(FasterRCNN):

    def __init__(self, num_classes, nms_thresh=0.5):
        backbone = resnet_fpn_backbone('resnet50', False)
        super(FRCNN_FPN, self).__init__(backbone, num_classes)

        self.roi_heads.nms_thresh = nms_thresh

    def detect(self, img):
        device = list(self.parameters())[0].device
        img = img.to(device)

        detections = self(img)[0]

        return detections['boxes'].detach().cpu(), detections['scores'].detach().cpu()