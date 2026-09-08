import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2

#===========================
# 📦 ResNet100 Backbone
#===========================
# Dùng ResNet100 pretrain hoặc custom build lại theo paper.
# Ở đây mình lấy ResNet100 simple version cho feature extraction

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Nếu kích thước input khác output, cần conv1x1 để chuyển kích thước shortcut
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)  # chuyển kích thước nếu cần

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += shortcut
        out = F.relu(out)
        return out


class ResNet100(nn.Module):
    def __init__(self):
        super(ResNet100, self).__init__()

        # Giai đoạn đầu: conv1 + maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(3, 2, 1)

        # Các tầng feature map: C3, C4, C5
        self.layer1 = self._make_layer(64, 256, 3, stride=1)  # giữ nguyên kích thước
        self.layer2 = self._make_layer(256, 512, 13, stride=2)  # giảm 1 nửa
        self.layer3 = self._make_layer(512, 1024, 30, stride=2)
        self.layer4 = self._make_layer(1024, 2048, 3, stride=2)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        # block đầu tiên có thể stride khác 1 để giảm kích thước
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # Conv1
        x = self.pool(x)

        C3 = self.layer1(x)
        C4 = self.layer2(C3)
        C5 = self.layer3(C4)

        return C3, C4, C5

#===========================
# 📦 Feature Pyramid Network (FPN)
#===========================
class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()

        # Chuyển số kênh về 256
        self.lateral3 = nn.Conv2d(256, 256, 1)
        self.lateral4 = nn.Conv2d(512, 256, 1)
        self.lateral5 = nn.Conv2d(1024, 256, 1)

        # Smooth conv 3x3
        self.smooth3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth5 = nn.Conv2d(256, 256, 3, 1, 1)

    def forward(self, C3, C4, C5):
        P5 = self.lateral5(C5)
        P4 = self.lateral4(C4) + F.interpolate(P5, size=C4.shape[2:], mode='nearest')
        P3 = self.lateral3(C3) + F.interpolate(P4, size=C3.shape[2:], mode='nearest')

        P5 = self.smooth5(P5)
        P4 = self.smooth4(P4)
        P3 = self.smooth3(P3)

        return P3, P4, P5

#===========================
# 📦 Detection Head
#===========================
class DetectionHead(nn.Module):
    def __init__(self, num_anchors=2):
        super(DetectionHead, self).__init__()

        # Classification head: phát hiện có face hay không
        self.cls_head = nn.Conv2d(256, num_anchors * 2, 1)

        # Regression head: dự đoán bbox [dx, dy, dw, dh]
        self.reg_head = nn.Conv2d(256, num_anchors * 4, 1)

    def forward(self, P3, P4, P5):
        cls3 = self.cls_head(P3)
        cls4 = self.cls_head(P4)
        cls5 = self.cls_head(P5)

        reg3 = self.reg_head(P3)
        reg4 = self.reg_head(P4)
        reg5 = self.reg_head(P5)

        return (cls3, cls4, cls5), (reg3, reg4, reg5)

#===========================
# 📦 RetinaFace Detect-only
#===========================
class RetinaFaceDetectOnly(nn.Module):
    def __init__(self):
        super(RetinaFaceDetectOnly, self).__init__()
        self.backbone = ResNet100()
        self.fpn = FPN()
        self.head = DetectionHead()

    def forward(self, x):
        C3, C4, C5 = self.backbone(x)
        P3, P4, P5 = self.fpn(C3, C4, C5)
        cls_heads, reg_heads = self.head(P3, P4, P5)
        return cls_heads, reg_heads

#===========================
# 📦 Test thử model với input
#===========================
model = RetinaFaceDetectOnly()
dummy_input = torch.randn(1, 3, 640, 640)
cls_heads, reg_heads = model(dummy_input)

for i, (cls, reg) in enumerate(zip(cls_heads, reg_heads)):
    print(f"P{i+3} cls shape: {cls.shape}, reg shape: {reg.shape}")
