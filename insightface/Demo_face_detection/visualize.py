import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt

#===========================
# ResNet100 Backbone
# Trực quan hóa (visualize) feature map ở từng tầng C3, C4, C5 trong backbone ResNet100
#   C3, C4, C5 là feature maps lấy ra từ các tầng sâu khác nhau trong backbone ResNet.
# Giúp mình quan sát trực tiếp các feature mà model đang "nhìn thấy" và trích xuất được khi ảnh đi qua từng tầng
#
# Kiểm tra xem mô hình có học đúng những pattern cần thiết không:
#
# Tầng đầu thì phát hiện edge, corner
#
# Tầng giữa thì học texture, shape

# Tầng cuối thì các pattern trừu tượng hơn như vùng mặt, hình dáng khuôn mặt…
#===========================
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += shortcut
        return F.relu(out)

class ResNet100(nn.Module):
    def __init__(self):
        super(ResNet100, self).__init__()
        #conv1 + bn1 + pool: xử lý ảnh đầu vào
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(3, 2, 1) #max pooling với kernel size=3, stride=2, padding=1
        self.layer1 = self._make_layer(64, 256, 3, stride=1)
        self.layer2 = self._make_layer(256, 512, 13, stride=2)
        self.layer3 = self._make_layer(512, 1024, 30, stride=2)
        self.layer4 = self._make_layer(1024, 2048, 3, stride=2)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        C3 = self.layer1(x)
        C4 = self.layer2(C3)
        C5 = self.layer3(C4)
        return C3, C4, C5

#===========================
# 📦 FPN
#===========================
class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        self.lateral3 = nn.Conv2d(256, 256, 1)
        self.lateral4 = nn.Conv2d(512, 256, 1)
        self.lateral5 = nn.Conv2d(1024, 256, 1)
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
        self.cls_head = nn.Conv2d(256, num_anchors * 2, 1)
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
        return C3, C4, C5, cls_heads, reg_heads

#===========================
# 📊 Visualize feature maps
#===========================
def visualize_feature_maps(feature_map, title):
    fmap = feature_map[0]  # bỏ batch dim
    num_channels = fmap.shape[0]
    plt.figure(figsize=(15, 8))
    for i in range(min(8, num_channels)):
        plt.subplot(2, 4, i+1)
        plt.imshow(fmap[i].detach().cpu().numpy(), cmap='gray')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

#===========================
# Load ảnh và chạy thử
#===========================
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img / 255.0
    img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()
    return img

model = RetinaFaceDetectOnly()
img_tensor = preprocess_image(r"C:\Users\DucPc\Desktop\Dev\AI_Python\insightface\Demo_face_detection\dataset\Duc\Duc012.jpg")

# Chạy model
C3, C4, C5, cls_heads, reg_heads = model(img_tensor)
P3, P4, P5 = model.fpn(C3, C4, C5)

# Hiển thị feature map của từng stage ResNet
visualize_feature_maps(C3, "C3 Feature Maps (ResNet100)")
visualize_feature_maps(C4, "C4 Feature Maps (ResNet100)")
visualize_feature_maps(C5, "C5 Feature Maps (ResNet100)")

# Hiển thị feature map của từng stage FPN
visualize_feature_maps(P3, "P3 Feature Maps (FPN)")
visualize_feature_maps(P4, "P4 Feature Maps (FPN)")
visualize_feature_maps(P5, "P5 Feature Maps (FPN)")

# Hàm visualize false color map
def visualize_false_color_subplot(ax, feature_map, title, channels=(0, 1, 2)):
    fmap = feature_map[0]
    c1, c2, c3 = channels

    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-5)

    ch1 = normalize(fmap[c1].detach().cpu().numpy())
    ch2 = normalize(fmap[c2].detach().cpu().numpy())
    ch3 = normalize(fmap[c3].detach().cpu().numpy())

    false_color_img = np.stack([ch1, ch2, ch3], axis=-1)
    ax.imshow(false_color_img)
    ax.set_title(f"{title}\n(ch {c1}, {c2}, {c3})", fontsize=9)
    ax.axis('off')

# Tạo figure 2 hàng 3 cột
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# C3, C4, C5
visualize_false_color_subplot(axes[0,0], C3, "C3 (ResNet)", (3, 17, 55))
visualize_false_color_subplot(axes[0,1], C4, "C4 (ResNet)", (5, 30, 80))
visualize_false_color_subplot(axes[0,2], C5, "C5 (ResNet)", (2, 20, 60))

# P3, P4, P5
visualize_false_color_subplot(axes[1,0], P3, "P3 (FPN)", (2, 10, 33))
visualize_false_color_subplot(axes[1,1], P4, "P4 (FPN)", (5, 25, 100))
visualize_false_color_subplot(axes[1,2], P5, "P5 (FPN)", (3, 20, 80))

plt.tight_layout()
plt.show()