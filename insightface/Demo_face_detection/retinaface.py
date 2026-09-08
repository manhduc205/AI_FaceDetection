import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

#===========================
# 📦 Basic Residual Block
# sử dụng các layer, resnet100 y hệt file visualize.py nhưng ở đây không tìm các tầng feature map
# mà nó trả về classifications, bboxes, landmarks
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

#===========================
# 📦 ResNet50 Backbone
#===========================
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
# 📦 Feature Pyramid Network (FPN)
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
# 📦 Class Head
#===========================
class ClassHead(nn.Module):
    def __init__(self, num_anchors=2):
        super(ClassHead, self).__init__()
        self.conv = nn.Conv2d(256, num_anchors * 2, 1)

    def forward(self, features):
        outputs = []
        for f in features:
            outputs.append(self.conv(f))
        return outputs

#===========================
# 📦 Bbox Head
#===========================
class BboxHead(nn.Module):
    def __init__(self, num_anchors=2):
        super(BboxHead, self).__init__()
        self.conv = nn.Conv2d(256, num_anchors * 4, 1)

    def forward(self, features):
        outputs = []
        for f in features:
            outputs.append(self.conv(f))
        return outputs

#===========================
# 📦 Landmark Head
#===========================
class LandmarkHead(nn.Module):
    def __init__(self, num_anchors=2):
        super(LandmarkHead, self).__init__()
        self.conv = nn.Conv2d(256, num_anchors * 10, 1)

    def forward(self, features):
        outputs = []
        for f in features:
            outputs.append(self.conv(f))
        return outputs

#===========================
# 📦 RetinaFace Detector
#===========================
class RetinaFace(nn.Module):
    def __init__(self):
        super(RetinaFace, self).__init__()
        self.backbone = ResNet100()
        self.fpn = FPN()
        self.class_head = ClassHead()
        self.bbox_head = BboxHead()
        self.landmark_head = LandmarkHead()

    def forward(self, x):
        C3, C4, C5 = self.backbone(x)
        P3, P4, P5 = self.fpn(C3, C4, C5)
        features = [P3, P4, P5]

        classifications = self.class_head(features)
        bboxes = self.bbox_head(features)
        landmarks = self.landmark_head(features)

        return classifications, bboxes, landmarks

# --- Hàm preprocess ảnh ---
def preprocess_image(img_path, size=640):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1,3,H,W)
    return img_tensor


# Hàm sinh anchors theo config RetinaFace (ví dụ)
def generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios):
    # feature_map_sizes: list (h,w) từng scale
    # anchor_sizes: list size box
    # anchor_ratios: list tỉ lệ anchor (ví dụ [1.0])

    anchors = []
    for idx, (fm_h, fm_w) in enumerate(feature_map_sizes):
        size = anchor_sizes[idx]
        for i in range(fm_h):
            for j in range(fm_w):
                cx = (j + 0.5) / fm_w
                cy = (i + 0.5) / fm_h
                for ratio in anchor_ratios:
                    w = size * ratio
                    h = size / ratio
                    anchors.append([cx, cy, w, h])
    return torch.tensor(anchors)  # (num_anchors, 4)

# Hàm chuyển bbox center format sang bbox (x1,y1,x2,y2) trên ảnh scale 0-1
def decode_bbox(raw_bboxes, anchors):
    # raw_bboxes: (N,4) = (dx,dy,dw,dh)
    # anchors: (N,4) = (cx,cy,w,h)
    anchors_cx = anchors[:,0]
    anchors_cy = anchors[:,1]
    anchors_w = anchors[:,2]
    anchors_h = anchors[:,3]

    dx = raw_bboxes[:,0]
    dy = raw_bboxes[:,1]
    dw = raw_bboxes[:,2]
    dh = raw_bboxes[:,3]

    pred_cx = dx * anchors_w + anchors_cx
    pred_cy = dy * anchors_h + anchors_cy
    pred_w = torch.exp(dw) * anchors_w
    pred_h = torch.exp(dh) * anchors_h

    x1 = pred_cx - pred_w / 2
    y1 = pred_cy - pred_h / 2
    x2 = pred_cx + pred_w / 2
    y2 = pred_cy + pred_h / 2

    return torch.stack([x1,y1,x2,y2], dim=1)  # (N,4)

# Hàm decode landmark (dạng offsets cx,cy theo anchors)
def decode_landmarks(raw_landmarks, anchors):
    # raw_landmarks: (N,10), 5 điểm (x,y) offset
    # anchors: (N,4) cx,cy,w,h
    anchors_cx = anchors[:,0]
    anchors_cy = anchors[:,1]
    anchors_w = anchors[:,2]
    anchors_h = anchors[:,3]

    raw_landmarks = raw_landmarks.view(-1, 5, 2)
    pred_landmarks = torch.zeros_like(raw_landmarks)

    for i in range(5):
        pred_landmarks[:,i,0] = raw_landmarks[:,i,0] * anchors_w + anchors_cx
        pred_landmarks[:,i,1] = raw_landmarks[:,i,1] * anchors_h + anchors_cy

    return pred_landmarks.view(-1, 10)  # (N,10)

# Vẽ bbox, landmark lên ảnh gốc
def draw_on_image(img, bboxes, landmarks, scores=None, threshold=0.5):
    h, w, _ = img.shape
    for i, bbox in enumerate(bboxes):
        if scores is not None and scores[i] < threshold:
            continue
        x1 = int(bbox[0]*w)
        y1 = int(bbox[1]*h)
        x2 = int(bbox[2]*w)
        y2 = int(bbox[3]*h)

        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

        lm = landmarks[i].view(-1,2)
        for (lx,ly) in lm:
            lx = int(lx*w)
            ly = int(ly*h)
            cv2.circle(img, (lx, ly), 2, (0,0,255), -1)
# --- Main chạy thử ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RetinaFace().to(device)
    model.eval()

    img_path = r"C:\Users\DucPc\Desktop\Dev\AI_Python\insightface\Demo_face_detection\dataset\Duc\Duc012.jpg"
    input_tensor = preprocess_image(img_path).to(device)

    with torch.no_grad():
        classifications, bboxes, landmarks = model(input_tensor)

    print("Classifications:", [c.shape for c in classifications])
    print("Bounding Boxes :", [b.shape for b in bboxes])
    print("Landmarks      :", [l.shape for l in landmarks])
