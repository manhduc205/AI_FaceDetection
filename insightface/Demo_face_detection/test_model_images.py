import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from iresnet import iresnet50
from ArcFace_Loss import ArcFace
from scipy.spatial.distance import cosine
import numpy as np


# ✔️ Kiểm tra chất lượng model sau khi train:
# → Xem với ảnh cùng người thì khoảng cách cosine thấp
# → Khác người thì cosine cao hơn.
#
# ✔️ Tạo cơ sở để build real-time face recog:
# → Sau này trong webcam/video, bạn sinh embedding từng khuôn mặt detect được
# → So sánh với embedding đã lưu của từng người (database)
# → Nếu cosine distance < threshold → xác định người đó là ai.

# Hyperparams
embedding_size = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),              # Chuyển ảnh PIL thành Tensor PyTorch [3, H, W], giá trị [0,1]
    transforms.Normalize([0.5], [0.5]) # Chuẩn hóa về [-1,1] để model học tốt hơn
])

# Load model
model = iresnet50(num_features=embedding_size).to(device)
model.load_state_dict(torch.load("iresnet_face.pth", map_location=device))
model.eval()

# Hàm lấy embedding từ ảnh
def get_embedding(img_path):
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(img_tensor)
    return embedding.cpu().numpy().flatten()

# Load ảnh và tính cosine distance
emb1 = get_embedding(r"C:\Users\DucPc\Desktop\Dev\AI_Python\insightface\Demo_face_detection\dataset\Duc\Duc001.jpg")
emb2 = get_embedding(r"C:\Users\DucPc\Desktop\Dev\AI_Python\insightface\Demo_face_detection\dataset\Duc\Duc012.jpg")

distance = cosine(emb1, emb2)
print("Cosine Distance:", distance)

threshold = 0.5  # bạn test rồi chỉnh threshold phù hợp
if distance < threshold:
    print("→ Cùng người")
else:
    print("→ Khác người")
