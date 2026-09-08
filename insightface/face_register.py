import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

# Khởi tạo model InsightFace với đường dẫn model local
model_path = "C:/Users/DucPc/Desktop/Dev/AI_Python/insightface"
app = FaceAnalysis(name="buffalo_l", root=model_path, providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Đường dẫn thư mục dataset chứa các folder ảnh
dataset_path = "dataset"
embeddings = {}

# Duyệt qua từng người trong dataset
for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_path):
        continue

    print(f"📦 Processing: {person_name}")

    # Duyệt từng ảnh trong folder của người đó
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"❌ Không đọc được ảnh: {img_path}")
            continue

        # Resize ảnh nếu to quá
        if img.shape[0] > 800 or img.shape[1] > 800:
            img = cv2.resize(img, (640, 640))

        # Lấy face và embedding
        faces = app.get(img)

        if len(faces) > 0:
            emb = faces[0].normed_embedding
            if person_name not in embeddings:
                embeddings[person_name] = []
            embeddings[person_name].append(emb)
            print(f"✅ Đã lấy embedding từ {img_name}")
        else:
            print(f"⚠️ Không tìm thấy khuôn mặt trong {img_name}")

# Lưu embeddings ra file pkl
output_path = "face_embeddings.pkl"
with open(output_path, "wb") as f:
    pickle.dump(embeddings, f)

print(f"🎉 Đã lưu embedding vào {output_path}!")
