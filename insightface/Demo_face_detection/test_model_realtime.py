import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from iresnet import iresnet50
from scipy.spatial.distance import cosine

# Hyperparams
embedding_size = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
threshold = 0.5

# Transform
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load model
model = iresnet50(num_features=embedding_size).to(device)
model.load_state_dict(torch.load("iresnet_face.pth", map_location=device))
model.eval()

# Hàm lấy embedding từ ảnh crop khuôn mặt
def get_embedding_from_face(face_img):
    img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(img_tensor)
    return embedding.cpu().numpy().flatten()

# Load database embedding
# Ví dụ 1 người: {'Duc': embedding_array}
known_embeddings = {
    'Duc': np.load("embedding_duc.npy"),
    'Minh': np.load("embedding_minh.npy")
}

# Mở webcam
cap = cv2.VideoCapture(0)

# Load Haarcascade để detect face
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect face
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        if face_img.size == 0:
            continue

        emb = get_embedding_from_face(face_img)

        # So sánh với từng embedding trong database
        name = "Unknown"
        for known_name, known_emb in known_embeddings.items():
            dist = cosine(emb, known_emb)
            if dist < threshold:
                name = known_name
                break

        # Vẽ khung + tên
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Real-time Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
