import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis
import onnxruntime as ort
from datetime import datetime
import pandas as pd
import time
import os

print("Available providers:", ort.get_available_providers())

with open("face_embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.6) # det_thresh: ngưỡng confidence


cap = cv2.VideoCapture(1)
cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Recognition", 1280, 720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

threshold = 0.5
recognized_faces = {}

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        box = face.bbox.astype(int)
        emb = face.normed_embedding

        max_sim = -1
        name = "Unknown"

        for person_name, person_embs in embeddings.items():
            for saved_emb in person_embs:
                sim = np.dot(emb, saved_emb)
                if sim > max_sim:
                    max_sim = sim
                    name = person_name

        if max_sim < threshold:
            name = "Unknown"

        print(f"Max similarity: {max_sim:.3f}, Name: {name}")

        # Chỉ thêm nếu chưa có trong danh sách và khác Unknown
        if name != "Unknown" and name not in recognized_faces:
            recognized_faces[name] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            print(f"-> Ghi nhận có mặt: {name}")

        # Vẽ khung và tên
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"{name}", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Thêm giờ vào góc phải trên cùng
    current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    text_size, _ = cv2.getTextSize(current_time, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    text_x = frame.shape[1] - text_size[0] - 10
    text_y = 30
    cv2.putText(frame, current_time, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Thêm số người đang nhận diện vào góc trái trên cùng
    num_people = len(faces)
    cv2.putText(frame, f"So nguoi: {num_people}", (text_x, text_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("Face Recognition", frame)

    save_dir = r"C:\Users\DucPc\Desktop\Dev\AI_Python\insightface\zdiemdanh"
    os.makedirs(save_dir, exist_ok= True)
    # Nếu quá 60 giây thì xuất file excel
    if time.time() - start_time >= 60:
        if recognized_faces:
            filename = datetime.now().strftime("Diem_danh_%d-%m-%Y_%H-%M-%S.xlsx")
            filepath = os.path.join(save_dir, filename)
            df = pd.DataFrame(list(recognized_faces.items()), columns=["Họ và tên", "First Detected Time"])
            df.to_excel(filepath, index=False)
            print(f"✅ Đã lưu danh sách vào '{filepath}'")
        start_time = time.time()  # reset lại bộ đếm thời gian

    # Bấm q để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Trước khi thoát hẳn, lưu file lần cuối nếu có dữ liệu
if recognized_faces:
    filename = datetime.now().strftime("Diem_danh_%d-%m-%Y_%H-%M-%S.xlsx")
    filepath = os.path.join(save_dir, filename)
    df = pd.DataFrame(list(recognized_faces.items()), columns=["Họ và tên", "First Detected Time"])
    df.to_excel(filepath, index=False)
    print(f"✅ Đã lưu danh sách vào '{filepath}'")


cap.release()
cv2.destroyAllWindows()
