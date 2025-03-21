import cv2
import torch
import numpy as np
import os
import json
import subprocess
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import simpledialog, messagebox
from sklearn.neighbors import KNeighborsClassifier
from facenet_pytorch import MTCNN, InceptionResnetV1
from DatabaseHooking import get_all_students, update_attendance

# Thêm import cho Pillow
from PIL import Image, ImageDraw, ImageFont

# ==============================
# Đọc cấu hình từ file config.json
# ==============================
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

# ==============================
# Thiết lập thiết bị: sử dụng GPU nếu có, ngược lại dùng CPU
# ==============================
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Sử dụng thiết bị:", device)

# ==============================
# Cấu hình API: Nếu đầy đủ thông tin cấu hình thì bật USE_EXTERNAL_API
# ==============================
external_api_key = config.get("external_api_key", "").strip()
external_detection_endpoint = config.get("external_detection_endpoint", "").strip()
external_verify_endpoint = config.get("external_verify_endpoint", "").strip()
if external_api_key and external_detection_endpoint and external_verify_endpoint:
    USE_EXTERNAL_API = True
    EXTERNAL_API_KEY = external_api_key
    EXTERNAL_DETECTION_ENDPOINT = external_detection_endpoint
    EXTERNAL_VERIFY_ENDPOINT = external_verify_endpoint
    print("API bên ngoài được kích hoạt.")
else:
    USE_EXTERNAL_API = False
    print("API bên ngoài không được kích hoạt do thiếu thông tin cấu hình.")


# ==============================
# Hàm đọc ảnh hỗ trợ Unicode và khoảng trắng trong đường dẫn
# ==============================
def read_image(filename):
    """
    Đọc ảnh từ đường dẫn chứa Unicode và khoảng trắng.
    Sử dụng pathlib để xử lý đường dẫn, mở file ở chế độ binary,
    và giải mã ảnh bằng cv2.imdecode.
    """
    try:
        path_obj = Path(filename)
        with open(path_obj, "rb") as f:
            data = f.read()
        bytes_array = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(bytes_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Không thể mở ảnh {filename}: {e}")
        return None

# ==============================
# Hàm vẽ text Unicode bằng Pillow
# ==============================
def draw_text_pillow(img, text, pos, text_color=(255, 255, 255), font_size=32):
    """
    Vẽ text Unicode (có dấu) lên ảnh OpenCV bằng Pillow.
    - img: ảnh numpy array (BGR)
    - text: chuỗi unicode
    - pos: (x, y) toạ độ góc trên trái
    - text_color: màu (R, G, B)
    - font_size: cỡ chữ
    Lưu ý: Cần có font hỗ trợ tiếng Việt, ví dụ 'Arial Unicode.ttf'.
    """
    # Chuyển ảnh BGR -> RGB để Pillow xử lý
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)

    draw = ImageDraw.Draw(pil_img)
    try:
        # Chọn font (phải có sẵn font .ttf hỗ trợ tiếng Việt)
        font = ImageFont.truetype("Arial Unicode.ttf", font_size)
    except:
        # Nếu không tìm thấy font, dùng default
        font = ImageFont.load_default()

    draw.text(pos, text, font=font, fill=text_color)

    # Chuyển ngược RGB -> BGR
    new_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return new_img

# ==============================
# Hàm tăng cường ảnh với CLAHE
# ==============================
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return enhanced

# ==============================
# Hàm xử lý ảnh: resize, CLAHE, giảm nhiễu và tăng cường độ nét
# ==============================
def enhance_image(frame):
    enhanced = apply_clahe(frame)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    return enhanced

def preprocess_frame(frame, target_width=1200):
    height, width = frame.shape[:2]
    scale = 1.0
    if width > target_width:
        scale = target_width / float(width)
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    processed = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
    processed = enhance_image(processed)
    return processed, scale

# ==============================
# Khởi tạo mô hình phát hiện khuôn mặt và trích xuất embedding
# ==============================
from facenet_pytorch import MTCNN, InceptionResnetV1
mtcnn = MTCNN(keep_all=True, device=device, thresholds=[0.50, 0.65, 0.75])
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ==============================
# Data augmentation
# ==============================
def augment_image(image):
    augmented = []
    augmented.append(image)  # Ảnh gốc
    augmented.append(cv2.flip(image, 1))  # Lật ngang
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 5, 1.0)
    augmented.append(cv2.warpAffine(image, M, (w, h)))  # Xoay 5 độ
    augmented.append(cv2.convertScaleAbs(image, alpha=1.2, beta=10))  # Tăng sáng
    augmented.append(cv2.convertScaleAbs(image, alpha=0.8, beta=-10))  # Giảm sáng
    return augmented

# ==============================
# Load khuôn mặt từ DB, tạo embedding
# ==============================
def load_known_faces(cursor):
    from DatabaseHooking import get_all_students
    students = get_all_students(cursor)
    known_faces = []
    # Dùng cache để tránh đọc file nhiều lần nếu đường dẫn trùng nhau
    path_cache = {}

    for student in students:
        student_id, HoVaTen, Lop, ImagePath, status, attendance_time = student
        # Dùng pathlib để xử lý đường dẫn
        path_obj = Path(ImagePath)

        if path_obj in path_cache:
            img = path_cache[path_obj]
        else:
            if not path_obj.exists():
                print(f"File không tồn tại: {path_obj}")
                continue
            img = read_image(str(path_obj))
            if img is None:
                continue
            path_cache[path_obj] = img

        # Nếu ảnh quá nhỏ, upscale
        if img.shape[0] < 100 or img.shape[1] < 100:
            img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented_images = augment_image(rgb_img)
        embeddings = []
        for aug_img in augmented_images:
            boxes, _ = mtcnn.detect(aug_img)
            if boxes is None or len(boxes) == 0:
                continue
            x1, y1, x2, y2 = map(int, boxes[0])
            face_crop = aug_img[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            h_crop, w_crop = face_crop.shape[:2]
            if h_crop < 80 or w_crop < 80:
                face_crop = cv2.resize(face_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            try:
                face_crop = cv2.resize(face_crop, (160, 160), interpolation=cv2.INTER_LINEAR)
            except Exception as e:
                print(f"Lỗi resize ảnh: {path_obj} - {e}")
                continue
            face_tensor = torch.tensor(face_crop, dtype=torch.float32).permute(2, 0, 1) / 255.0
            face_tensor = face_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = resnet(face_tensor).detach().cpu().numpy()[0]
            embeddings.append(embedding)
        if len(embeddings) == 0:
            print(f"Không tạo được embedding cho ảnh: {path_obj}")
            continue
        known_faces.append({
            "id": student_id,
            "name": HoVaTen,
            "embeddings": embeddings,
            "ref_image": img
        })
    return known_faces

# ==============================
# Huấn luyện KNN
# ==============================
def train_knn_classifier(known_faces):
    from sklearn.neighbors import KNeighborsClassifier
    X = []
    y = []
    student_dict = {}
    reference_images = {}
    for face in known_faces:
        for emb in face["embeddings"]:
            X.append(emb)
            y.append(face["id"])
        student_dict[face["id"]] = face["name"]
        reference_images[face["id"]] = face["ref_image"]
    if len(X) == 0:
        raise ValueError("Không có embedding nào được load!")
    n_neighbors = min(3, len(X))
    knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric="euclidean")
    knn_clf.fit(X, y)
    return knn_clf, student_dict, reference_images

# ==============================
# Hàm mở webcam USB ngoài (index=1) bằng CAP_MSMF
# ==============================
def get_camera_source():
    return 1

def open_camera(camera_source):
    backend = cv2.CAP_MSMF
    video_capture = cv2.VideoCapture(camera_source, backend)
    if video_capture.isOpened():
        print(f"🎥 Đã mở webcam {camera_source} với CAP_MSMF")
        return video_capture
    else:
        print("❌ Không thể mở webcam USB ngoài!")
        return None

def open_stream_with_ffmpeg(rtsp_url, width=640, height=480):
    command = [
        'ffmpeg',
        '-rtsp_transport', 'tcp',
        '-i', rtsp_url,
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-'
    ]
    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10 ** 8)
    return pipe, width, height

# ==============================
# Hàm chính: Nhận diện khuôn mặt theo thời gian thực
# ==============================
def main(cnx, cursor, camera_source=None):
    from DatabaseHooking import update_attendance

    # Tải dữ liệu khuôn mặt từ DB và huấn luyện KNN
    known_faces = load_known_faces(cursor)
    if known_faces:
        knn_clf, student_dict, reference_images = train_knn_classifier(known_faces)
    else:
        messagebox.showwarning("Cảnh báo",
                               "Không có khuôn mặt nào được tải từ DB! Hệ thống sẽ chạy ở chế độ không nhận dạng.")
        knn_clf = None
        student_dict = {}
        reference_images = {}

    if camera_source is None:
        camera_source = get_camera_source()

    # Mở webcam bằng CAP_MSMF
    video_capture = open_camera(camera_source)
    if not video_capture:
        messagebox.showerror("Lỗi", "Không thể truy cập webcam USB ngoài!")
        return

    print("Bắt đầu nhận diện khuôn mặt theo thời gian thực. Nhấn 'q' để thoát.")
    threshold = 0.6

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Không thể đọc khung hình từ webcam.")
            break

        processed_frame, scale = preprocess_frame(frame, target_width=1200)
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        face_locations = []
        face_results = []
        face_crops_list = []

        boxes, _ = mtcnn.detect(rgb_frame)
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                if x1 < 0 or y1 < 0 or x2 > rgb_frame.shape[1] or y2 > rgb_frame.shape[0]:
                    continue
                face_crop = rgb_frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue
                h_crop, w_crop = face_crop.shape[:2]
                if h_crop < 80 or w_crop < 80:
                    face_crop = cv2.resize(face_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                try:
                    face_crop_resized = cv2.resize(face_crop, (160, 160), interpolation=cv2.INTER_LINEAR)
                except Exception:
                    continue
                face_tensor = torch.tensor(face_crop_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
                face_tensor = face_tensor.unsqueeze(0).to(device)
                face_locations.append((y1, x2, y2, x1))
                face_crops_list.append(face_crop)

            if len(face_crops_list) > 0 and knn_clf is not None:
                faces_batch = []
                for crop in face_crops_list:
                    if crop.size == 0:
                        continue
                    c_h, c_w = crop.shape[:2]
                    if c_h < 80 or c_w < 80:
                        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    try:
                        crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_LINEAR)
                    except:
                        continue
                    crop_tensor = torch.tensor(crop, dtype=torch.float32).permute(2, 0, 1) / 255.0
                    faces_batch.append(crop_tensor)
                if len(faces_batch) > 0:
                    faces_batch = torch.stack(faces_batch).to(device)
                    with torch.no_grad():
                        embeddings = resnet(faces_batch).detach().cpu().numpy()
                    for emb in embeddings:
                        distances, indices = knn_clf.kneighbors([emb], n_neighbors=1)
                        if distances[0][0] < threshold:
                            candidate_id = knn_clf.predict([emb])[0]
                            name = student_dict.get(candidate_id, "Unknown")
                        else:
                            name = "Unknown"
                        face_results.append((name, candidate_id if name != "Unknown" else None))

        now = datetime.now()
        detection_timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        if len(face_locations) == len(face_results):
            for (name, student_id), (top, right, bottom, left) in zip(face_results, face_locations):
                top = int(top / scale)
                right = int(right / scale)
                bottom = int(bottom / scale)
                left = int(left / scale)

                # Cập nhật attendance
                if name != "Unknown" and student_id is not None:
                    status = "✓" if now.hour < 7 else "Late"
                    update_attendance(cursor, cnx, int(student_id), status, detection_timestamp)

                # Vẽ khung
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

                # Vẽ text Unicode bằng Pillow
                # Thay vì cv2.putText, ta gọi draw_text_pillow
                frame = draw_text_pillow(
                    frame,
                    name,
                    (left + 6, bottom - 30),  # Lùi lên một chút để không bị đè
                    text_color=(255, 255, 255),
                    font_size=28
                )

        cv2.imshow("Automated-Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    cursor.close()
    cnx.close()


if __name__ == "__main__":
    from tkinter import messagebox

    messagebox.showinfo("Thông báo", "Hãy chạy FacialRecognition thông qua giao diện GUI!")
