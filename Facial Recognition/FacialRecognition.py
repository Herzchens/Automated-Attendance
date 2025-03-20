import cv2
import torch
import numpy as np
import requests
import json
import subprocess
from datetime import datetime
import tkinter as tk
from tkinter import simpledialog, messagebox
from sklearn.neighbors import KNeighborsClassifier
from facenet_pytorch import MTCNN, InceptionResnetV1
from DatabaseHooking import get_all_students, update_attendance

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
# Khởi tạo mô hình phát hiện khuôn mặt và trích xuất embedding
# ==============================
mtcnn = MTCNN(keep_all=True, device=device, thresholds=[0.55, 0.65, 0.75])
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


# ==============================
# Hàm tạo các biến thể của ảnh (data augmentation)
# ==============================
def augment_image(image):
    augmented = []
    # Ảnh gốc
    augmented.append(image)
    # Ảnh lật ngang
    flipped = cv2.flip(image, 1)
    augmented.append(flipped)
    # Xoay 5 độ
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 5, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    augmented.append(rotated)
    # Tăng sáng
    bright = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
    augmented.append(bright)
    # Giảm sáng
    dark = cv2.convertScaleAbs(image, alpha=0.8, beta=-10)
    augmented.append(dark)
    return augmented


# ==============================
# Hàm lấy nguồn camera dựa trên cấu hình
# ==============================
def get_camera_source():
    if config.get("camera_type", "Webcam mặc định") == "Webcam mặc định":
        return 0
    else:
        if config.get("camera_url"):
            return config["camera_url"]
        else:
            protocol = config.get("camera_protocol", "RTSP")
            user = config.get("camera_user", "")
            password = config.get("camera_pass", "")
            ip = config.get("camera_ip", "")
            port = config.get("camera_port", "")
            return f"{protocol.lower()}://{user}:{password}@{ip}:{port}"


# ==============================
# Hàm mở luồng video qua FFmpeg (dùng cho RTSP)
# ==============================
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
# Hàm load khuôn mặt đã biết từ DB và tạo embedding qua data augmentation
# ==============================
def load_known_faces(cursor):
    students = get_all_students(cursor)  # trả về 6 cột: id, HoVaTen, Lop, ImagePath, DiemDanhStatus, ThoiGianDiemDanh
    known_faces = []
    for student in students:
        student_id, HoVaTen, Lop, ImagePath, status, attendance_time = student
        img = cv2.imread(ImagePath)
        if img is None:
            print(f"Không thể tải ảnh: {ImagePath}")
            continue
        ref_image = img.copy()  # dùng cho API xác thực
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented_images = augment_image(rgb_img)
        embeddings = []
        for aug_img in augmented_images:
            boxes, _ = mtcnn.detect(aug_img)
            if boxes is None:
                continue
            x1, y1, x2, y2 = boxes[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            face_crop = aug_img[y1:y2, x1:x2]
            # Nếu khuôn mặt nhỏ, upscale nó
            h_crop, w_crop = face_crop.shape[:2]
            min_size = 80
            if h_crop < min_size or w_crop < min_size:
                face_crop = cv2.resize(face_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            try:
                face_crop = cv2.resize(face_crop, (160, 160), interpolation=cv2.INTER_LINEAR)
            except Exception as e:
                print(f"Lỗi resize ảnh: {ImagePath}", e)
                continue
            face_tensor = torch.tensor(face_crop, dtype=torch.float32).permute(2, 0, 1) / 255.0
            face_tensor = face_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = resnet(face_tensor).detach().cpu().numpy()[0]
            embeddings.append(embedding)
        if len(embeddings) == 0:
            print(f"Không tạo được embedding cho ảnh: {ImagePath}")
            continue
        known_faces.append({
            "id": student_id,
            "name": HoVaTen,
            "embeddings": embeddings,
            "ref_image": ref_image
        })
    return known_faces


# ==============================
# Huấn luyện mô hình KNN trên các embedding đã trích xuất
# ==============================
def train_knn_classifier(known_faces):
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
# Hàm xử lý ảnh: resize, giảm nhiễu và tăng cường độ nét
# ==============================
def enhance_image(frame):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(frame, -1, kernel)


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
# Hàm gọi API để lấy faceId từ ảnh
# ==============================
def get_face_id(image):
    ret, buf = cv2.imencode('.jpg', image)
    if not ret:
        return None
    headers = {
        'Ocp-Apim-Subscription-Key': EXTERNAL_API_KEY,
        'Content-Type': 'application/octet-stream'
    }
    params = {'returnFaceId': 'true'}
    response = requests.post(EXTERNAL_DETECTION_ENDPOINT, params=params, headers=headers, data=buf.tobytes())
    if response.status_code == 200:
        faces = response.json()
        if faces and len(faces) > 0:
            return faces[0].get("faceId")
    else:
        print("Lỗi gọi API detect:", response.status_code, response.text)
    return None


# ==============================
# Hàm xác thực khuôn mặt qua API (so sánh face_crop và reference_image)
# ==============================
def verify_face_with_api(face_crop, reference_image):
    face_id1 = get_face_id(face_crop)
    face_id2 = get_face_id(reference_image)
    if face_id1 is None or face_id2 is None:
        return False
    headers = {
        'Ocp-Apim-Subscription-Key': EXTERNAL_API_KEY,
        'Content-Type': 'application/json'
    }
    payload = {
        "faceId1": face_id1,
        "faceId2": face_id2
    }
    response = requests.post(EXTERNAL_VERIFY_ENDPOINT, json=payload, headers=headers)
    if response.status_code == 200:
        result = response.json()
        confidence = result.get("confidence", 0)
        if confidence > 0.7:
            return True
    else:
        print("Lỗi gọi API verify:", response.status_code, response.text)
    return False


# ==============================
# Hàm chính: nhận diện thời gian thực, cập nhật điểm danh và lưu timestamp
# ==============================
def main(cnx, cursor, camera_source=None):
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

    # Nếu camera_source là RTSP, dùng FFmpeg để mã hoá lại luồng
    use_ffmpeg = False
    if isinstance(camera_source, str) and camera_source.lower().startswith("rtsp://"):
        pipe, frame_width, frame_height = open_stream_with_ffmpeg(camera_source, width=640, height=480)
        use_ffmpeg = True
    else:
        video_capture = cv2.VideoCapture(camera_source)
        # Ép sử dụng codec MJPEG (nếu có thể) cho ổn định
        video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    if not use_ffmpeg:
        if not video_capture.isOpened():
            messagebox.showerror("Lỗi Camera", "Không thể truy cập camera!")
            return

    print("Bắt đầu nhận diện khuôn mặt theo thời gian thực. Nhấn 'q' để thoát.")
    threshold = 0.6  # Ngưỡng KNN

    while True:
        if use_ffmpeg:
            raw_frame = pipe.stdout.read(frame_width * frame_height * 3)
            if len(raw_frame) != frame_width * frame_height * 3:
                break
            frame = np.frombuffer(raw_frame, np.uint8).reshape((frame_height, frame_width, 3))
        else:
            ret, frame = video_capture.read()
            if not ret:
                print("Không thể đọc khung hình từ camera. Kiểm tra kết nối hoặc URL của camera.")
                break

        processed_frame, scale = preprocess_frame(frame, target_width=1200)
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        face_locations = []
        face_results = []
        face_crops_list = []  # dùng cho xác thực qua API

        boxes, _ = mtcnn.detect(rgb_frame)
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                face_crop = rgb_frame[y1:y2, x1:x2]
                try:
                    face_crop_resized = cv2.resize(face_crop, (160, 160), interpolation=cv2.INTER_LINEAR)
                except Exception as e:
                    continue
                face_tensor = torch.tensor(face_crop_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
                face_tensor = face_tensor.unsqueeze(0).to(device)
                face_locations.append((y1, x2, y2, x1))  # (top, right, bottom, left)
                face_crops_list.append(face_crop)  # dùng crop gốc (RGB) cho API

            if face_crops_list:
                faces_batch = torch.stack([
                    torch.tensor(cv2.resize(crop, (160, 160), interpolation=cv2.INTER_LINEAR),
                                 dtype=torch.float32).permute(2, 0, 1) / 255.0
                    for crop in face_crops_list
                ]).to(device)
                with torch.no_grad():
                    embeddings = resnet(faces_batch).detach().cpu().numpy()
                for idx, emb in enumerate(embeddings):
                    if knn_clf is not None:
                        distances, indices = knn_clf.kneighbors([emb], n_neighbors=1)
                        if distances[0][0] < threshold:
                            candidate_id = knn_clf.predict([emb])[0]
                            name = student_dict.get(candidate_id, "Unknown")
                            if USE_EXTERNAL_API and candidate_id in reference_images:
                                ref_img = reference_images[candidate_id]
                                face_crop_bgr = cv2.cvtColor(face_crops_list[idx], cv2.COLOR_RGB2BGR)
                                if not verify_face_with_api(face_crop_bgr, ref_img):
                                    name = "Unknown"
                        else:
                            name = "Unknown"
                    else:
                        name = "Unknown"
                    face_results.append((name, candidate_id if name != "Unknown" else None))

        now = datetime.now()
        detection_timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        deadline = now.replace(hour=7, minute=0, second=0, microsecond=0)
        if len(face_locations) == len(face_results):
            for (name, student_id), (top, right, bottom, left) in zip(face_results, face_locations):
                top = int(top / scale)
                right = int(right / scale)
                bottom = int(bottom / scale)
                left = int(left / scale)
                if name != "Unknown" and student_id is not None:
                    status = "✓" if now <= deadline else "Muộn"
                    update_attendance(cursor, cnx, int(student_id), status, detection_timestamp)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow("Automated-Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if use_ffmpeg:
        pipe.stdout.close()
        pipe.stderr.close()
        pipe.terminate()
    else:
        video_capture.release()
    cv2.destroyAllWindows()
    cursor.close()
    cnx.close()


if __name__ == "__main__":
    from tkinter import messagebox

    messagebox.showinfo("Thông báo", "Hãy chạy FacialRecognition thông qua giao diện GUI!")
