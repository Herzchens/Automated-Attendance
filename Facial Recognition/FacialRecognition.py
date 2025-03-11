import cv2
import torch
from datetime import datetime
import tkinter as tk
from tkinter import simpledialog, messagebox
from sklearn.neighbors import KNeighborsClassifier
from facenet_pytorch import MTCNN, InceptionResnetV1

from DatabaseHooking import get_all_students, update_attendance

device = torch.device("cpu")

# Khởi tạo mô hình phát hiện khuôn mặt MTCNN chạy trên CPU
mtcnn = MTCNN(keep_all=True, device=device)
# Khởi tạo mô hình InceptionResnetV1 để trích xuất embedding, chạy trên CPU
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def get_camera_source():
    """
    Lấy URL của camera qua hộp thoại.
    Nếu không nhập, mặc định sử dụng webcam (0).
    Ví dụ: "rtsp://username:password@ip_address:port/stream"
    """
    root = tk.Tk()
    root.withdraw()
    camera_source = simpledialog.askstring("Camera Source", "Nhập URL của camera (để trống nếu dùng webcam):")
    root.destroy()
    if not camera_source:
        return 0  # Sử dụng webcam mặc định
    return camera_source


def load_known_faces(cursor):
    """
    Load hình ảnh từ DB và sử dụng MTCNN, resnet để trích xuất embedding.
    Trả về danh sách các dict gồm: id, name, và embedding (numpy array 512-d).
    """
    students = get_all_students(cursor)
    known_faces = []
    for student in students:
        student_id, HoVaTen, Lop, ImagePath, status, attendance_time = student
        img = cv2.imread(ImagePath)
        if img is None:
            print(f"Không thể tải ảnh: {ImagePath}")
            continue
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb_img)
        if boxes is None:
            print(f"Không tìm thấy khuôn mặt trong ảnh: {ImagePath}")
            continue
        # Lấy box đầu tiên
        x1, y1, x2, y2 = boxes[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        face_crop = rgb_img[y1:y2, x1:x2]
        try:
            face_crop = cv2.resize(face_crop, (160, 160))
        except Exception as e:
            print(f"Lỗi resize ảnh: {ImagePath}", e)
            continue
        face_tensor = torch.tensor(face_crop, dtype=torch.float32).permute(2, 0, 1) / 255.0
        face_tensor = face_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = resnet(face_tensor).detach().cpu().numpy()[0]
        known_faces.append({
            "id": student_id,
            "name": HoVaTen,
            "embedding": embedding
        })
    return known_faces


def train_knn_classifier(known_faces):
    """
    Huấn luyện mô hình KNN với các embedding khuôn mặt đã load.
    Trả về:
      - knn_clf: mô hình KNN đã huấn luyện.
      - student_dict: bảng tra cứu từ student_id sang tên.
    """
    X = []
    y = []
    student_dict = {}
    for face in known_faces:
        X.append(face["embedding"])
        y.append(face["id"])
        student_dict[face["id"]] = face["name"]
    if len(X) == 0:
        raise ValueError("Không có embedding nào được load!")
    n_neighbors = min(3, len(X))
    knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric="euclidean")
    knn_clf.fit(X, y)
    return knn_clf, student_dict


def preprocess_frame(frame, target_width=600):
    """
    Tiền xử lý ảnh:
      - Nếu chiều rộng lớn hơn target_width, resize ảnh về kích thước phù hợp.
      - Áp dụng bộ lọc bilateral để giảm nhiễu.
    Trả về:
      - frame đã xử lý.
      - hệ số scale (để scale lại tọa độ).
    """
    height, width = frame.shape[:2]
    scale = 1.0
    if width > target_width:
        scale = target_width / float(width)
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    processed = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
    return processed, scale


def main(cnx, cursor, camera_source=None):
    """
    Hàm chính nhận diện khuôn mặt và cập nhật điểm danh.
    Sử dụng deep learning với MTCNN và resnet chạy trên CPU nhằm tối ưu cho GPU yếu.

    CHỈNH SỬA: Nếu không có khuôn mặt trong DB, vẫn mở camera và hiển thị cảnh báo.
    """
    known_faces = load_known_faces(cursor)
    if known_faces:
        knn_clf, student_dict = train_knn_classifier(known_faces)
    else:
        messagebox.showwarning("Cảnh báo",
                               "Không có khuôn mặt nào được tải từ DB! Hệ thống sẽ chạy ở chế độ không nhận dạng.")
        knn_clf = None
        student_dict = {}

    if camera_source is None:
        camera_source = get_camera_source()
    if isinstance(camera_source, str) and camera_source.lower().startswith("rtsp://"):
        video_capture = cv2.VideoCapture(camera_source, cv2.CAP_FFMPEG)
    else:
        video_capture = cv2.VideoCapture(camera_source)
    if not video_capture.isOpened():
        messagebox.showerror("Lỗi Camera", "Không thể truy cập camera!")
        return

    print("Bắt đầu nhận diện khuôn mặt. Nhấn 'q' để thoát.")
    process_this_frame = True
    threshold = 0.6  # Ngưỡng khoảng cách để xác định khớp

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Không thể đọc khung hình từ camera. Kiểm tra kết nối hoặc URL của camera.")
            break

        # Giảm độ phân giải để tăng tốc xử lý
        processed_frame, scale = preprocess_frame(frame, target_width=600)
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        face_locations = []
        face_results = []
        if process_this_frame:
            # Sử dụng MTCNN để phát hiện khuôn mặt
            boxes, _ = mtcnn.detect(rgb_frame)
            if boxes is not None:
                face_crops = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    face_crop = rgb_frame[y1:y2, x1:x2]
                    try:
                        face_crop = cv2.resize(face_crop, (160, 160))
                    except Exception as e:
                        continue
                    face_tensor = torch.tensor(face_crop, dtype=torch.float32).permute(2, 0, 1) / 255.0
                    face_crops.append(face_tensor)
                    face_locations.append((y1, x2, y2, x1))  # (top, right, bottom, left)
                if face_crops:
                    faces_batch = torch.stack(face_crops).to(device)
                    with torch.no_grad():
                        embeddings = resnet(faces_batch).detach().cpu().numpy()
                    for emb in embeddings:
                        if knn_clf is not None:
                            distances, indices = knn_clf.kneighbors([emb], n_neighbors=1)
                            if distances[0][0] < threshold:
                                student_id = knn_clf.predict([emb])[0]
                                name = student_dict.get(student_id, "Unknown")
                            else:
                                name = "Unknown"
                        else:
                            name = "Unknown"
                        face_results.append((name, None))
        process_this_frame = not process_this_frame

        now = datetime.now()
        deadline = now.replace(hour=7, minute=0, second=0, microsecond=0)
        if len(face_locations) == len(face_results):
            for (name, student_id), (top, right, bottom, left) in zip(face_results, face_locations):
                top = int(top / scale)
                right = int(right / scale)
                bottom = int(bottom / scale)
                left = int(left / scale)
                if name != "Unknown" and student_id is not None:
                    status = "✓" if now <= deadline else "Muộn"
                    update_attendance(cursor, cnx, student_id, status, now)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

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
