import cv2
import math
import torch
from sklearn import neighbors
import pickle
from PIL import Image, ImageDraw, ImageFont
import face_recognition
import numpy as np
import os
import json
from DatabaseHooking import connect_db

# Đọc cấu hình từ file config.json
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Sử dụng thiết bị:", device)

# ==================================
# Các biến bật/tắt (có thể cập nhật từ GUI)
# ==================================
enable_stabilization = config.get("enable_stabilization", True)
enable_clahe = config.get("enable_clahe", True)
enable_sharpening = config.get("enable_sharpening", True)
enable_denoising = config.get("enable_denoising", True)

# ==================================
# Các hàm xử lý ảnh
# ==================================
def apply_clahe(image):
    """Cân bằng sáng cục bộ (CLAHE)."""
    if not enable_clahe:
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return enhanced

def enhance_image(frame):
    """Kết hợp CLAHE và làm nét (sharpen) nếu bật."""
    enhanced = apply_clahe(frame)
    if enable_sharpening:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
    return enhanced

def preprocess_frame(frame, target_width=1200):
    """
    Resize ảnh nếu cần, sau đó áp dụng khử nhiễu, CLAHE và làm nét.
    """
    height, width = frame.shape[:2]
    scale = 1.0
    if width > target_width:
        scale = target_width / float(width)
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    if enable_denoising:
        frame = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
        frame = cv2.fastNlMeansDenoisingColored(frame, None, h=20, hColor=20,
                                                templateWindowSize=7, searchWindowSize=21)
    processed = enhance_image(frame)
    return processed, scale

def stabilize_frame(prev_gray, curr_gray, curr_frame):
    """
    Ổn định khung hình dựa trên optical flow.
    Nếu không bật hoặc không có khung hình trước, trả về ảnh hiện tại.
    """
    if not enable_stabilization or prev_gray is None:
        return curr_frame
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
    if prev_pts is None:
        return curr_frame
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    idx = np.where(status == 1)[0]
    if len(idx) < 10:
        return curr_frame
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]
    m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
    if m is None:
        return curr_frame
    stabilized = cv2.warpAffine(curr_frame, m, (curr_frame.shape[1], curr_frame.shape[0]))
    return stabilized

# ==================================
# Code huấn luyện KNN (nếu cần)
# ==================================
def train_from_db(cursor, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    X = []
    y = []
    query = "SELECT HoVaTen, ImagePath FROM Students"
    cursor.execute(query)
    results = cursor.fetchall()
    if verbose:
        print(f"Đã lấy được {len(results)} mẫu từ cơ sở dữ liệu.")
    for record in results:
        name, image_path = record
        if not os.path.exists(image_path):
            if verbose:
                print(f"Không tìm thấy file ảnh: {image_path}")
            continue
        image = face_recognition.load_image_file(image_path)
        face_bounding_boxes = face_recognition.face_locations(image)
        if len(face_bounding_boxes) != 1:
            if verbose:
                print(f"Ảnh {image_path} không phù hợp để huấn luyện: " +
                      ("Không phát hiện khuôn mặt" if len(face_bounding_boxes) < 1 else "Phát hiện nhiều khuôn mặt"))
            continue
        face_encoding = face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0]
        X.append(face_encoding)
        y.append(name)
    if len(X) == 0:
        raise Exception("Không có dữ liệu huấn luyện hợp lệ. Kiểm tra lại bảng Students và đường dẫn ảnh.")
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chọn n_neighbors tự động:", n_neighbors)
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)
    return knn_clf

# ==================================
# Hàm predict nhận diện khuôn mặt
# ==================================
def predict(X_frame, knn_clf=None, model_path=None, distance_threshold=0.5):
    if knn_clf is None and model_path is None:
        raise Exception("Phải cung cấp knn classifier qua knn_clf hoặc model_path")
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)
    X_face_locations = face_recognition.face_locations(X_frame)
    if len(X_face_locations) == 0:
        return []
    faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    return [(pred, loc) if rec else ("unknown", loc)
            for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

def show_prediction_labels_on_image(frame, predictions):
    """Vẽ bounding box và tên lên ảnh."""
    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.load_default()
    for name, (top, right, bottom, left) in predictions:
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
        name_text = str(name)
        text_bbox = draw.textbbox((0, 0), name_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name_text, fill=(255, 255, 255, 255), font=font)
    del draw
    opencvimage = np.array(pil_image)
    return opencvimage

# ==================================
# Hàm main chạy vòng lặp điểm danh với multi-frame verification & decay
# ==================================
def main(cnx, cursor, camera_source):
    """
    Hàm main lấy ảnh từ camera_source, xử lý ảnh, nhận diện khuôn mặt,
    và chỉ xác nhận điểm danh nếu cùng một tên được nhận diện liên tục qua nhiều khung hình.
    Cơ chế decay sẽ giảm dần số đếm nếu tên không được nhận diện ở khung hình hiện tại.
    """
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print("Không thể mở camera!")
        return

    knn_model_path = "trained_knn_model.clf"  # Đường dẫn đến model đã huấn luyện
    prev_gray = None
    detection_buffer = {}  # Lưu số lần nhận diện cho từng tên
    consecutive_frames_threshold = 3  # Số khung hình liên tục cần thiết để xác nhận
    decay_rate = 1  # Mỗi khung hình nếu không xuất hiện tên sẽ giảm đi 1
    confirmed_faces = set()  # Danh sách các khuôn mặt đã điểm danh

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, scale = preprocess_frame(frame, target_width=1200)
        curr_gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        if enable_stabilization:
            processed_frame = stabilize_frame(prev_gray, curr_gray, processed_frame)
        prev_gray = curr_gray

        try:
            predictions = predict(processed_frame, model_path=knn_model_path, distance_threshold=0.5)
        except Exception as e:
            print("Lỗi khi nhận diện:", e)
            predictions = []

        # Lấy tập các tên nhận diện từ khung hình hiện tại (bỏ qua "unknown")
        current_names = set([pred for pred, _ in predictions if pred != "unknown"])

        # Cập nhật detection_buffer: nếu tên xuất hiện tăng, nếu không giảm dần
        for name in list(detection_buffer.keys()):
            if name in current_names:
                detection_buffer[name] += 1
            else:
                detection_buffer[name] = max(detection_buffer[name] - decay_rate, 0)
                if detection_buffer[name] == 0:
                    del detection_buffer[name]
        for name in current_names:
            if name not in detection_buffer:
                detection_buffer[name] = 1

        # Xác nhận điểm danh nếu số lần nhận diện vượt ngưỡng và chưa được điểm danh
        for name, count in detection_buffer.items():
            if count >= consecutive_frames_threshold and name not in confirmed_faces:
                confirmed_faces.add(name)
                print(f"Đã điểm danh: {name}")

        output_frame = show_prediction_labels_on_image(processed_frame, predictions)
        cv2.imshow("Face Recognition", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Vui lòng chạy qua GUI.py để đảm bảo chất lượng")
