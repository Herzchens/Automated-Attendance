# FacialRecognition.py

import cv2
import face_recognition
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter import simpledialog, messagebox

# Các hàm thao tác với cơ sở dữ liệu được lấy từ DatabaseHooking
from DatabaseHooking import get_all_students, update_attendance


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
    Lấy danh sách học sinh từ DB, load ảnh từ ImagePath, và mã hóa khuôn mặt.
    Trả về danh sách các dict chứa: id, name, class, encoding.
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
        encodings = face_recognition.face_encodings(rgb_img)
        if encodings:
            known_faces.append({
                "id": student_id,
                "name": HoVaTen,
                "class": Lop,
                "encoding": encodings[0]
            })
        else:
            print(f"Không tìm thấy khuôn mặt trong ảnh: {ImagePath}")
    return known_faces


def main(cnx, cursor):
    """
    Hàm chính nhận diện khuôn mặt và cập nhật điểm danh.
    Tham số:
      - cnx, cursor: kết nối đến MySQL được thiết lập từ GUI (sau khi đăng nhập thành công).
    """
    # Load danh sách học sinh và mã hóa khuôn mặt
    known_faces = load_known_faces(cursor)
    if not known_faces:
        messagebox.showerror("Lỗi", "Không có khuôn mặt nào được tải từ DB!")
        return

    known_encodings = [face["encoding"] for face in known_faces]
    known_ids = [face["id"] for face in known_faces]
    known_names = [face["name"] for face in known_faces]

    # Lấy nguồn camera (URL hoặc webcam)
    camera_source = get_camera_source()
    video_capture = cv2.VideoCapture(camera_source)
    if not video_capture.isOpened():
        messagebox.showerror("Lỗi Camera", "Không thể truy cập camera!")
        return

    print("Bắt đầu nhận diện khuôn mặt. Nhấn 'q' để thoát.")
    process_this_frame = True  # Xử lý xen kẽ mỗi frame để tăng hiệu suất

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Không thể đọc khung hình từ camera.")
            break

        # Resize frame để tăng tốc xử lý
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        if process_this_frame:
            # Nhận diện vị trí khuôn mặt và mã hóa
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_results = []
            for face_encoding in face_encodings:
                # So sánh với các khuôn mặt đã biết
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                name = "Unknown"
                student_id = None
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_names[best_match_index]
                        student_id = known_ids[best_match_index]
                face_results.append((name, student_id))

        process_this_frame = not process_this_frame  # Xen kẽ xử lý frame để giảm tải

        now = datetime.now()
        deadline = now.replace(hour=7, minute=0, second=0, microsecond=0)
        # Vẽ khung quanh khuôn mặt và cập nhật điểm danh
        for (name, student_id), (top, right, bottom, left) in zip(face_results, face_locations):
            if name != "Unknown" and student_id is not None:
                status = "✓" if now <= deadline else "Muộn"
                update_attendance(cursor, cnx, student_id, status, now)

            # Vì đã xử lý trên frame đã giảm kích thước, nhân đôi tọa độ để vẽ trên frame gốc
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow("Facial Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    cursor.close()
    cnx.close()


if __name__ == "__main__":
    # Nếu chạy file này độc lập, thông báo rằng cần được gọi thông qua GUI (với kết nối DB đã thiết lập)
    messagebox.showinfo("Thông báo", "Hãy chạy FacialRecognition thông qua giao diện GUI!")
