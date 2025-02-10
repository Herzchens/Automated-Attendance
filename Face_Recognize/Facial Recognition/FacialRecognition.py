# FacialRecognition.py
import cv2
import face_recognition
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox
from datetime import datetime

from DatabaseHooking import connect_db, create_tables, get_all_students, update_attendance


def get_db_credentials():
    """
    Lấy thông tin đăng nhập DB qua hộp thoại.
    """
    root = tk.Tk()
    root.withdraw()  # Ẩn cửa sổ chính
    user = simpledialog.askstring("Database Login", "Nhập DB username:")
    password = simpledialog.askstring("Database Login", "Nhập DB password:", show="*")
    host = simpledialog.askstring("Database Login", "Nhập DB host:", initialvalue="localhost")
    root.destroy()
    return user, password, host


def load_known_faces(cursor):
    """
    Lấy danh sách học sinh từ DB, load ảnh từ ImagePath, và mã hóa khuôn mặt.
    Trả về danh sách dict chứa: id, name, class, encoding.
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


def main():
    user, password, host = get_db_credentials()
    cnx, cursor = connect_db(user, password, host)
    if cnx is None:
        messagebox.showerror("Lỗi kết nối DB", "Không thể kết nối đến cơ sở dữ liệu!")
        return
    create_tables(cursor)

    known_faces = load_known_faces(cursor)
    known_encodings = [face["encoding"] for face in known_faces]
    known_ids = [face["id"] for face in known_faces]
    known_names = [face["name"] for face in known_faces]

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        messagebox.showerror("Lỗi Webcam", "Không thể truy cập webcam!")
        return

    print("Bắt đầu nhận diện khuôn mặt. Nhấn 'q' để thoát.")
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Không thể đọc khung hình từ webcam.")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_results = []
        for face_encoding in face_encodings:
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

        now = datetime.now()
        deadline = now.replace(hour=7, minute=0, second=0, microsecond=0)
        for (name, student_id), (top, right, bottom, left) in zip(face_results, face_locations):
            if name != "Unknown" and student_id is not None:
                status = "✓" if now <= deadline else "Muộn"
                update_attendance(cursor, cnx, student_id, status, now)

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
    main()
