import cv2
import math
import torch
import pickle
import base64
import json
import os
import threading
import face_recognition
import numpy as np
from sklearn import neighbors
import customtkinter as ctk

from DatabaseHooking import connect_db
from Image_Utilities import (
    SuperResolution,
    ImageSharpening,
    ImageDenoising,
    ColorBrightnessAdjustment,
    GeometricEnhancements,
    ControlledBlurring,
    EdgeEnhancement,
    FrequencyDomainProcessing,
    FaceRecognitionEnhancement,
    DistortionCorrection,
    VideoStabilization
)

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

db_username = base64.b64decode(config.get("db_username", "")).decode("utf-8")
db_password = base64.b64decode(config.get("db_password", "")).decode("utf-8")
db_host = config.get("db_host", "localhost")
camera_type = config.get("camera_type", "Webcam mặc định")
camera_url = config.get("camera_url", "")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Sử dụng thiết bị:", device)

def train_from_db(cursor, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    X = []
    y = []
    query = "SELECT HoVaTen, ImagePath FROM Students"
    cursor.execute(query)
    results = cursor.fetchall()
    if verbose:
        print(f"Đã lấy được {len(results)} mẫu từ CSDL.")
    for record in results:
        name, image_path = record
        if not os.path.exists(image_path):
            if verbose:
                print(f"Không tìm thấy file: {image_path}")
            continue
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) != 1:
            if verbose:
                msg = "Không phát hiện khuôn mặt" if len(face_locations) < 1 else "Phát hiện nhiều khuôn mặt"
                print(f"Ảnh {image_path} không phù hợp: {msg}")
            continue
        encoding = face_recognition.face_encodings(image, known_face_locations=face_locations)[0]
        X.append(encoding)
        y.append(name)
    if len(X) == 0:
        raise Exception("Không có dữ liệu huấn luyện hợp lệ.")
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

toggle_items = [
    ("Siêu phân giải (Bicubic)", "SuperResolution_bicubic_interpolation"),
    ("Làm sắc nét (Unsharp Masking)", "ImageSharpening_unsharp_masking"),
    ("Lọc cao tần (High-pass Filtering)", "ImageSharpening_high_pass_filtering"),
    ("Làm sắc nét (Laplacian)", "ImageSharpening_laplacian_sharpening"),
    ("Làm sắc nét (Gradient Based)", "ImageSharpening_gradient_based_sharpening"),
    ("Khử mờ Wiener", "ImageSharpening_wiener_deconvolution"),
    ("Lọc nhiễu Gaussian", "ImageDenoising_gaussian_filtering"),
    ("Lọc nhiễu Median", "ImageDenoising_median_filtering"),
    ("Lọc nhiễu Bilateral", "ImageDenoising_bilateral_filtering"),
    ("Lọc nhiễu Non-local", "ImageDenoising_non_local_means_denoising"),
    ("Lọc nhiễu Wavelet", "ImageDenoising_wavelet_denoising"),
    ("Lọc nhiễu Diffusion", "ImageDenoising_anisotropic_diffusion"),
    ("Chỉnh sửa Gamma", "ColorBrightnessAdjustment_gamma_correction"),
    ("Cân bằng Histogram", "ColorBrightnessAdjustment_histogram_equalization"),
    ("Histogram thích ứng", "ColorBrightnessAdjustment_adaptive_histogram_equalization"),
    ("Retinex", "ColorBrightnessAdjustment_retinex_algorithm"),
    ("Chỉnh sửa trắng cân", "ColorBrightnessAdjustment_white_balance_correction"),
    ("Thay đổi kích thước", "GeometricEnhancements_scaling_resampling"),
    ("Xoay", "GeometricEnhancements_rotation"),
    ("Biến đổi phối cảnh", "GeometricEnhancements_perspective_transformation"),
    ("Toán tử hình học", "GeometricEnhancements_morphological_operations"),
    ("Làm mờ Gaussian", "ControlledBlurring_gaussian_blur"),
    ("Mô phỏng mờ chuyển động", "ControlledBlurring_motion_blur_simulation"),
    ("Mờ zoom tròn", "ControlledBlurring_radial_zoom_blur"),
    ("Mờ bề mặt", "ControlledBlurring_surface_blur"),
    ("Phát hiện cạnh", "EdgeEnhancement_edge_detection"),
    ("Xử lý gradient", "EdgeEnhancement_gradient_domain_processing"),
    ("Biến đổi Fourier", "FrequencyDomainProcessing_fourier_transform_processing"),
    ("Lọc cao/thấp", "FrequencyDomainProcessing_high_low_pass_filtering"),
    ("Biến đổi Wavelet", "FrequencyDomainProcessing_wavelet_transform"),
    ("Log Transformation", "FaceRecognitionEnhancement_log_transformation"),
    ("Power Law Transformation", "FaceRecognitionEnhancement_power_law_transformation"),
    ("Contrast Stretching", "FaceRecognitionEnhancement_contrast_stretching"),
    ("Chuyển đổi màu", "FaceRecognitionEnhancement_color_space_conversion"),
    ("Lọc cạnh tinh tế", "FaceRecognitionEnhancement_edge_aware_filtering"),
    ("High Boost Filtering", "DistortionCorrection_high_boost_filtering"),
    ("Ổn định video", "VideoStabilization")
]

toggles = {}
for disp, key in toggle_items:
    toggles[key] = config.get(key, False)

def update_config_file():
    for key in toggles:
        config[key] = toggles[key]
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

def create_manual_config_window():
    manual_window = ctk.CTkToplevel()
    manual_window.title("Chỉnh sửa thủ công cấu hình")
    manual_window.geometry("400x500")
    frame = ctk.CTkFrame(manual_window)
    frame.pack(fill="both", expand=True, padx=10, pady=10)

    manual_keys = [
        "theme", "language", "db_host", "camera_type", "camera_url",
        "camera_simple_mode", "camera_protocol", "camera_user",
        "camera_pass", "camera_ip", "camera_port"
    ]
    entries = {}
    for key in manual_keys:
        label = ctk.CTkLabel(frame, text=key)
        label.pack(pady=(10,0), anchor="w")
        entry = ctk.CTkEntry(frame)
        entry.insert(0, str(config.get(key, "")))
        entry.pack(pady=(0,5), fill="x", padx=5)
        entries[key] = entry

    def save_manual_config():
        for key, entry in entries.items():
            value = entry.get()
            if key == "camera_simple_mode":
                config[key] = True if value.lower() in ["true", "1", "yes"] else False
            else:
                config[key] = value
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        manual_window.destroy()
        print("Cấu hình thủ công đã được lưu.")

    btn_save = ctk.CTkButton(frame, text="Lưu cấu hình thủ công", command=save_manual_config)
    btn_save.pack(pady=20)

control_window = None
def create_control_window():
    global control_window
    control_window = ctk.CTk()
    control_window.title("Control Panel")
    control_window.geometry("300x700")
    frame = ctk.CTkFrame(control_window)
    frame.pack(fill="both", expand=True, padx=10, pady=10)
    toggle_vars = {}
    for disp, key in toggle_items:
        var = ctk.BooleanVar(value=toggles[key])
        toggle_vars[key] = var
        switch = ctk.CTkSwitch(master=frame, text=disp, variable=var, command=lambda k=key, v=var: on_toggle(k, v))
        switch.pack(pady=3, anchor="w")
    btn_manual = ctk.CTkButton(master=frame, text="Chỉnh sửa cấu hình thủ công", command=create_manual_config_window)
    btn_manual.pack(pady=10)
    btn_hide = ctk.CTkButton(master=frame, text="Hide Panel", command=lambda: control_window.withdraw())
    btn_hide.pack(pady=10)
    control_window.protocol("WM_DELETE_WINDOW", lambda: control_window.withdraw())
    control_window.mainloop()

def on_toggle(key, var):
    toggles[key] = var.get()
    update_config_file()
def apply_enhancements(frame):
    if toggles["SuperResolution_bicubic_interpolation"]:
        frame = SuperResolution.bicubic_interpolation(frame)
    if toggles["ImageSharpening_unsharp_masking"]:
        frame = ImageSharpening.unsharp_masking(frame)
    if toggles["ImageSharpening_high_pass_filtering"]:
        frame = ImageSharpening.high_pass_filtering(frame)
    if toggles["ImageSharpening_laplacian_sharpening"]:
        frame = ImageSharpening.laplacian_sharpening(frame)
    if toggles["ImageSharpening_gradient_based_sharpening"]:
        frame = ImageSharpening.gradient_based_sharpening(frame)
    if toggles["ImageSharpening_wiener_deconvolution"]:
        frame = ImageSharpening.wiener_deconvolution(frame)
    if toggles["ImageDenoising_gaussian_filtering"]:
        frame = ImageDenoising.gaussian_filtering(frame)
    if toggles["ImageDenoising_median_filtering"]:
        frame = ImageDenoising.median_filtering(frame)
    if toggles["ImageDenoising_bilateral_filtering"]:
        frame = ImageDenoising.bilateral_filtering(frame)
    if toggles["ImageDenoising_non_local_means_denoising"]:
        frame = ImageDenoising.non_local_means_denoising(frame)
    if toggles["ImageDenoising_wavelet_denoising"]:
        frame = ImageDenoising.wavelet_denoising(frame)
    if toggles["ImageDenoising_anisotropic_diffusion"]:
        frame = ImageDenoising.anisotropic_diffusion(frame)
    if toggles["ColorBrightnessAdjustment_gamma_correction"]:
        frame = ColorBrightnessAdjustment.gamma_correction(frame)
    if toggles["ColorBrightnessAdjustment_histogram_equalization"]:
        frame = ColorBrightnessAdjustment.histogram_equalization(frame)
    if toggles["ColorBrightnessAdjustment_adaptive_histogram_equalization"]:
        frame = ColorBrightnessAdjustment.adaptive_histogram_equalization(frame)
    if toggles["ColorBrightnessAdjustment_retinex_algorithm"]:
        frame = ColorBrightnessAdjustment.retinex_algorithm(frame)
    if toggles["ColorBrightnessAdjustment_white_balance_correction"]:
        frame = ColorBrightnessAdjustment.white_balance_correction(frame)
    if toggles["GeometricEnhancements_scaling_resampling"]:
        frame = GeometricEnhancements.scaling_resampling(frame, scale_factor=1.0)
    if toggles["GeometricEnhancements_rotation"]:
        frame = GeometricEnhancements.rotation(frame, angle=0)
    if toggles["GeometricEnhancements_perspective_transformation"]:
        h, w = frame.shape[:2]
        src = [[0,0], [w,0], [w,h], [0,h]]
        dst = [[0,0], [w,0], [w,h], [0,h]]
        frame = GeometricEnhancements.perspective_transformation(frame, src, dst)
    if toggles["GeometricEnhancements_morphological_operations"]:
        frame = GeometricEnhancements.morphological_operations(frame, operation='opening')
    if toggles["ControlledBlurring_gaussian_blur"]:
        frame = ControlledBlurring.gaussian_blur(frame)
    if toggles["ControlledBlurring_motion_blur_simulation"]:
        frame = ControlledBlurring.motion_blur_simulation(frame, kernel_size=15, angle=0)
    if toggles["ControlledBlurring_radial_zoom_blur"]:
        frame = ControlledBlurring.radial_zoom_blur(frame)
    if toggles["ControlledBlurring_surface_blur"]:
        frame = ControlledBlurring.surface_blur(frame)
    if toggles["EdgeEnhancement_edge_detection"]:
        frame = EdgeEnhancement.edge_detection(frame, method='canny')
    if toggles["EdgeEnhancement_gradient_domain_processing"]:
        frame = EdgeEnhancement.gradient_domain_processing(frame)
    if toggles["FrequencyDomainProcessing_fourier_transform_processing"]:
        frame = FrequencyDomainProcessing.fourier_transform_processing(frame)
    if toggles["FrequencyDomainProcessing_high_low_pass_filtering"]:
        frame = FrequencyDomainProcessing.high_low_pass_filtering(frame, filter_type='high', cutoff=30)
    if toggles["FrequencyDomainProcessing_wavelet_transform"]:
        frame = FrequencyDomainProcessing.wavelet_transform(frame, wavelet='db1', level=1)[0]
    if toggles["FaceRecognitionEnhancement_log_transformation"]:
        frame = FaceRecognitionEnhancement.log_transformation(frame)
    if toggles["FaceRecognitionEnhancement_power_law_transformation"]:
        frame = FaceRecognitionEnhancement.power_law_transformation(frame)
    if toggles["FaceRecognitionEnhancement_contrast_stretching"]:
        frame = FaceRecognitionEnhancement.contrast_stretching(frame)
    if toggles["FaceRecognitionEnhancement_color_space_conversion"]:
        frame = FaceRecognitionEnhancement.color_space_conversion(frame, conversion='YCrCb')
    if toggles["FaceRecognitionEnhancement_edge_aware_filtering"]:
        frame = FaceRecognitionEnhancement.edge_aware_filtering(frame)
    if toggles["DistortionCorrection_high_boost_filtering"]:
        frame = DistortionCorrection.high_boost_filtering(frame)
    if toggles["VideoStabilization"]:
        if not hasattr(apply_enhancements, "stabilizer"):
            apply_enhancements.stabilizer = VideoStabilization()
        frame = apply_enhancements.stabilizer.stabilize_frame(frame)
    return frame

def predict(frame, knn_clf=None, model_path=None, distance_threshold=0.5):
    if knn_clf is None and model_path is None:
        raise Exception("Phải cung cấp knn classifier.")
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)
    faces = face_recognition.face_locations(frame)
    if len(faces) == 0:
        return []
    encodings = face_recognition.face_encodings(frame, known_face_locations=faces)
    distances = knn_clf.kneighbors(encodings, n_neighbors=1)
    matches = [distances[0][i][0] <= distance_threshold for i in range(len(faces))]
    return [(pred, loc) if m else ("unknown", loc)
            for pred, loc, m in zip(knn_clf.predict(encodings), faces, matches)]

def show_labels(frame, predictions):
    from PIL import Image, ImageDraw, ImageFont
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()
    for name, (top, right, bottom, left) in predictions:
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
        txt = str(name)
        bbox = draw.textbbox((0, 0), txt, font=font)
        draw.rectangle(((left, bottom - (bbox[3]-bbox[1]) - 10), (right, bottom)), fill=(0, 0, 255))
        draw.text((left + 6, bottom - (bbox[3]-bbox[1]) - 5), txt, fill=(255, 255, 255), font=font)
    return np.array(pil_img)

global_control_panel = None

def face_loop(cnx, cursor, camera_source):
    knn_model_path = "trained_knn_model.clf"
    if not os.path.exists(knn_model_path):
        print("Không tồn tại model, huấn luyện từ CSDL...")
        knn_clf = train_from_db(cursor, model_save_path=knn_model_path, verbose=True)
    else:
        with open(knn_model_path, 'rb') as f:
            knn_clf = pickle.load(f)
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print("Không mở được camera!")
        return
    buffer_dict = {}
    thresh = 3
    decay = 1
    confirmed = set()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = apply_enhancements(frame)
        predictions = []
        try:
            predictions = predict(frame, knn_clf=knn_clf, distance_threshold=0.5)
        except Exception as e:
            print("Lỗi nhận diện:", e)
        names = set([p for p, _ in predictions if p != "unknown"])
        for name in list(buffer_dict.keys()):
            if name in names:
                buffer_dict[name] += 1
            else:
                buffer_dict[name] = max(buffer_dict[name] - decay, 0)
                if buffer_dict[name] == 0:
                    del buffer_dict[name]
        for name in names:
            if name not in buffer_dict:
                buffer_dict[name] = 1
        for name, count in buffer_dict.items():
            if count >= thresh and name not in confirmed:
                confirmed.add(name)
                print(f"Đã điểm danh: {name}")
        out_frame = show_labels(frame, predictions)
        cv2.imshow("Face Recognition", out_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('i') or key == ord('I'):
            if global_control_panel is None or not cv2.getWindowProperty("Control Panel", 0) >= 0:
                threading.Thread(target=create_control_window, daemon=True).start()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cnx, cursor = connect_db(db_username, db_password, db_host)
    if cnx is None or cursor is None:
        print("Lỗi kết nối CSDL MySQL!")
        exit(1)
    threading.Thread(target=create_control_window, daemon=True).start()
    camera_source = 0
    face_loop(cnx, cursor, camera_source)
