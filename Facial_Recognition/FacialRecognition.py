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

toggles = {
    "SuperResolution_bicubic_interpolation": config.get("SuperResolution_bicubic_interpolation", False),
    "ImageSharpening_unsharp_masking": config.get("ImageSharpening_unsharp_masking", False),
    "ImageSharpening_high_pass_filtering": config.get("ImageSharpening_high_pass_filtering", False),
    "ImageSharpening_laplacian_sharpening": config.get("ImageSharpening_laplacian_sharpening", False),
    "ImageSharpening_gradient_based_sharpening": config.get("ImageSharpening_gradient_based_sharpening", False),
    "ImageSharpening_wiener_deconvolution": config.get("ImageSharpening_wiener_deconvolution", False),
    "ImageDenoising_gaussian_filtering": config.get("ImageDenoising_gaussian_filtering", False),
    "ImageDenoising_median_filtering": config.get("ImageDenoising_median_filtering", False),
    "ImageDenoising_bilateral_filtering": config.get("ImageDenoising_bilateral_filtering", False),
    "ImageDenoising_non_local_means_denoising": config.get("ImageDenoising_non_local_means_denoising", False),
    "ImageDenoising_wavelet_denoising": config.get("ImageDenoising_wavelet_denoising", False),
    "ImageDenoising_anisotropic_diffusion": config.get("ImageDenoising_anisotropic_diffusion", False),
    "ColorBrightnessAdjustment_gamma_correction": config.get("ColorBrightnessAdjustment_gamma_correction", False),
    "ColorBrightnessAdjustment_histogram_equalization": config.get("ColorBrightnessAdjustment_histogram_equalization", False),
    "ColorBrightnessAdjustment_adaptive_histogram_equalization": config.get("ColorBrightnessAdjustment_adaptive_histogram_equalization", False),
    "ColorBrightnessAdjustment_retinex_algorithm": config.get("ColorBrightnessAdjustment_retinex_algorithm", False),
    "ColorBrightnessAdjustment_white_balance_correction": config.get("ColorBrightnessAdjustment_white_balance_correction", False),
    "GeometricEnhancements_scaling_resampling": config.get("GeometricEnhancements_scaling_resampling", False),
    "GeometricEnhancements_rotation": config.get("GeometricEnhancements_rotation", False),
    "GeometricEnhancements_perspective_transformation": config.get("GeometricEnhancements_perspective_transformation", False),
    "GeometricEnhancements_morphological_operations": config.get("GeometricEnhancements_morphological_operations", False),
    "ControlledBlurring_gaussian_blur": config.get("ControlledBlurring_gaussian_blur", False),
    "ControlledBlurring_motion_blur_simulation": config.get("ControlledBlurring_motion_blur_simulation", False),
    "ControlledBlurring_radial_zoom_blur": config.get("ControlledBlurring_radial_zoom_blur", False),
    "ControlledBlurring_surface_blur": config.get("ControlledBlurring_surface_blur", False),
    "EdgeEnhancement_edge_detection": config.get("EdgeEnhancement_edge_detection", False),
    "EdgeEnhancement_gradient_domain_processing": config.get("EdgeEnhancement_gradient_domain_processing", False),
    "FrequencyDomainProcessing_fourier_transform_processing": config.get("FrequencyDomainProcessing_fourier_transform_processing", False),
    "FrequencyDomainProcessing_high_low_pass_filtering": config.get("FrequencyDomainProcessing_high_low_pass_filtering", False),
    "FrequencyDomainProcessing_wavelet_transform": config.get("FrequencyDomainProcessing_wavelet_transform", False),
    "FaceRecognitionEnhancement_log_transformation": config.get("FaceRecognitionEnhancement_log_transformation", False),
    "FaceRecognitionEnhancement_power_law_transformation": config.get("FaceRecognitionEnhancement_power_law_transformation", False),
    "FaceRecognitionEnhancement_contrast_stretching": config.get("FaceRecognitionEnhancement_contrast_stretching", False),
    "FaceRecognitionEnhancement_color_space_conversion": config.get("FaceRecognitionEnhancement_color_space_conversion", False),
    "FaceRecognitionEnhancement_edge_aware_filtering": config.get("FaceRecognitionEnhancement_edge_aware_filtering", False),
    "DistortionCorrection_high_boost_filtering": config.get("DistortionCorrection_high_boost_filtering", False),
    "VideoStabilization": config.get("VideoStabilization", False)
}

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

global_control_panel = None

toggle_groups = {
    "Chỉnh sửa & Phục hồi": [
        ("Siêu phân giải (Bicubic)", "SuperResolution_bicubic_interpolation", "Tăng độ phân giải, giữ chi tiết gốc"),
        ("Làm sắc nét (Unsharp Masking)", "ImageSharpening_unsharp_masking", "Tăng độ sắc nét nhẹ"),
        ("Lọc cao tần (High-pass Filtering)", "ImageSharpening_high_pass_filtering", "Nâng cao chi tiết mà không lệch màu"),
        ("Làm sắc nét (Laplacian)", "ImageSharpening_laplacian_sharpening", "Sử dụng laplacian để tăng cạnh"),
        ("Làm sắc nét (Gradient Based)", "ImageSharpening_gradient_based_sharpening", "Dựa vào gradient, cải thiện đường biên"),
        ("Khử mờ Wiener", "ImageSharpening_wiener_deconvolution", "Giảm mờ, phục hồi độ rõ nét"),
        ("Lọc nhiễu Gaussian", "ImageDenoising_gaussian_filtering", "Giảm nhiễu nhẹ"),
        ("Lọc nhiễu Median", "ImageDenoising_median_filtering", "Loại bỏ nhiễu, giữ biên"),
        ("Lọc nhiễu Bilateral", "ImageDenoising_bilateral_filtering", "Giữ chi tiết sau khi giảm nhiễu"),
        ("Lọc nhiễu Non-local", "ImageDenoising_non_local_means_denoising", "Phức tạp, dành cho nghiên cứu"),
        ("Lọc nhiễu Wavelet", "ImageDenoising_wavelet_denoising", "Sử dụng wavelet cho giảm nhiễu"),
        ("Lọc nhiễu Diffusion", "ImageDenoising_anisotropic_diffusion", "Giảm nhiễu qua diffusion")
    ],
    "Điều chỉnh màu & sáng": [
        ("Chỉnh sửa Gamma", "ColorBrightnessAdjustment_gamma_correction", "Hiệu chỉnh gamma tự nhiên"),
        ("Cân bằng Histogram", "ColorBrightnessAdjustment_histogram_equalization", "Cân bằng sáng tối"),
        ("Histogram thích ứng", "ColorBrightnessAdjustment_adaptive_histogram_equalization", "Thích ứng theo vùng ảnh"),
        ("Retinex", "ColorBrightnessAdjustment_retinex_algorithm", "Nâng cao tương phản tự nhiên"),
        ("Chỉnh sửa trắng cân", "ColorBrightnessAdjustment_white_balance_correction", "Sửa lỗi màu sắc")
    ],
    "Biến đổi hình học nhẹ": [
        ("Thay đổi kích thước", "GeometricEnhancements_scaling_resampling", "Phóng to/thu nhỏ ảnh"),
        ("Xoay", "GeometricEnhancements_rotation", "Xoay ảnh nhẹ"),
        ("Biến đổi phối cảnh", "GeometricEnhancements_perspective_transformation", "Điều chỉnh phối cảnh"),
        ("Toán tử hình học", "GeometricEnhancements_morphological_operations", "Thao tác hình học cơ bản")
    ],
    "Hiệu ứng Blur": [
        ("Làm mờ Gaussian", "ControlledBlurring_gaussian_blur", "Mờ ảnh theo Gaussian"),
        ("Mô phỏng mờ chuyển động", "ControlledBlurring_motion_blur_simulation", "Mô phỏng chuyển động mờ"),
        ("Mờ zoom tròn", "ControlledBlurring_radial_zoom_blur", "Hiệu ứng zoom mờ"),
        ("Mờ bề mặt", "ControlledBlurring_surface_blur", "Làm mờ bề mặt tinh tế")
    ],
    "Biến đổi nâng cao": [
        ("Phát hiện cạnh", "EdgeEnhancement_edge_detection", "Trích xuất biên, hiệu ứng đen trắng"),
        ("Xử lý gradient", "EdgeEnhancement_gradient_domain_processing", "Tạo hiệu ứng gradient mạnh")
    ],
    "Xử lý trong miền tần số": [
        ("Biến đổi Fourier", "FrequencyDomainProcessing_fourier_transform_processing", "Phân tích theo tần số"),
        ("Lọc cao/thấp", "FrequencyDomainProcessing_high_low_pass_filtering", "Lọc theo tần số, nghiên cứu"),
        ("Biến đổi Wavelet", "FrequencyDomainProcessing_wavelet_transform", "Xử lý bằng wavelet")
    ],
    "Tăng cường khuôn mặt": [
        ("Log Transformation", "FaceRecognitionEnhancement_log_transformation", "Nâng cao tương phản khuôn mặt"),
        ("Power Law Transformation", "FaceRecognitionEnhancement_power_law_transformation", "Điều chỉnh ánh sáng khuôn mặt"),
        ("Contrast Stretching", "FaceRecognitionEnhancement_contrast_stretching", "Mở rộng tương phản khuôn mặt"),
        ("Chuyển đổi màu", "FaceRecognitionEnhancement_color_space_conversion", "Chuyển đổi không gian màu"),
        ("Lọc cạnh tinh tế", "FaceRecognitionEnhancement_edge_aware_filtering", "Tăng chi tiết khuôn mặt")
    ],
    "Khác": [
        ("High Boost Filtering", "DistortionCorrection_high_boost_filtering", "Tăng cường chi tiết đặc biệt"),
        ("Ổn định video", "VideoStabilization", "Giảm rung, ổn định video")
    ]
}

def create_control_window():
    global global_control_panel
    control_window = ctk.CTk()
    control_window.title("Control Panel")
    control_window.geometry("300x700")
    scrollable_frame = ctk.CTkScrollableFrame(control_window, width=280, height=680)
    scrollable_frame.pack(padx=10, pady=10, fill="both", expand=True)
    toggle_vars = {}

    for group_name, toggles_list in toggle_groups.items():
        group_label = ctk.CTkLabel(scrollable_frame, text=group_name, font=ctk.CTkFont(size=16, weight='bold'))
        group_label.pack(pady=(10, 5), anchor="w")
        for disp, key, note in toggles_list:
            subframe = ctk.CTkFrame(scrollable_frame)
            subframe.pack(fill="x", pady=3, padx=5)
            var = ctk.BooleanVar(value=toggles[key])
            toggle_vars[key] = var
            switch = ctk.CTkSwitch(master=subframe, text=disp, variable=var,
                                   command=lambda k=key, v=var: on_toggle(k, v))
            switch.pack(side="left")
            note_label = ctk.CTkLabel(subframe, text=note, fg_color="transparent", text_color="gray")
            note_label.pack(side="left", padx=10)
    btn_manual = ctk.CTkButton(master=scrollable_frame, text="Chỉnh sửa cấu hình thủ công", command=create_manual_config_window)
    btn_manual.pack(pady=10)
    btn_hide = ctk.CTkButton(master=scrollable_frame, text="Hide Panel", command=lambda: control_window.withdraw())
    btn_hide.pack(pady=10)
    control_window.protocol("WM_DELETE_WINDOW", lambda: control_window.withdraw())
    global_control_panel = control_window
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