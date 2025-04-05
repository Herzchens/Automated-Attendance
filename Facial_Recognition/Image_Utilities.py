import cv2
import numpy as np
import pywt
from scipy.signal import wiener
from skimage.restoration import denoise_wavelet, denoise_nl_means
from skimage.util import img_as_ubyte
from skimage.filters import sobel


# ====================================================
# 1. Tăng độ phân giải (Super-Resolution)
# ====================================================
class SuperResolution:
    @staticmethod
    def bicubic_interpolation(image, scale_factor=2):
        h, w = image.shape[:2]
        new_dim = (int(w * scale_factor), int(h * scale_factor))
        return cv2.resize(image, new_dim, interpolation=cv2.INTER_CUBIC)

# ====================================================
# 2. Làm nét ảnh (Image Sharpening)
# ====================================================
class ImageSharpening:
    @staticmethod
    def unsharp_masking(image, kernel_size=(5,5), sigma=1.0, amount=1.5):
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)

    @staticmethod
    def high_pass_filtering(image):
        kernel = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]])
        high_pass = cv2.filter2D(image, -1, kernel)
        return cv2.addWeighted(image, 1, high_pass, 0.5, 0)

    @staticmethod
    def laplacian_sharpening(image):
        lap = cv2.Laplacian(image, cv2.CV_64F)
        lap = cv2.convertScaleAbs(lap)
        return cv2.addWeighted(image, 1, lap, 1, 0)

    @staticmethod
    def gradient_based_sharpening(image):
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient = cv2.magnitude(sobelx, sobely)
        gradient = cv2.convertScaleAbs(gradient)
        return cv2.addWeighted(image, 1, gradient, 0.5, 0)

    @staticmethod
    def wiener_deconvolution(image, kernel_size=5):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        deconv = wiener(gray, (kernel_size, kernel_size))
        return np.clip(deconv, 0, 255).astype(np.uint8)

# ====================================================
# 3. Khử nhiễu ảnh (Image Denoising)
# ====================================================
class ImageDenoising:
    @staticmethod
    def gaussian_filtering(image, sigma=1):
        return cv2.GaussianBlur(image, (0, 0), sigma)

    @staticmethod
    def median_filtering(image, kernel_size=3):
        return cv2.medianBlur(image, kernel_size)

    @staticmethod
    def bilateral_filtering(image, diameter=9, sigmaColor=75, sigmaSpace=75):
        return cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)

    @staticmethod
    def non_local_means_denoising(image, h=10, patch_size=7, patch_distance=11):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        denoised = denoise_nl_means(image, h=h, patch_size=patch_size, patch_distance=patch_distance, multichannel=True)
        return img_as_ubyte(denoised)

    @staticmethod
    def wavelet_denoising(image, wavelet='db1', mode='soft', rescale_sigma=True):
        if len(image.shape) == 3:
            channels = cv2.split(image)
            denoised_channels = []
            for ch in channels:
                denoised = denoise_wavelet(ch, wavelet=wavelet, mode=mode, rescale_sigma=rescale_sigma)
                denoised_channels.append(img_as_ubyte(denoised))
            return cv2.merge(denoised_channels)
        else:
            denoised = denoise_wavelet(image, wavelet=wavelet, mode=mode, rescale_sigma=rescale_sigma)
            return img_as_ubyte(denoised)

    @staticmethod
    def anisotropic_diffusion(image, num_iter=10, kappa=50, gamma=0.1):
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        for _ in range(num_iter):
            deltaN = np.roll(image, -1, axis=0) - image
            deltaS = np.roll(image, 1, axis=0) - image
            deltaE = np.roll(image, -1, axis=1) - image
            deltaW = np.roll(image, 1, axis=1) - image
            cN = np.exp(-(deltaN/kappa)**2)
            cS = np.exp(-(deltaS/kappa)**2)
            cE = np.exp(-(deltaE/kappa)**2)
            cW = np.exp(-(deltaW/kappa)**2)
            image = image + gamma * (cN*deltaN + cS*deltaS + cE*deltaE + cW*deltaW)
        return np.clip(image, 0, 255).astype(np.uint8)

# ====================================================
# 4. Cân bằng sáng và màu sắc (Color & Brightness Adjustment)
# ====================================================
class ColorBrightnessAdjustment:
    @staticmethod
    def gamma_correction(image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    @staticmethod
    def histogram_equalization(image):
        if len(image.shape) == 2:
            return cv2.equalizeHist(image)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    @staticmethod
    def adaptive_histogram_equalization(image, clipLimit=2.0, tileGridSize=(8,8)):
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        if len(image.shape) == 2:
            return clahe.apply(image)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:,:,0] = clahe.apply(ycrcb[:,:,0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    @staticmethod
    def retinex_algorithm(image, sigma=30):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32) + 1.0
        blur = cv2.GaussianBlur(image, (0,0), sigma)
        retinex = np.log(image) - np.log(blur + 1)
        retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
        return retinex.astype(np.uint8)

    @staticmethod
    def white_balance_correction(image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        avg_a = np.average(lab[:,:,1])
        avg_b = np.average(lab[:,:,2])
        lab[:,:,1] = lab[:,:,1] - ((avg_a - 128) * (lab[:,:,0] / 255.0) * 1.1)
        lab[:,:,2] = lab[:,:,2] - ((avg_b - 128) * (lab[:,:,0] / 255.0) * 1.1)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# ====================================================
# 5. Biến đổi hình dạng và cấu trúc ảnh (Geometric & Structural Enhancements)
# ====================================================
class GeometricEnhancements:
    @staticmethod
    def scaling_resampling(image, scale_factor=1.0, interpolation=cv2.INTER_LINEAR):
        h, w = image.shape[:2]
        new_dim = (int(w * scale_factor), int(h * scale_factor))
        return cv2.resize(image, new_dim, interpolation=interpolation)

    @staticmethod
    def rotation(image, angle, center=None, scale=1.0):
        h, w = image.shape[:2]
        if center is None:
            center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        return cv2.warpAffine(image, M, (w, h))

    @staticmethod
    def perspective_transformation(image, src_points, dst_points):
        M = cv2.getPerspectiveTransform(np.float32(src_points), np.float32(dst_points))
        h, w = image.shape[:2]
        return cv2.warpPerspective(image, M, (w, h))

    @staticmethod
    def morphological_operations(image, operation='opening', kernel_size=3, iterations=1):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        if operation == 'erosion':
            return cv2.erode(image, kernel, iterations=iterations)
        elif operation == 'dilation':
            return cv2.dilate(image, kernel, iterations=iterations)
        elif operation == 'opening':
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif operation == 'closing':
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        else:
            raise ValueError("Unsupported morphological operation")

# ====================================================
# 6. Làm mờ ảnh có kiểm soát (Controlled Blurring & Smoothing)
# ====================================================
class ControlledBlurring:
    @staticmethod
    def gaussian_blur(image, sigma=1):
        return cv2.GaussianBlur(image, (0, 0), sigma)

    @staticmethod
    def motion_blur_simulation(image, kernel_size=15, angle=0):
        M = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        angle_rad = np.deg2rad(angle)
        cos_val = np.cos(angle_rad)
        sin_val = np.sin(angle_rad)
        for i in range(kernel_size):
            x = int(center + (i - center) * cos_val)
            y = int(center + (i - center) * sin_val)
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                M[y, x] = 1
        M = M / np.sum(M)
        return cv2.filter2D(image, -1, M)

    @staticmethod
    def radial_zoom_blur(image, strength=0.5):
        h, w = image.shape[:2]
        blurred = np.zeros_like(image, dtype=np.float32)
        steps = 10
        for i in range(steps):
            scale = 1 + strength * (i / steps)
            resized = cv2.resize(image, (int(w/scale), int(h/scale)))
            resized = cv2.resize(resized, (w, h))
            blurred = blurred + resized.astype(np.float32)
        blurred /= steps
        return blurred.astype(np.uint8)

    @staticmethod
    def surface_blur(image, sigma_space=10, sigma_r=0.15):
        return cv2.edgePreservingFilter(image, flags=1, sigma_s=sigma_space, sigma_r=sigma_r)

# ====================================================
# 7. Tăng cường cạnh và phát hiện biên (Edge & Contrast Enhancements)
# ====================================================
class EdgeEnhancement:
    @staticmethod
    def edge_detection(image, method='canny'):
        if method == 'sobel':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            grad = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
            return cv2.convertScaleAbs(grad)
        elif method == 'prewitt':
            kernelx = np.array([[1, 0, -1],
                                [1, 0, -1],
                                [1, 0, -1]])
            kernely = np.array([[1, 1, 1],
                                [0, 0, 0],
                                [-1, -1, -1]])
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            grad_x = cv2.filter2D(gray, -1, kernelx)
            grad_y = cv2.filter2D(gray, -1, kernely)
            gradient = cv2.addWeighted(np.abs(grad_x), 0.5, np.abs(grad_y), 0.5, 0)
            return gradient
        elif method == 'canny':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.Canny(gray, 100, 200)
        else:
            raise ValueError("Phương pháp edge detection không được hỗ trợ")

    @staticmethod
    def gradient_domain_processing(image):
        grad = sobel(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX)
        grad = cv2.cvtColor(grad.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(image, 0.8, grad, 0.2, 0)

# ====================================================
# 8. Xử lý ảnh trong miền tần số (Frequency Domain Processing)
# ====================================================
class FrequencyDomainProcessing:
    @staticmethod
    def fourier_transform_processing(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        spectrum = 20 * np.log(np.abs(fshift) + 1)
        spectrum = cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX)
        return spectrum.astype(np.uint8)

    @staticmethod
    def high_low_pass_filtering(image, filter_type='high', cutoff=30):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros_like(gray, dtype=np.uint8)
        if filter_type == 'low':
            cv2.circle(mask, (ccol, crow), cutoff, 1, thickness=-1)
        elif filter_type == 'high':
            mask.fill(1)
            cv2.circle(mask, (ccol, crow), cutoff, 0, thickness=-1)
        else:
            raise ValueError("filter_type phải là 'high' hoặc 'low'")
        fshift_filtered = fshift * mask
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        return img_back.astype(np.uint8)

    @staticmethod
    def wavelet_transform(image, wavelet='db1', level=1):
        coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)
        return coeffs

# ====================================================
# 9. Cải thiện nhận diện khuôn mặt trên ảnh chất lượng thấp
# ====================================================
class FaceRecognitionEnhancement:
    @staticmethod
    def log_transformation(image, c=1):
        image_float = image.astype(np.float32)
        log_image = c * np.log1p(image_float)
        log_image = cv2.normalize(log_image, None, 0, 255, cv2.NORM_MINMAX)
        return log_image.astype(np.uint8)

    @staticmethod
    def power_law_transformation(image, c=1, gamma=1.0):
        image_float = image.astype(np.float32)
        power_image = c * np.power(image_float, gamma)
        power_image = cv2.normalize(power_image, None, 0, 255, cv2.NORM_MINMAX)
        return power_image.astype(np.uint8)

    @staticmethod
    def contrast_stretching(image):
        in_min = np.percentile(image, 2)
        in_max = np.percentile(image, 98)
        stretched = (image - in_min) * (255 / (in_max - in_min))
        stretched = np.clip(stretched, 0, 255)
        return stretched.astype(np.uint8)

    @staticmethod
    def color_space_conversion(image, conversion='YCrCb'):
        if conversion == 'YCrCb':
            return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        elif conversion == 'HSV':
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif conversion == 'Lab':
            return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        else:
            raise ValueError("Conversion không được hỗ trợ")

    @staticmethod
    def edge_aware_filtering(image):
        return cv2.bilateralFilter(image, 9, 75, 75)

# ====================================================
# 10. Hiệu chỉnh biến dạng do camera chất lượng thấp
# ====================================================
class DistortionCorrection:
    @staticmethod
    def radial_distortion_correction(image, camera_matrix, dist_coeffs):
        h, w = image.shape[:2]
        new_cam_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        return cv2.undistort(image, camera_matrix, dist_coeffs, None, new_cam_mtx)

    @staticmethod
    def high_boost_filtering(image, kernel_size=(3,3), boost=1.5):
        blurred = cv2.GaussianBlur(image, kernel_size, 0)
        high_freq = cv2.subtract(image, blurred)
        high_boost = cv2.addWeighted(image, 1, high_freq, boost, 0)
        return high_boost
class VideoStabilization:
    def __init__(self):
        self.prev_gray = None

    def stabilize_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return frame

        warp_matrix = np.eye(2, 3, dtype=np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)

        try:
            cc, warp_matrix = cv2.findTransformECC(self.prev_gray, gray, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria)
        except cv2.error as e:
            print("findTransformECC error:", e)

        stabilized_frame = cv2.warpAffine(frame, warp_matrix, (frame.shape[1], frame.shape[0]),
                                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        self.prev_gray = gray

        return stabilized_frame
