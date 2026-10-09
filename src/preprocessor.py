from pathlib import Path
from typing import Union, Tuple
import cv2
import numpy as np
import tempfile
import io
from PIL import Image, ImageOps

# Enable HEIC / HEIF support for Apple iPhone photos
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

class ImagePreprocessor:
    """
    Optimized preprocessor for Indonesian KTP images.
    - Supports JPG, PNG, WEBP, HEIC/HEIF (iPhone photos).
    - Automatically corrects EXIF orientation (handles rotated iPhone camera shots).
    - Upscales small/low-res images to optimal OCR resolution (~900px height) using cubic interpolation.
    - Downscales oversized images to prevent latency blowup.
    - Preserves font edge clarity and avoids destructive contrast artifacting.
    """

    def __init__(self, target_height: int = 900, max_width: int = 1600, max_dim: int | None = None):
        self.target_height = target_height
        self.max_width = max_dim if max_dim is not None else max_width

    def load_image(self, image_input: Union[str, Path, bytes, np.ndarray]) -> np.ndarray:
        """
        Load image from path, bytes, or numpy array.
        Uses Pillow with HEIC support and EXIF auto-transposition to ensure
        photos from iPhones and cameras are oriented correctly.
        """
        if isinstance(image_input, np.ndarray):
            return image_input.copy()

        try:
            if isinstance(image_input, (str, Path)):
                pil_img = Image.open(str(image_input))
            elif isinstance(image_input, bytes):
                pil_img = Image.open(io.BytesIO(image_input))
            else:
                raise TypeError(f"Unsupported image input type: {type(image_input)}")

            # Automatically correct rotation according to EXIF metadata (crucial for smartphone photos)
            pil_img = ImageOps.exif_transpose(pil_img)

            # Ensure image is RGB (convert RGBA, Grayscale, Palette)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            # Convert PIL RGB to OpenCV BGR
            rgb_arr = np.array(pil_img)
            bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
            return bgr_arr

        except Exception as e:
            # Fallback to direct OpenCV reading if Pillow fails
            if isinstance(image_input, (str, Path)):
                img = cv2.imread(str(image_input))
                if img is not None:
                    return img
            elif isinstance(image_input, bytes):
                nparr = np.frombuffer(image_input, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    return img
            raise ValueError(f"Gagal membaca format gambar (pastikan file gambar valid atau berformat HEIC/JPG/PNG): {e}")

    def deskew(self, img: np.ndarray) -> np.ndarray:
        """
        Detects and corrects slight image tilt (rotational skew) common in smartphone photos.
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Edge detection with Canny
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=15)
            if lines is not None:
                angles = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    # Only consider subtle tilt angles between -20 and 20 degrees
                    if -20 < angle < 20 and abs(angle) > 0.4:
                        angles.append(angle)
                if len(angles) >= 3:
                    median_angle = float(np.median(angles))
                    if abs(median_angle) > 0.5:
                        (h, w) = img.shape[:2]
                        center = (w // 2, h // 2)
                        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        except Exception:
            pass
        return img

    def optimize_resolution(self, img: np.ndarray) -> np.ndarray:
        """
        Scales image to optimal reading resolution.
        Text in KTP needs to be ~20-30px tall for high OCR accuracy.
        """
        h, w = img.shape[:2]

        # Upscale if low resolution (e.g. height < target_height)
        if h < self.target_height:
            scale = self.target_height / float(h)
            new_w, new_h = int(w * scale), self.target_height
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        # Downscale if excessively large (> max_width) to keep sub-second latency
        elif w > self.max_width:
            scale = self.max_width / float(w)
            new_w, new_h = self.max_width, int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return img

    def process(self, image_input: Union[str, Path, bytes, np.ndarray]) -> Tuple[np.ndarray, Path]:
        """
        Runs loading (with HEIC & EXIF rotation), deskewing, resolution optimization,
        and writes temporary PNG file for OCR parsing.
        Returns:
            Tuple of (processed_cv2_image, temp_file_path)
        """
        img = self.load_image(image_input)
        img_deskewed = self.deskew(img)
        img_optimized = self.optimize_resolution(img_deskewed)

        # Save to temp file as standard PNG
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()

        cv2.imwrite(str(temp_path), img_optimized)
        return img_optimized, temp_path
