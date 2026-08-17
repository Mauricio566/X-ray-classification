import base64
import cv2
import numpy as np
from src.processing.interfaces.main import IImageProcessor


class ImageUtils(IImageProcessor):
    def decode_base64(self, base64_string: str) -> np.ndarray:
        try:
            if "," in base64_string:
                base64_string = base64_string.split(",")[1]

            img_data = base64.b64decode(base64_string)
            np_arr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("La imagen no pudo ser decodificada.")

            return image
        except Exception as e:
            raise ValueError(f"Error decodificando Base64: {str(e)}")

    def encode_base64(self, image: np.ndarray) -> str:
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')

    def apply_heatmap_colors(self, heatmap: np.ndarray, shape_ref: tuple) -> np.ndarray:
        heatmap_resized = cv2.resize(heatmap, (shape_ref[1], shape_ref[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        return colored

    def overlay_heatmap(self, original_img: np.ndarray, colored_heatmap: np.ndarray, alpha=0.5) -> np.ndarray:
        return cv2.addWeighted(original_img, 1 - alpha, colored_heatmap, alpha, 0)