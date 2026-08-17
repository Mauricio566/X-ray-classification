# direct inference without server or http requests

import os
import sys
import cv2
import base64
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.processing.runner import XRayInferencePipeline


def base64_to_image(b64_string):
    img_data = base64.b64decode(b64_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    pipeline = XRayInferencePipeline()
    image_path = "examples/images/normal_rx_test.jpeg"

    with open(image_path, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode('utf-8')

    print(" Running inference pipeline...")
    result = pipeline.run(b64_string)

    perf = result['performance']
    pred = result['prediction']

    print("\n INFERENCE RESULTS:")
    print("-" * 30)
    print(f"  Predicted Class:  {pred['label'].upper()} ({pred['confidence'] * 100:.2f}%)")
    print("-" * 30)
    print(f"  Preprocessing: {perf['preprocess_time_ms']} ms")
    print(f"  Model (Inference): {perf['inference_time_ms']} ms")
    print(f"  Explainability:    {perf['explainability_time_ms']} ms")
    print("-" * 30)
    print(f"  TOTAL LATENCY:    {perf['total_latency_ms']} ms")

    # --- Visualización Gráfica ---
    original_img = cv2.imread(image_path)
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    heatmap_rgb = base64_to_image(result['explainability']['heatmap_base64'])
    overlay_rgb = base64_to_image(result['explainability']['overlay_base64'])

    # Plotear
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_rgb)
    plt.title("1. Original X-Ray")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_rgb)
    plt.title("2. Grad-CAM Heatmap")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(overlay_rgb)
    plt.title(f"3. Overlay ({pred['label']})")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

