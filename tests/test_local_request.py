import requests
import base64
import os
import json
from io import BytesIO

# config
#API_URL = "http://localhost:8080/cnn_xray_demo"
API_URL = "http://localhost:8000/cnn_xray_demo"

# image path
IMAGE_PATH = "examples/images/anomaly_rx_test.jpeg"

# save
OUTPUT_DIR = "tests/results"


def decode_and_save_image(base64_string, output_path):
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]

        image_data = base64.b64decode(base64_string)
        with open(output_path, "wb") as f:
            f.write(image_data)
        print(f" Image saved in: {output_path}")
    except Exception as e:
        print(f" Error saving image {output_path}: {e}")


def main():
    # 1. check image
    if not os.path.exists(IMAGE_PATH):
        print(f" Error: No test image found at: {IMAGE_PATH}")
        print(" Please edit the 'IMAGE_PATH' variable in this script.")
        return

    # 2. create folder
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f" Starting test request to: {API_URL}")
    print(f" Image to process: {IMAGE_PATH}")

    # 3. code image
    with open(IMAGE_PATH, "rb") as image_file:
        base64_utf8_str = base64.b64encode(image_file.read()).decode('utf-8')

    # 4. prepare
    payload = {
        "image_base64": base64_utf8_str
    }

    # 5. send request POST
    try:
        print(" Sending request to server (this may take a few seconds the first time)...")
        response = requests.post(API_URL, json=payload)

        # 6. process response
        if response.status_code == 200:
            data = response.json()

            print("\n SUCCESS! Response received from server:")
            print("-" * 40)

            # show predict
            pred = data["prediction"]
            print(f" PREDICTION: {pred['label']} (Confidence: {pred['confidence']:.4f})")

            # times
            perf = data["performance"]
            print(f"  TIMES:")
            print(f"   - Preprocessing: {perf['preprocess_time_ms']} ms")
            print(f"   - Inference (Model): {perf['inference_time_ms']} ms")
            print(f"   - GradCAM (Heatmap): {perf['explainability_time_ms']} ms")
            print(f"   - Total Latency: {perf['total_latency_ms']} ms")

            # save images
            expl = data["explainability"]
            decode_and_save_image(expl["heatmap_base64"], os.path.join(OUTPUT_DIR, "resultado_heatmap.png"))
            decode_and_save_image(expl["overlay_base64"], os.path.join(OUTPUT_DIR, "resultado_overlay.png"))

            print("-" * 40)
            print(f" Check the folder '{OUTPUT_DIR}' to see the generated images.")

        else:
            print(f" Error in request. Code: {response.status_code}")
            print(f"Detail: {response.text}")

    except requests.exceptions.ConnectionError:
        print(f" Connection Error: Could not connect to {API_URL}")
        print("   Is your Docker container running? (docker ps)")
        print("   Are you using the correct port (8080)?")
    except Exception as e:
        print(f" Unexpected error: {e}")


if __name__ == "__main__":
    main()

    