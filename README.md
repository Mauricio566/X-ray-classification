# X-Ray Evaluation API (CNN + GradCAM)
This service provides an API to classify images into 2 categories (Normal vs Anomaly). 
We use a CNN based on YOLOv11.

# Description
The service implements a complete computer vision pipeline. It receives an image in Base64 format, 
preprocesses it, and executes two simultaneous tasks:

1-Classification: It uses YOLOv11-cls to determine if the X-ray contains anomalies.
2-Explainability: It uses GradCAM algorithms to generate a heat map highlighting the regions where the model "looked" to make its decision.

# how to use the API
the main endpoint is /cnn_xray_demo.
You must send the image encoded in base64

expected successful response
The service returns the prediction, the explainability (base64 overlay image) 
and the execution times.

# Inference
We have two ways to test inference: directly through the API or directly.

Method 1: The API
we run the script test_local_request.py which is located inside the folder called "tests"
(First, you need to have the server running.)

Method 2: directly
we run the script xrays_evaluation_pipeline.py which is located inside the folder called "examples"

