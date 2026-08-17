import os
import sys
import logging as log

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


#from server.schemas import XRayOutput, XRayInput
from src.server.schemas import XRayOutput, XRayInput
from src.processing.runner import XRayInferencePipeline

log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# init FastAPI
app = FastAPI(
    title="X-ray Evaluation API (CNN Demo)",
    description="Educational microservice for classifying x-rays using YOLOv11 + GradCAM.",
    version="1.0.0"
)

# CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# instance
cnn_xrays_demo = XRayInferencePipeline()


# --- Endpoints ---
# This leads us to the documentation
@app.get("/", include_in_schema=False)
async def root():
    """docs API."""
    return RedirectResponse(url="/docs")


#We checked the service status
@app.get("/health")
async def health_check():
    """Healthcheck to GCP."""
    return {"status": "ok", "model": "YOLO11m-cls"}


@app.post("/cnn_xray_demo",
          response_model=XRayOutput,
          tags=["Predicciones CNN"],
          summary="Classify an X-ray and generate a heat map")

async def predict_xray(request: XRayInput) -> XRayOutput:
    """
    :param
        image in Base64
    :return
        prediction (anomaly/normal)
        heatmap
        performance
    """
    try:
        log.info(f"CNN request received. Base64 size: {len(request.image_base64)}")
        result = cnn_xrays_demo.run(request.image_base64)
        log.info(f" Inference successful. Result: {result['prediction']['label']}")
        return XRayOutput(**result)

    except ValueError as ve:
        log.error(f"Validation or decoding error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        log.error(f"Critical error during inference: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An internal error occurred while processing the X-ray: {str(e)}"
        )
    
# uvicorn src.server.app:app --reload    