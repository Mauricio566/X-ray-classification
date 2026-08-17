from pathlib import Path
import time

import mlflow
from ultralytics import YOLO
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ---------------------------------------------------------
# Configuración
# ---------------------------------------------------------

MODELS = {
    "nano": Path("models/YOLO/xrays_evaluation_model_nano_v1.pt"),
    "medium": Path("models/YOLO/xrays_evaluation_model_medium_v1.pt"),
    "xlarge": Path("models/YOLO/xrays_evaluation_model_xlarge_v1.pt"),
}

TEST_PATH = Path(
    r"D:\X-ray classification project\data\train\ingeniia_services_xrays_evaluation_img_v1.0.0_training_20251121\split_data\test"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# ---------------------------------------------------------
# MLflow
# ---------------------------------------------------------

#mlflow.set_tracking_uri("sqlite:///../../.mlflow/mlflow.db")
mlflow.set_tracking_uri(
    "sqlite:///D:/X-ray classification project/.mlflow/mlflow.db"
)
mlflow.set_experiment("xrays_evaluation")


# ---------------------------------------------------------
# Evaluación de un modelo
# ---------------------------------------------------------

def evaluate_model(model_name: str, model_path: Path, test_path: Path) -> dict:

    print(f"\n{'=' * 60}")
    print(f"Evaluando modelo: {model_path.name}")
    print(f"{'=' * 60}")

    model = YOLO(str(model_path))

    y_true = []
    y_pred = []
    inference_times = []


    for class_name in ["anomaly", "normal"]:

        class_dir = test_path / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"No existe la carpeta de clase: {class_dir}"
            )

        for image_path in class_dir.iterdir():

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            """results = model.predict(
                str(image_path),
                verbose=False,
            )"""

            start_time = time.perf_counter()
            
            results = model.predict(
                str(image_path),
                verbose=False,
            )

            end_time = time.perf_counter()
            
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            
            #print(f"Inference time for {image_path.name}: {inference_time:.4f} seconds")

            result = results[0]

            predicted_class = result.probs.top1
            predicted_name = result.names[predicted_class]

            y_true.append(class_name)
            y_pred.append(predicted_name)

    avg_inference_time = sum(inference_times) / len(inference_times) * 1000  # Convert to milliseconds        

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            pos_label="anomaly",
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            pos_label="anomaly",
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            pos_label="anomaly",
            zero_division=0,
        ),
        "images_total": len(y_true),
        "avg_inference_time_ms": avg_inference_time,
    }

    print(f"Images:    {metrics['images_total']}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"Avg Inference Time: {metrics['avg_inference_time_ms']:.4f} ms")

    # -----------------------------------------------------
    # Registrar Run en MLflow
    # -----------------------------------------------------
    print("MLflow tracking URI:", mlflow.get_tracking_uri())
    print("MLflow experiment:", mlflow.get_experiment_by_name("xrays_evaluation"))


    with mlflow.start_run(run_name=f"xray_{model_name}"):

        print("RUN ID:", mlflow.active_run().info.run_id)

        # Parámetros
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_file", model_path.name)
        mlflow.log_param("test_dataset", str(test_path))
        mlflow.log_param("task", "classification")

        # Métricas
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1", metrics["f1"])
        mlflow.log_metric("images_total", metrics["images_total"])
        mlflow.log_metric(
            "avg_inference_time_ms", metrics["avg_inference_time_ms"]
        )


        # Registrar el modelo como artefacto
        mlflow.log_artifact(
            str(model_path),
            artifact_path="model",
        ) 

        print(
            f"✅ MLflow Run registrado: xray_{model_name}"
        )

    return metrics


# ---------------------------------------------------------
# Ejecutar los 3 modelos
# ---------------------------------------------------------

def main():

    all_results = {}

    for model_name, model_path in MODELS.items():

        if not model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo: {model_path}"
            )

        metrics = evaluate_model(
            model_name=model_name,
            model_path=model_path,
            test_path=TEST_PATH,
        )

        all_results[model_name] = metrics

    # -----------------------------------------------------
    # Resumen
    # -----------------------------------------------------

    print("\n")
    print("=" * 80)
    print("COMPARACIÓN DE MODELOS")
    print("=" * 80)

    for model_name, metrics in all_results.items():

        print(
            f"{model_name.upper():8} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1']:.4f}"
        )


if __name__ == "__main__":
    main()

# python -m mlflow ui --backend-store-uri "sqlite:///D:/X-ray classification project/.mlflow/mlflow.db"    