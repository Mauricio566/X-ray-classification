from pathlib import Path

from ultralytics import YOLO
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# Configuration of the models and test dataset path

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
# Evaluación de un modelo
# ---------------------------------------------------------

def evaluate_model(model_path: Path, test_path: Path) -> dict:
    print(f"\n{'=' * 60}")
    print(f"Evaluando modelo: {model_path.name}")
    print(f"{'=' * 60}")

    model = YOLO(str(model_path))

    y_true = []
    y_pred = []

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

            results = model.predict(
                str(image_path),
                verbose=False,
            )

            result = results[0]

            predicted_class = result.probs.top1
            predicted_name = result.names[predicted_class]

            y_true.append(class_name)
            y_pred.append(predicted_name)

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
    }

    print(f"Images:    {metrics['images_total']}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")

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
            model_path=model_path,
            test_path=TEST_PATH,
        )

        all_results[model_name] = metrics

    # -----------------------------------------------------
    # Resumen final
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