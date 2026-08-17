import os
import json
import yaml
import torch
import mlflow
import argparse
import logging as log

from typing import Dict, Any, List, Optional

from ultralytics import YOLO, settings

from pathlib import Path
import sys

from mlops.mlflow_utils import setup_mlflow_for_service

CURRENT_FILE = Path(__file__).resolve() #D:\X-ray classification project\src\training\train.py

PROJECT_ROOT = CURRENT_FILE.parents[2] # D:\X-ray classification project

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# 1.... Logging terminal messages
def setup_logging(level=log.INFO) -> None:
    log.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[log.StreamHandler(sys.stdout)],
        force=True,
    )

    for noisy_logger in ("mlflow", "urllib3", "matplotlib"):
        log.getLogger(noisy_logger).setLevel(log.WARNING)



# 2... utilidades Turn a route into an absolute route within your project.        
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def resolve_project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)

    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()

# Read the YAML configuration file and convert it into a Python dictionary.
def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = resolve_project_path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de configuración: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(
            "El archivo YAML no contiene una configuración válida."
        )

    return cfg

# It receives a dictionary that we pass to it and saves it as a .json file.
def save_json(data: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    

# Convierte un diccionario con diccionarios dentro, en un diccionario "plano".
def flatten_dict(
    data: Dict[str, Any],
    parent_key: str = "",
    sep: str = "."
) -> Dict[str, Any]:
    items = []

    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)

        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            if isinstance(value, (str, int, float, bool)) or value is None:
                items.append((new_key, value))
            else:
                items.append((new_key, str(value)))

    return dict(items)    

# registramos los parametros de entrenamiento en mlflow de forma segura 
def safe_log_params(params: Dict[str, Any], max_value_length: int = 250) -> None:
    """
    It records parameters individually to prevent a duplicate or excessively long parameter from breaking the entire log.
    
    """
    for key, value in params.items():
        safe_key = str(key)[:250]
        safe_value = str(value)[:max_value_length]

        try:
            mlflow.log_param(safe_key, safe_value)
        except Exception as exc:
            log.warning(f"No se pudo registrar parámetro en MLflow: {safe_key}={safe_value}. Error: {exc}")


# Count how many images exist within a folder.
def count_images_in_dir(path: Path) -> int:
    if not path.exists():
        return 0

    return sum(
        1
        for file in path.rglob("*")
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )

#Create a summary of how many images are in each split and in each class.
def build_classification_dataset_summary(dataset_path: Path) -> Dict[str, Any]:
    """
    Genera un resumen simple para datasets de clasificación estilo:
    split_data/
      train/
        class_a/
        class_b/
      val/
        class_a/
        class_b/
      test/
        class_a/
        class_b/
    """
    summary = {
        "dataset_path": str(dataset_path),
        "splits": {},
    }

    for split_name in ["train", "val", "valid", "test"]:
        split_dir = dataset_path / split_name

        if not split_dir.exists():
            continue

        class_summary = {}
        total_images = 0

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            count = count_images_in_dir(class_dir)
            class_summary[class_dir.name] = count
            total_images += count

        summary["splits"][split_name] = {
            "split_dir": str(split_dir),
            "total_images": total_images,
            "classes": class_summary,
        }

    return summary

#Verify that the YAML has all the necessary sections for training.
#like project_info,environment,data_source etc
#Then check that the model is a classification model.
#and demands task: classify
def validate_classification_config(cfg: Dict[str, Any]) -> None:
    required_sections = [
        "project_info",
        "environment",
        "data_source",
        "model_config",
        "training_params",
        "mlops_config",
    ]

    for section in required_sections:
        if section not in cfg:
            raise KeyError(f"Falta sección requerida en YAML: {section}")

    task = str(cfg["model_config"].get("task", "")).lower().strip()

    if task != "classify":
        raise ValueError("Este script está diseñado para clasificación. Usa model_config.task: 'classify'.")

def choose_validation_split(dataset_path: Path, preferred_split: str = "test") -> str:

    preferred_dir = dataset_path / preferred_split

    if preferred_dir.exists():
        return preferred_split

    if (dataset_path / "val").exists():
        return "val"

    if (dataset_path / "valid").exists():
        return "val"

    return "val"

# 3... Training

def train_yolo_classification(config_path: str | Path) -> None: #We take the path from the YAML configuration file
    cfg = load_config(config_path) # yaml to dict
    validate_classification_config(cfg) #The configuration is verified to be valid.

    config_abs_path = resolve_project_path(config_path) # Convert a relative path to an absolute path using PROJECT_ROOT, example "config_path"config/training/experiments/01-xray.yaml" to "D:\X-ray classification project\config\training\experiments\01-xray.yaml"

    # we prepare mlflow
    mlops_runtime = setup_mlflow_for_service(
        cfg=cfg,
        current_file=__file__, # We passed you the location of train.py.
        default_service_name="x-ray-classification-project",
        default_workflow_type="training",
    )
# We extract specific values ​​from the dictionary for later use.
    experiment_name = mlops_runtime["experiment_name"]
    run_name = mlops_runtime["run_name"]
    standard_tags = mlops_runtime["standard_tags"]

# We extract specific values ​​from the dictionary(yaml)
    data_source = cfg["data_source"]
    model_config = cfg["model_config"]
    training_params = cfg["training_params"]
    environment = cfg["environment"]
    mlops_config = cfg["mlops_config"]

    dataset_path = resolve_project_path( # "datasets/split_data" to "D:\X-ray classification project\datasets\split_data"
        data_source["dataset_path"]
    )

    base_model_path = resolve_project_path( # example: models/yolo11n-cls.pt to "D:\X-ray classification project\models\yolo11n-cls.pt"
        model_config["base_model_path"]
    )

# Exceptions in case we cannot find dataset and model paths
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {dataset_path}"
        )

    if not base_model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo preentrenado: {base_model_path}"
        )

# also conversion to absolute path
    local_runs_dir = resolve_project_path(
        mlops_config.get(
            "local_runs_dir",
            "runs/YOLO_CLS"
        )
    )

# We created 2 nested folders
    local_runs_dir.mkdir(
        parents=True,
        exist_ok=True
    )

# We create another folder. If it doesn't exist, it creates it. If it already exists, that's fine. We'll store project reports there.
    local_reports_dir = PROJECT_ROOT / "reports"
    local_reports_dir.mkdir(
        parents=True,
        exist_ok=True
    )

# We go through the train val and test folders and we will know how many images there are in each class
    dataset_summary = build_classification_dataset_summary(
        dataset_path
    )

# We concatenate routes, we join reports + mi_entrenamiento_dataset_summary.json and we produce reports/mi_entrenamiento_dataset_summary.json
    dataset_summary_path = (
        local_reports_dir
        / f"{run_name}_dataset_summary.json"
    )

# It takes a dictionary and saves it as a .json file.
    save_json(
        dataset_summary,
        dataset_summary_path
    )

# initial information before training
    log.info("🚀 Iniciando entrenamiento YOLO classification")
    log.info(f"✔ Config path: {config_abs_path}")
    log.info(f"✔ Dataset path: {dataset_path}")
    log.info(f"✔ Base model path: {base_model_path}")
    log.info(f"✔ MLflow experiment: {experiment_name}")
    log.info(f"✔ MLflow run name: {run_name}")
    log.info(f"✔ Local runs dir: {local_runs_dir}")


# configurations to prepare the training of YOLO and MLflow.
    settings.update(
        {
            "mlflow": True,# activate mlflow
            "runs_dir": str(local_runs_dir),# where to save the YOLO results.
        }
    )

# Save these two values ​​as environment variables so the process can know: which MLflow experiment to use. what name to give the run.
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
    os.environ["MLFLOW_RUN_NAME"] = run_name

    # load model
    model = YOLO(str(base_model_path))

    # training
    results = model.train(
        data=str(dataset_path),
        epochs=training_params["epochs"],
        batch=training_params["batch_size"],
        imgsz=data_source["img_size"],
        patience=training_params["patience"],
        task=model_config["task"],
        device=environment["device"],
        workers=environment["workers"],
        optimizer=training_params.get("optimizer", "auto"),
        seed=environment.get("seed", 42),
        lr0=training_params.get("lr0", 0.01),
        lrf=training_params.get("lrf", 0.01),
        cos_lr=training_params.get("cos_lr", False),
        cache=training_params.get("cache", False),
        amp=training_params.get("amp", True),
        plots=training_params.get("plots", True),
        project=str(local_runs_dir),
        name=run_name,
        exist_ok=True,
        save=True,
        verbose=True,
    )    

    #final_validation
    #preferred_split = cfg.get("validation_config", {}).get("split", "test")
    preferred_split = cfg.get("validation_config", {}).get("split", "valid") # .get(clave, valor_por_defecto) busca una clave sin provocar un error si no existe.
    validation_split = choose_validation_split(dataset_path, preferred_split) # "Which folder (valid or test) are we going to use for validation?"

    log.info(f" Running final validation on split ='{validation_split}'")

# We run the model validation
    metrics = model.val(
        data=str(dataset_path),
        split=validation_split,
        imgsz=data_source["img_size"],
        batch=training_params["batch_size"],
        device=environment["device"],
        workers=environment["workers"],
        plots=True,
        project=str(local_runs_dir),
        name=f"{run_name}_final_val_{validation_split}",
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Registrar metadata adicional en MLflow
    # -----------------------------------------------------
    last_run = mlflow.last_active_run()

    if last_run:
        run_id = last_run.info.run_id
        log.info(f"🧾 Registrando metadata adicional en MLflow run_id={run_id}")

        with mlflow.start_run(run_id=run_id):
            mlflow.set_tags(standard_tags)

            mlflow.log_artifact(str(config_abs_path), artifact_path="configs")
            mlflow.log_artifact(str(dataset_summary_path), artifact_path="dataset")

            safe_log_params(flatten_dict(cfg))

            for split_name, split_info in dataset_summary["splits"].items():
                mlflow.log_metric(
                    f"dataset_{split_name}_images_total",
                    split_info["total_images"],
                )

                for class_name, count in split_info["classes"].items():
                    safe_class_name = class_name.replace(" ", "_").replace("-", "_")
                    mlflow.log_metric(
                        f"dataset_{split_name}_{safe_class_name}_images",
                        count,
                    )

            if hasattr(metrics, "results_dict") and isinstance(metrics.results_dict, dict):
                for key, value in metrics.results_dict.items():
                    if isinstance(value, (int, float)):
                        metric_name = (
                            str(key)
                            .replace("(", "")
                            .replace(")", "")
                            .replace(" ", "_")
                            .replace("/", "_")
                        )
                        mlflow.log_metric(f"final_{metric_name}", float(value))

            save_dir = Path(getattr(results, "save_dir", local_runs_dir / run_name))
            weights_dir = save_dir / "weights"

            best_pt = weights_dir / "best.pt"
            last_pt = weights_dir / "last.pt"

            if best_pt.exists():
                mlflow.log_artifact(str(best_pt), artifact_path="weights")

            if last_pt.exists():
                mlflow.log_artifact(str(last_pt), artifact_path="weights")

    else:
        log.warning("⚠️ No se encontró un run activo o reciente de MLflow.")

    log.info("✅ Entrenamiento YOLO classification finalizado correctamente.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. CLI

if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Training pipeline for X-ray classification with YOLO."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/training/experiments/01-xray-classification-yolo11n-cls.yaml",
        help="Path to the experiment YAML config, relative to the project root.",
    )

    args = parser.parse_args()

    try:
        train_yolo_classification(args.config)

    except Exception as e:
        log.error(
            f"❌ Error crítico durante el entrenamiento: {e}",
            exc_info=True
        )
        sys.exit(1)    


