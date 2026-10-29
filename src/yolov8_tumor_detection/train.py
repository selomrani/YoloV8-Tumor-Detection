import os
import mlflow
from mlflow.tracking import MlflowClient
from ultralytics import YOLO, settings

settings.update({"mlflow": True})


os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "Yolo_tumor_detection"


# we start by a smaller Yolov8 model and then we will move to a bigger one

yolov8_variants_list = ["yolov8n", "yolov8s", "yolov8m"]

params = {
    "epochs": 100,
    "batch": 16,
    "imgsz": 640,
    "lr0": 0.001,
    "optimizer": "SGD",
    "patience": 20,
    "amp": True,
    "save_period": 10,
}

mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])

# now we will loop through the list of yolov8 variants and train each model
for yolov8_variant in yolov8_variants_list:
    with mlflow.start_run(run_name=yolov8_variant):
        mlflow.log_param("yolov8_variant", yolov8_variant)
        mlflow.log_params(params)

        model = YOLO(f"{yolov8_variant}.pt")
        model.train(data="data.yaml", project="runs", name=yolov8_variant, **params)


# register the best run and set the champion alias
client = MlflowClient()
model_name = "yolo_tumor_detector"

exp = client.get_experiment_by_name("Yolo_tumor_detection")
best_run = client.search_runs(
    [exp.experiment_id],
    order_by=["metrics.`metrics/mAP50-95B` DESC"],
    max_results=1,
)[0]

# the Ultralytics callback only logs the raw weights directory, so log a proper
# MLflow model into that run before registering it
best_weights = client.download_artifacts(best_run.info.run_id, "weights/best.pt")
with mlflow.start_run(run_id=best_run.info.run_id):
    model_info = mlflow.pytorch.log_model(
        YOLO(best_weights).model, name="model", serialization_format="pickle"
    )

version = mlflow.register_model(model_info.model_uri, model_name)
client.set_registered_model_alias(model_name, "champion", version.version)
