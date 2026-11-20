# Setup / restore (local PC)

Everything needed to resume this project on a fresh machine. Code + datasets live in
git; the binary state (MLflow tracking store, weights, logs) lives in Azure Blob.

## 1. Clone + environment

```bash
git clone https://github.com/selomrani/YoloV8-Tumor-Detection.git
cd YoloV8-Tumor-Detection
git checkout dev

# Python 3.11 (see .python-version)
uv sync                       # torch + ultralytics + mlflow, from uv.lock
```

The datasets are in the repo already — `data/processed` has the train/val/test
splits and `Data/Raw` the originals, so no download step is needed.

## 2. Restore MLflow state + weights

Container: `vast/kcrowly-vast`, prefix `yolov8-tumor/2026-10-30/`.

```bash
export RCLONE_CONFIG_AZ_TYPE=azureblob
export RCLONE_CONFIG_AZ_ACCOUNT=vast
export RCLONE_CONFIG_AZ_KEY="<storage-account-key>"   # never commit this

DEST=az:kcrowly-vast/yolov8-tumor/2026-10-30
rclone copy "$DEST/mlruns"    ./mlruns    --transfers 8 --checkers 16 --progress
rclone copy "$DEST/runs"      ./runs      --transfers 8 --checkers 16 --progress
rclone copy "$DEST/logs"      ./logs      --progress
rclone copy "$DEST/evaluation" ./evaluation --progress
```

`mlruns/` is path-portable: run artifact URIs are stored relative
(`mlflow-artifacts:/1/<run_id>/artifacts`), so no rewriting is needed.

## 3. Start the tracking server

```bash
./deploy/mlflow.sh                 # -> http://127.0.0.1:5001
# or manually:
mlflow server --host 127.0.0.1 --port 5001 \
  --backend-store-uri sqlite:///mlruns/mlflow.db \
  --artifacts-destination mlruns/artifacts
```

Then the UI shows experiment `Yolo_tumor_detection`, the three runs
(`yolov8n/s/m`) and registered model `yolo_tumor_detector` with alias `champion`.

## 4. Train / serve

```bash
.venv/bin/python src/yolov8_tumor_detection/train.py
```

`train.py` reads `MLFLOW_TRACKING_URI` from the environment and falls back to
`http://localhost:5001`, so it works locally and on the Vast instance unchanged.

To serve the champion with FastAPI, load it from the alias:

```python
from mlflow import MlflowClient
from ultralytics import YOLO
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5001")
mv = MlflowClient().get_model_version_by_alias("yolo_tumor_detector", "champion")
pt = MlflowClient().download_artifacts(mv.run_id, "weights/best.pt", dst_path="/tmp")
model = YOLO(pt)     # do NOT use mlflow.pytorch.load_model() directly: it returns a
                     # bare DetectionModel (raw forward pass, no letterbox/NMS)
```

## Notes / known gotchas

- **Champion selection**: the training script ranks runs by the *last epoch's*
  `metrics/mAP50-95B`, not the peak. `yolov8n` peaked higher (0.7599 @ epoch 64) but
  finished at 0.7332, so `champion` currently points at `yolov8m` (0.7522).
- **Early stopping** (`patience=20`) fired on all three runs: n=84, s=41, m=35 epochs.
- **Checkpoint paths**: `project="runs"` is applied on top of Ultralytics'
  `runs_dir`, so output lands in `runs/detect/runs/<name>/`.
- **`save_period=10`** writes 100 MB checkpoints each run; consider `-1` for blob/CI use.
- `runs/` weights are duplicates of the copies inside `mlruns/artifacts/.../weights/`.
