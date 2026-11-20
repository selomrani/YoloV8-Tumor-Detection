from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO

MODEL_PATH = Path("runs/detect/train-8/weights/best.pt")

app = FastAPI(title="Brain Tumor Detection API")
model = YOLO(str(MODEL_PATH))

@app.get("/test")
def test():
    return {"message": "Hello, World!"}

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH.name}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename or "upload").suffix or ".jpg"

    with NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        temp.write(await file.read())
        image_path = temp.name

    results = model.predict(
        source=image_path,
        imgsz=640,
        conf=0.25,
        iou=0.45,
        verbose=False,
    )

    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls.item())
        detections.append({
            "class": model.names[class_id],
            "confidence": round(float(box.conf.item()), 4),
            "bbox": [round(value, 2) for value in box.xyxy[0].tolist()],
        })

    return {"detections": detections}