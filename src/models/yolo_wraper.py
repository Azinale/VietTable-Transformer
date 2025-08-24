import numpy as np
from configs import settings

yolo_model_global = None
yolo_classes_global = {}

def load_yolo_model_safe(model_path: str, device: str):
    """Load YOLO model an toàn"""
    global yolo_model_global, yolo_classes_global
    try:
        from ultralytics import YOLO
        yolo_model_global = YOLO(model_path)
        yolo_model_global.to(device)

        # chạy thử với dummy input
        dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
        _ = yolo_model_global(dummy_img, device=(0 if device=="cuda" else "cpu"), verbose=False)

        if hasattr(yolo_model_global, "names") and yolo_model_global.names:
            raw_names = yolo_model_global.names
            if isinstance(raw_names, dict):
                yolo_classes_global = {int(k): v for k, v in raw_names.items() if str(k).isdigit()}
            elif isinstance(raw_names, (list, tuple)):
                yolo_classes_global = {i: name for i, name in enumerate(raw_names)}

        return yolo_model_global, yolo_classes_global
    except Exception as e:
        print(f"[YOLO ERROR] Không load được model {model_path}: {e}")
        return None, {}

def detect_tables(model, image_np: np.ndarray, conf: float = 0.5):
    """Detect bảng từ ảnh bằng YOLO"""
    if model is None:
        return []
    results = model.predict(image_np, conf=conf, verbose=False)
    detections = []
    for r in results:
        if hasattr(r, "boxes"):
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
                detections.append({
                    "bbox": (x1, y1, x2-x1, y2-y1),
                    "cls": int(b.cls[0].item()),
                    "conf": float(b.conf[0].item())
                })
    return detections
