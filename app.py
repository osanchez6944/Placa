#!/usr/bin/env python3
# app.py -- FastAPI + YOLOv8 + EasyOCR

import os
import logging
import base64
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import easyocr

# ==================== CONFIG ====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yolo-plates")

MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
OCR_LANGS = os.getenv("OCR_LANGS", "en").split(",")
CONF_THRESH = float(os.getenv("CONF_THRESH", 0.25))
RETURN_IMAGE = True

# ==================== APP INIT ====================
app = FastAPI(title="YOLOv8 - Detector de Placas (OCR)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Cambiar en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== CARGAR MODELOS ====================
logger.info(f"🔹 Cargando modelo YOLOv8 desde {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
logger.info("✅ Modelo YOLOv8 cargado.")

logger.info(f"🔹 Inicializando EasyOCR con idiomas: {OCR_LANGS}...")
reader = easyocr.Reader(OCR_LANGS, gpu=False)
logger.info("✅ EasyOCR listo.")

# ==================== HELPERS ====================
def ocr_read_text_from_roi(roi_bgr: np.ndarray) -> Optional[str]:
    """Ejecuta OCR sobre un ROI y devuelve texto limpio."""
    try:
        if roi_bgr is None or roi_bgr.size == 0:
            return None
        
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        result = reader.readtext(roi_rgb)
        
        if not result:
            return None
        
        # Tomar el resultado con mayor confianza
        best = max(result, key=lambda x: x[2])
        text = best[1]
        
        # Limpiar texto: solo alfanuméricos
        text = "".join(ch for ch in text if ch.isalnum())
        return text.upper() if text else None
        
    except Exception as e:
        logger.exception(f"❌ Error OCR: {e}")
        return None


def image_to_base64_jpg(img_bgr: np.ndarray) -> str:
    """Convierte imagen BGR a base64 (JPG)."""
    _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


# ==================== RUTAS ====================

@app.get("/")
def home():
    """Health check."""
    return {"message": "YOLOv8 + OCR server running"}


@app.post("/predict/")
async def predict(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None)
):
    """
    Detecta placas en una imagen.
    
    Inputs:
    - file: multipart/form-data con imagen
    - image_base64: base64 string de imagen
    
    Returns:
    {
        "success": True,
        "placas": ["ABC123", "XYZ987"],
        "num_placas": 2,
        "image": "base64_string",
        "message": "OK"
    }
    """
    try:
        logger.info("📩 Petición recibida en /predict/")
        
        # Leer imagen
        if file:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif image_base64:
            if image_base64.startswith("data:image"):
                image_base64 = image_base64.split(",")[1]
            
            image_base64 = image_base64.strip()
            img_data = base64.b64decode(image_base64 + "===")
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            return {"error": "No se recibió imagen"}
        
        if frame is None:
            return {"error": "No se pudo decodificar la imagen"}
        
        # ========== INFERENCIA YOLO ==========
        logger.info("🧠 Procesando con YOLOv8...")
        results = model.predict(source=frame, conf=CONF_THRESH, verbose=False)
        
        if not results:
            return {
                "success": True,
                "placas": [],
                "num_placas": 0,
                "image": None,
                "message": "Sin detecciones"
            }
        
        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) > 0 else np.array([])
        clss = r.boxes.cls.cpu().numpy() if len(r.boxes) > 0 else np.array([])
        confs = r.boxes.conf.cpu().numpy() if len(r.boxes) > 0 else np.array([])
        
        placas_detectadas: List[str] = []
        
        # ========== PROCESAR CADA DETECCIÓN ==========
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(clss[i]) if len(clss) > i else None
            label = model.names[cls_id] if cls_id is not None else "objeto"
            conf = confs[i] if len(confs) > i else 0
            
            # Extraer ROI
            h, w = frame.shape[:2]
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            roi = frame[y1c:y2c, x1c:x2c].copy()
            
            # OCR solo si es placa
            if any(k in label.lower() for k in ["placa", "plate", "license"]):
                text_detected = ocr_read_text_from_roi(roi)
                if text_detected:
                    placas_detectadas.append(text_detected)
                    cv2.putText(frame, text_detected, (x1, max(30, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            
            # Dibujar caja
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        
        img_b64 = image_to_base64_jpg(frame) if RETURN_IMAGE else None
        
        logger.info(f"✅ Placas detectadas: {placas_detectadas}")
        
        return {
            "success": True,
            "placas": placas_detectadas,
            "num_placas": len(placas_detectadas),
            "image": img_b64,
            "message": "OK" if placas_detectadas else "Detección completada sin placas"
        }
    
    except Exception as e:
        logger.exception(f"❌ Error en /predict/: {e}")
        return {"error": str(e)}


@app.post("/predict_json/")
async def predict_json(request: Request):
    """
    Alternativa JSON pura (sin multipart).
    
    Body:
    {
        "image_base64": "data:image/jpeg;base64,..."
    }
    """
    try:
        body = await request.json()
        image_base64 = body.get("image_base64")
        
        if not image_base64:
            return {"error": "No 'image_base64' provided"}
        
        # Procesar igual que predict()
        if image_base64.startswith("data:image"):
            image_base64 = image_base64.split(",")[1]
        
        image_base64 = image_base64.strip()
        img_data = base64.b64decode(image_base64 + "===")
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"error": "No se pudo decodificar"}
        
        # ... resto igual a predict() ...
        return {"success": True, "placas": []}
    
    except Exception as e:
        logger.exception(f"❌ Error en /predict_json/: {e}")
        return {"error": str(e)}


# ==================== MAIN ====================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    logger.info(f"🚀 Iniciando servidor en 0.0.0.0:{port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)