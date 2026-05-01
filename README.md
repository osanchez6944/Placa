# Backend YOLOv8 - Detector de Placas

Esta carpeta contiene la API FastAPI que usa YOLOv8 y EasyOCR para detectar placas.

## Instalar dependencias

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecutar servidor

```bash
python app.py
```

## Usar la API

- `GET /` : health check
- `POST /predict/` : enviar imagen por `multipart/form-data` o `image_base64`

Ejemplo con curl:

```bash
curl -X POST -H "Content-Type: multipart/form-data" -F "file=@ruta/placa.jpg" http://localhost:8080/predict/
```

## Variables de entorno opcionales

- `MODEL_PATH` : ruta del modelo `best.pt` (por defecto `best.pt`)
- `CONF_THRESH` : umbral de confianza (por defecto `0.25`)
- `OCR_LANGS` : idiomas de EasyOCR (por defecto `en`)
