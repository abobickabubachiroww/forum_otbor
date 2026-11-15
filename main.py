from fastapi import FastAPI, UploadFile, File, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import os

from uniface.detection import RetinaFace
from uniface.spoofing import MiniFASNet

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Инициализируем поиск лица и проверку на спуф
detector = RetinaFace()
spoofer = MiniFASNet()

# Модель  определеня возраста (с https://huggingface.co/TrungTran/faceage_ClientScan/tree/main)
AGE_MODEL_PATH = "faceage_dino_fp32.onnx"  

if not os.path.exists(AGE_MODEL_PATH):
    raise FileNotFoundError(
        f"Модель не найдена: {os.path.abspath(AGE_MODEL_PATH)}"
    )

age_session = ort.InferenceSession(AGE_MODEL_PATH, providers=["CPUExecutionProvider"])
age_input_name = age_session.get_inputs()[0].name

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ф-я обрезки лица, +10%  к краю
def _crop_face(img_bgr: np.ndarray, bbox: list, padding: float = 0.1) -> np.ndarray:
    x1, y1, x2, y2 = map(int, bbox)
    w, h = x2 - x1, y2 - y1
    pad_w, pad_h = int(w * padding), int(h * padding)

    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(img_bgr.shape[1], x2 + pad_w)
    y2 = min(img_bgr.shape[0], y2 + pad_h)

    return img_bgr[y1:y2, x1:x2]

# подготовка фото для модели
def _preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(face_rgb).resize((224, 224), Image.BICUBIC)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1)[np.newaxis]

# ф-я анализа
def _analyze(image_bytes: bytes) -> dict:
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Не удалось прочитать изображение"}

    faces = detector.detect(img)
    if not faces:
        return {"error": "Лицо не обнаружено"}
    
    #если лиц несколько, берем первое
    face = faces[0]
    bbox = face.bbox

    #проверка на спуф
    spoof_result = spoofer.predict(img, bbox)
    if not spoof_result.is_real:
        return {
            "liveness": False,
            "reason": "SPOOF_DETECTED",
            "live_score": float(spoof_result.confidence),
        }

    #вырезаем лицо
    face_crop = _crop_face(img, bbox)
    if face_crop.size == 0:
        return {"error": "Не удалось вырезать лицо"}

    #предсказание возраста
    age_input = _preprocess_face(face_crop)
    age_output = age_session.run(None, {age_input_name: age_input})[0]
    predicted_age = float(age_output.flatten()[0])

    return {
        "liveness": True,
        "live_score": float(spoof_result.confidence),
        "age": round(predicted_age, 1),
    }


# Роуты веб
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="camera.html")


@app.post("/api/verify")
async def api_verify(image: UploadFile = File(...)):
    data = await image.read()
    result = await run_in_threadpool(_analyze, data)
    if "error" in result:
        return JSONResponse(status_code=400, content=result)
    return result