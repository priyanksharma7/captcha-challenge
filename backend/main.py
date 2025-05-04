from fastapi import FastAPI, Request, Header, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from io import BytesIO
from datetime import datetime, timezone, timedelta
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import base64
import logging
from random import randrange

# === Config ===
WIDTH, HEIGHT = 200, 100
MINSIZE, MAXSIZE = 24, 48
MINX, MINY = 20, 20
MAXX, MAXY = WIDTH - 60, HEIGHT - 60
NUMCHARS = 4
SYMBOL_SIZE = 50
SYMBOL_SET = "0123456789"
RESULTS_FILE = "results.csv"

# === App Setup ===
app = FastAPI(title="CAPTCHA Solver API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("captcha-api")

# === Model Loading ===
try:
    sym_model = tf.keras.models.load_model("models/Symbol_Recognizer-100epochs.keras")
    box_model = tf.keras.models.load_model("models/BBox_Regressor-100epochs.keras")
    # Warm-up
    sym_model.predict(np.zeros((NUMCHARS, SYMBOL_SIZE, SYMBOL_SIZE, 3)))
    box_model.predict(np.zeros((1, WIDTH, HEIGHT, 3)))
    logger.info("✅ Models loaded and warmed up.")
except Exception as e:
    logger.error(f"❌ Failed to load models: {e}")
    raise RuntimeError("Model initialization failed.")

# === CAPTCHA Generation ===
def generate_captcha():
    fonts = [f for f in os.listdir('fonts') if f.endswith(".ttf")]
    while True:
        x_pos = 0
        captcha = ''
        img = Image.new('RGB', (WIDTH, HEIGHT), color=(255, 255, 255))
        canvas = ImageDraw.Draw(img)
        for _ in range(NUMCHARS):
            font = ImageFont.truetype(f'fonts/{fonts[randrange(len(fonts))]}', randrange(MINSIZE, MAXSIZE))
            char = SYMBOL_SET[randrange(len(SYMBOL_SET))]
            captcha += char
            x_pos += randrange(10, MINX)
            y_pos = randrange(MINY, MAXY)
            coords = canvas.textbbox((x_pos, y_pos), char, font)
            canvas.text((x_pos, y_pos), char, font=font, fill=(0, 0, 0), anchor="la")
            x_pos = coords[2]
        yield img, captcha

def get_sub_image(image, box):
    return image.crop(box).resize((SYMBOL_SIZE, SYMBOL_SIZE))

def solve_captcha(image: Image.Image) -> str:
    x = np.asarray(image).astype(np.float32) / 255.0
    boxes = box_model.predict(x.reshape(1, WIDTH, HEIGHT, 3), verbose=False)[0]
    sub_images = []
    for i in range(NUMCHARS):
        box = boxes[4*i:4*(i+1)]
        sub_img = get_sub_image(image, box)
        tensor = np.asarray(sub_img).astype(np.float32) / 255.0
        sub_images.append(tensor.reshape(SYMBOL_SIZE, SYMBOL_SIZE, 3))
    batch = np.stack(sub_images)
    preds = sym_model.predict(batch, verbose=False)
    return ''.join(str(np.argmax(p)) for p in preds)

# === API Models ===
class SolveRequest(BaseModel):
    image: str

class StoreRequest(BaseModel):
    username: str
    userscore: int
    usertime: float
    ai_score: int
    ai_time: float

# === API Endpoints ===

@app.get("/generate")
def generate(num_images: int = Query(8, ge=1, le=100)):
    images, truths = [], []
    generator = generate_captcha()
    for _ in range(num_images):
        img, truth = next(generator)
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        images.append(f"data:image/png;base64,{img_str}")
        truths.append(truth)
    return {"images": images, "truths": truths}

@app.post("/solve")
def solve(req: SolveRequest):
    try:
        img_data = base64.b64decode(req.image.split(",")[1])
        image = Image.open(BytesIO(img_data)).convert("RGB")
        answer = solve_captcha(image)
        return {"answer": answer}
    except Exception as e:
        logger.warning(f"Error solving CAPTCHA: {e}")
        return {"answer": "Error"}

@app.post("/store")
def store(data: StoreRequest):
    record = {
        "username": data.username,
        "userscore": data.userscore,
        "usertime": data.usertime,
        "ai_score": data.ai_score,
        "ai_time": data.ai_time,
        "timestamp": (datetime.now(timezone.utc) + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S'),
    }
    df = pd.DataFrame([record])
    if os.path.exists(RESULTS_FILE):
        df.to_csv(RESULTS_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(RESULTS_FILE, index=False)
    return {"status": "success"}

@app.get("/leaderboard")
def leaderboard():
    if not os.path.exists(RESULTS_FILE):
        return []
    df = pd.read_csv(RESULTS_FILE)
    df.sort_values(["userscore", "usertime"], ascending=[False, True], inplace=True)
    best = df.groupby("username", as_index=False).first()
    top = best.sort_values(["userscore", "usertime"], ascending=[False, True]).head(10)
    return top.to_dict(orient="records")

@app.get("/download")
def download_results(x_admin_key: str = Header(...)):
    if x_admin_key != 'priyank123!':
        raise HTTPException(status_code=403, detail="Access denied")
    if not os.path.exists(RESULTS_FILE):
        raise HTTPException(status_code=404, detail="results.csv not found")
    return FileResponse(path=RESULTS_FILE, media_type='text/csv', filename="results.csv")