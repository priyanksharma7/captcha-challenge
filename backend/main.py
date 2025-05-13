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
MAXY = HEIGHT - 60
NUMCHARS = 4
SYMBOL_SIZE = 50
SYMBOL_SET = "0123456789"
FONTS_DIR = "fonts"
RESULTS_FILE = "results.csv"

# === App Setup ===
app = FastAPI(title="CAPTCHA Solver API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://captcha-priyank.onrender.com"],
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
    logger.info("Models loaded and warmed up.")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    raise RuntimeError("Model initialization failed.")

# === CAPTCHA Generation ===
def generate_captcha():
    fonts = [f for f in os.listdir(FONTS_DIR) if f.endswith(".ttf")]
    while True:
        x_pos = 0
        answer = ''
        char_info_list = []
        img = Image.new('RGB', (WIDTH, HEIGHT), color=(255, 255, 255))
        canvas = ImageDraw.Draw(img)
        for _ in range(NUMCHARS):
            font = ImageFont.truetype(f'{FONTS_DIR}/{fonts[randrange(len(fonts))]}', randrange(MINSIZE, MAXSIZE))
            char = SYMBOL_SET[randrange(len(SYMBOL_SET))]
            answer += char
            x_pos += randrange(10, MINX)
            y_pos = randrange(MINY, MAXY)
            canvas.text((x_pos, y_pos), char, font=font, fill=(0, 0, 0), anchor="la")
            char_info_list.append((char, font, x_pos, y_pos))
            coords = canvas.textbbox((x_pos, y_pos), char, font)
            x_pos = coords[2]

        noisy_image = Image.new('RGB', (WIDTH, HEIGHT), color=(255, 255, 255))
        canvas = ImageDraw.Draw(noisy_image)
        
        # Add Noise by adding random lines
        for _ in range(randrange(5, 10)):  # Number of lines
            x1, y1 = randrange(WIDTH), randrange(HEIGHT)
            x2, y2 = randrange(WIDTH), randrange(HEIGHT)
            line_color = tuple(randrange(50, 150) for _ in range(3))  # Grayish lines
            canvas.line((x1, y1, x2, y2), fill=line_color, width=randrange(1,5))

        for char, font, x_pos, y_pos in char_info_list:
            char_img = Image.new('RGBA', (100, 100), (255, 255, 255, 0))
            char_draw = ImageDraw.Draw(char_img)
            char_draw.text((25, 25), char, font=font, fill=(0, 0, 0))
            angle = randrange(-70, 70)
            rotated_char = char_img.rotate(angle, resample=Image.BICUBIC, expand=True)
            
            # Extract bounding box from alpha channel to locate the drawn text
            alpha = rotated_char.getchannel('A')
            bbox = alpha.getbbox()
            char_crop = rotated_char.crop(bbox)
    
            # Paste on the base canvas
            composite_layer = Image.new('RGBA', (WIDTH, HEIGHT), (255, 255, 255, 0))
            composite_layer.paste(char_crop, (x_pos, y_pos))
            noisy_image = Image.alpha_composite(noisy_image.convert('RGBA'), composite_layer).convert('RGB')
        
        yield img, noisy_image, answer

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
    clean_images, noisy_images, truths = [], [], []
    for _ in range(num_images):
        clean_img, noisy_img, truth = next(generate_captcha())

        noisy_buffer = BytesIO()
        noisy_img.save(noisy_buffer, format="PNG")
        noisy_str = base64.b64encode(noisy_buffer.getvalue()).decode()
        noisy_images.append(f"data:image/png;base64,{noisy_str}")

        clean_buffer = BytesIO()
        clean_img.save(clean_buffer, format="PNG")
        clean_str = base64.b64encode(clean_buffer.getvalue()).decode()
        clean_images.append(f"data:image/png;base64,{clean_str}")

        truths.append(truth)
    return {"noisy_images": noisy_images, "clean_images": clean_images, "truths": truths}

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