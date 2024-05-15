import io
import pickle

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from mtcnn import MTCNN
from PIL import Image, ImageOps

detector = MTCNN()
app = FastAPI()
origins = [
    "http://localhost:5173",
]


model = None
with open("./model.pickle", "rb") as f:
    model = pickle.load(f)


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/test")
async def test():
    return {"message": "server is online"}


@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    image_bytes = await image.read()
    with io.BytesIO(image_bytes) as img_data:
        image = Image.open(img_data).convert("RGB")
    img = np.array(image)
    result = detector.detect_faces(img)
    if len(result[0]["box"]) == 4:
        x, y, width, height = result[0]["box"]
        headshot = img[y : height + y, x : width + x]
        headshot = Image.fromarray(headshot)
        # with Image.open(headshot) as im:
        bwImage = ImageOps.fit(headshot, (91, 125)).convert("L")
        image_data = np.array(bwImage).flatten()
        return {"bmi": round(model.predict([image_data])[0], 2), "error": ""}
    else:
        return {"bmi": "0", "error": "face_not_detected"}
