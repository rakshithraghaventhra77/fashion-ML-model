from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import logging

from embedding_generator import extract_embedding
from model_loader import get_model
from search_engine import search
from config import TOP_K

app = FastAPI(title="Image Similarity Service")

# Allow frontend / Java calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load model at startup
model = get_model()


@app.get("/health")
def health_check():
    return {"status": "Service is running"}


@app.post("/recommend")
async def recommend(file: UploadFile = File(...)):

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        query_vector = extract_embedding(img, model)

        results = search(query_vector, TOP_K)

        return {
            "success": True,
            "recommendations": results
        }

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Image processing failed")