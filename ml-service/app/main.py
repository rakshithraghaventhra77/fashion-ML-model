from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
import io
import logging
import os

from app.embedding_generator import extract_embedding
from app.image_store import resolve_image_path
from app.model_loader import get_model
from app.metadata_store import build_enriched_product, load_product_metadata
from app.search_engine import search
from app.config import TOP_K

app = FastAPI(title="Image Similarity Service")

# Configure CORS with explicit origins (for production, use env var)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model at startup
try:
    model = get_model()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}", exc_info=True)
    model = None


@app.get("/health")
def health_check():
    return {
        "status": "Service is running",
        "model_loaded": model is not None,
        "metadata_loaded": len(load_product_metadata()) > 0,
    }


@app.get("/images/{product_id}")
def get_image(product_id: str):
    image_path = resolve_image_path(product_id)

    if image_path is None:
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(str(image_path))


@app.post("/recommend")
async def recommend(file: UploadFile = File(...)):

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        query_vector = extract_embedding(img, model)

        results = search(query_vector, TOP_K)
        enriched_results = [build_enriched_product(file_path) for file_path in results]

        return {
            "success": True,
            "recommendations": enriched_results
        }

    except Exception as e:
        logger.exception(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Image processing failed") from e