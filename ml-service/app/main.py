from fastapi import FastAPI, UploadFile, File
import numpy as np
import pickle
import faiss
from embedding_generator import build_model, extract_embedding
from PIL import Image
import io

app = FastAPI()

# Load model ONCE
model = build_model()

# Load embeddings and filenames
embeddings = np.load("../index/embeddings.npy").astype("float32")

with open("../index/filenames.pkl", "rb") as f:
    filenames = pickle.load(f)

# Load FAISS index ONCE
index = faiss.read_index("../index/faiss_index.index")


@app.post("/recommend")
async def recommend(file: UploadFile = File(...)):

    contents = await file.read()

    img = Image.open(io.BytesIO(contents)).convert("RGB")

    # Extract embedding
    query_vector = extract_embedding(img, model)

    query_vector = np.array([query_vector]).astype("float32")

    # Search FAISS
    distances, indices = index.search(query_vector, 5)

    results = [filenames[i] for i in indices[0]]

    return {"recommendations": results}
