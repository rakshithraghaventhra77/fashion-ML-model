from config import FAISS_INDEX_PATH, FILENAMES_PATH
import faiss
import pickle
import numpy as np

index = faiss.read_index(str(FAISS_INDEX_PATH))

with open(FILENAMES_PATH, "rb") as f:
    filenames = pickle.load(f)

def search(query_vector, top_k=5):
    query_vector = np.array([query_vector]).astype("float32")
    distances, indices = index.search(query_vector, top_k)
    return [filenames[i] for i in indices[0]]
