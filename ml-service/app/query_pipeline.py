from pathlib import Path

import numpy as np
import pickle
from embedding_generator import build_model, extract_embedding
from simple_search import find_similar


# Load embeddings and filenames
base_dir = Path(__file__).resolve().parent
index_dir = base_dir.parent / "index"

embeddings = np.load(index_dir / "embeddings.npy")

with open(index_dir / "filenames.pkl", "rb") as f:
    filenames = pickle.load(f)


# Load model once
model = build_model()


def recommend(image_path, top_k=5):

    # Step 1: Extract embedding for new image
    query_vector = extract_embedding(image_path, model)

    # Step 2: Search similar embeddings
    indices = find_similar(query_vector, top_k)

    # Step 3: Return filenames
    results = [filenames[i] for i in indices]

    return results


if __name__ == "__main__":

    test_image = "../../data/images/sample.jpg"  # change this

    results = recommend(test_image)

    print("Recommended images:\n")

    for r in results:
        print(r)
