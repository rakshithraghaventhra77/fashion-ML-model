import numpy as np
import pickle
from .embedding_generator import build_model, extract_embedding
from .simple_search import find_similar


# Load embeddings and filenames
embeddings = np.load("../index/embeddings.npy")

with open("../index/filenames.pkl", "rb") as f:
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
