from pathlib import Path

import numpy as np
import pickle


# STEP 1: Load saved embeddings
base_dir = Path(__file__).resolve().parent
index_dir = base_dir.parent / "index"
embeddings = np.load(index_dir / "embeddings.npy")

# STEP 2: Load image file names
with open(index_dir / "filenames.pkl", "rb") as f:
embeddings = np.load("../index/embeddings.npy")

# STEP 2: Load image file names
with open("../index/filenames.pkl", "rb") as f:
    filenames = pickle.load(f)


# STEP 3: Function to find similar images
def find_similar(query_vector, top_k=5):

    distances = []

    # Compare query vector with every stored embedding
    for i in range(len(embeddings)):

        # Euclidean distance
        distance = np.linalg.norm(embeddings[i] - query_vector) #this is mathematically-->distance = sqrt(sum((a - b)^2))
        distance = np.linalg.norm(embeddings[i] - query_vector)

        distances.append(distance)

    # Convert to numpy array
    distances = np.array(distances)

    # Sort distances and get smallest ones
    sorted_indices = np.argsort(distances)

    # Return top_k closest matches
    return sorted_indices[:top_k]


# STEP 4: Run test
if __name__ == "__main__":

    # Use first image embedding as example query
    query_vector = embeddings[0]

    results = find_similar(query_vector)

    print("Most similar images:\n")

    for index in results:
        print(filenames[index])
