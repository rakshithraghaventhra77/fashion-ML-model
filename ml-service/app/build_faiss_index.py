from pathlib import Path

import numpy as np
import faiss

from embedding_generator import generate_embeddings

base_dir = Path(__file__).resolve().parent
index_dir = base_dir.parent / "index"
embeddings_path = index_dir / "embeddings.npy"

if not embeddings_path.exists():
	data_images_dir = base_dir.parent.parent / "data" / "images"
	generate_embeddings(data_images_dir, output_dir=index_dir)

# load embeddings
embeddings = np.load(embeddings_path).astype("float32")

print("Embeddings shape:", embeddings.shape)

dimension = embeddings.shape[1]
#Create FAISS index using L2 distance
index = faiss.IndexFlatL2(dimension)

#IndexFlatL2 is a simple index that computes L2 distance between vectors. It is suitable for small datasets and provides exact nearest neighbor search. For larger datasets, you might want to consider more complex indices like IndexIVFFlat or IndexHNSW.
index.add(embeddings)
print("Total vectors:" , index.ntotal)
#Save the index to disk
faiss.write_index(index, "../index/faiss_index.index")
print("FAISS index built and saved successfully.")