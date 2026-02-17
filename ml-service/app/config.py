from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

INDEX_DIR = PROJECT_ROOT / "index"

EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
FILENAMES_PATH = INDEX_DIR / "filenames.pkl"
FAISS_INDEX_PATH = INDEX_DIR / "faiss_index.index"

IMAGE_SIZE = 224
TOP_K = 5
