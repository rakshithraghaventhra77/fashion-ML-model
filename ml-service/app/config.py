from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT.parent / "data"

INDEX_DIR = PROJECT_ROOT / "index"
DATA_IMAGES_DIR = DATA_DIR / "images"
MYNTRA_IMAGES_DIR = DATA_DIR / "myntradataset" / "images"

STYLES_CSV_PATH = DATA_DIR / "styles.csv"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
FILENAMES_PATH = INDEX_DIR / "filenames.pkl"
FAISS_INDEX_PATH = INDEX_DIR / "faiss_index.index"

PUBLIC_ML_BASE_URL = os.getenv("PUBLIC_ML_BASE_URL", "http://localhost:8000")

IMAGE_SIZE = 224
TOP_K = 5
