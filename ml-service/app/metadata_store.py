from pathlib import Path
import csv
from functools import lru_cache
import re

from app.config import PUBLIC_ML_BASE_URL, STYLES_CSV_PATH


@lru_cache(maxsize=1)
def load_product_metadata(csv_path: Path = STYLES_CSV_PATH) -> dict[str, dict[str, str]]:
    metadata_by_id: dict[str, dict[str, str]] = {}

    if not csv_path.exists():
        return metadata_by_id

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            product_id = str(row.get("id", "")).strip()
            if not product_id:
                continue

            metadata_by_id[product_id] = {
                "id": product_id,
                "gender": row.get("gender", ""),
                "masterCategory": row.get("masterCategory", ""),
                "subCategory": row.get("subCategory", ""),
                "articleType": row.get("articleType", ""),
                "baseColour": row.get("baseColour", ""),
                "season": row.get("season", ""),
                "year": row.get("year", ""),
                "usage": row.get("usage", ""),
                "productDisplayName": row.get("productDisplayName", ""),
            }

    return metadata_by_id


def build_enriched_product(file_path: str) -> dict[str, str]:
    # Handles both Linux and Windows-style paths in indexed filenames.
    normalized = file_path.replace("\\", "/")
    candidate_name = normalized.rsplit("/", 1)[-1]
    product_id = Path(candidate_name).stem

    if not product_id.isdigit():
        match = re.search(r"(\d+)(?:\.[a-zA-Z0-9]+)?$", candidate_name)
        if match:
            product_id = match.group(1)

    metadata_by_id = load_product_metadata()
    metadata = metadata_by_id.get(product_id, {})

    return {
        "id": product_id,
        "image": f"{product_id}.jpg",
        "sourcePath": file_path,
        "imageUrl": f"{PUBLIC_ML_BASE_URL}/images/{product_id}",
        "gender": metadata.get("gender", ""),
        "masterCategory": metadata.get("masterCategory", ""),
        "subCategory": metadata.get("subCategory", ""),
        "articleType": metadata.get("articleType", ""),
        "baseColour": metadata.get("baseColour", ""),
        "season": metadata.get("season", ""),
        "year": metadata.get("year", ""),
        "usage": metadata.get("usage", ""),
        "productDisplayName": metadata.get("productDisplayName", ""),
    }