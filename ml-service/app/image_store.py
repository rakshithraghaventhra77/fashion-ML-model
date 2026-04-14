from pathlib import Path

from app.config import DATA_IMAGES_DIR, MYNTRA_IMAGES_DIR


def resolve_image_path(product_id: str) -> Path | None:
    candidate_directories = [DATA_IMAGES_DIR, MYNTRA_IMAGES_DIR]
    candidate_extensions = [".jpg", ".jpeg", ".png"]

    for directory in candidate_directories:
        for extension in candidate_extensions:
            candidate = directory / f"{product_id}{extension}"
            if candidate.exists():
                return candidate

    return None