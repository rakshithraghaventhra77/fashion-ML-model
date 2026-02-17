from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
import pickle #saving and loading Python objs


# -----------------------------
# STEP 1: Build the CNN model
# -----------------------------
def build_model():

    # Load pretrained MobileNetV2 without classification layer
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # We are NOT training it
    base_model.trainable = False

    # Add pooling layer to convert (7,7,1280) → (1280,)
    model = Sequential([
        base_model,
        GlobalMaxPooling2D()
    ])

    return model


# -----------------------------
# STEP 2: Convert image → vector
# -----------------------------
def extract_embedding(image_path, model):

    # Open image
    img = Image.open(image_path).convert("RGB")

    # Resize to model input size
    img = img.resize((224, 224))

    # Convert to numpy array
    img_array = np.array(img)

    # Add batch dimension → (1,224,224,3)
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess according to MobileNet
    img_array = preprocess_input(img_array)

    # Get features from model
    features = model.predict(img_array, verbose=0)

    # features shape: (1, 1280)

    # Flatten to 1D → (1280,)
    features = features.flatten()

    # Normalize vector (very important)
    features = features / norm(features)

    return features


# -----------------------------
# STEP 3: Loop through all images
# -----------------------------
def generate_embeddings(image_folder, output_dir=None):

    model = build_model()

    embeddings = []
    filenames = []

    image_folder = Path(image_folder).resolve()
    output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent.parent / "index"
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_path in tqdm(image_folder.iterdir()):

        if file_path.suffix.lower() in (".jpg", ".jpeg", ".png"):

            try:
                embedding = extract_embedding(str(file_path), model)
                embeddings.append(embedding)
                filenames.append(str(file_path))

            except Exception as e:
                print("Skipping file:", file_path.name, "Error:", e)

    embeddings = np.array(embeddings)
    print("Final embedding shape:", embeddings.shape)

    # Save embeddings
    np.save(output_dir / "embeddings.npy", embeddings)

    # Save filenames
    with open(output_dir / "filenames.pkl", "wb") as f:
        pickle.dump(filenames, f)

    print("Done. Embeddings saved.")

# -----------------------------
# RUN SCRIPT
# -----------------------------
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    generate_embeddings(base_dir.parent.parent / "data" / "images")

'''

Image file
   ↓
Resize (224×224)
   ↓
Preprocess (scale to [-1,1])
   ↓
MobileNetV2 (ImageNet pretrained, frozen)
   ↓
GlobalMaxPooling
   ↓
Flatten
   ↓
L2 Normalize
   ↓
Save embeddings.npy + filenames.pkl


'''