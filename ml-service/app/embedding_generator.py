import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
import pickle


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
def generate_embeddings(image_folder):

    model = build_model()

    embeddings = []
    filenames = []

    for file in tqdm(os.listdir(image_folder)):

        if file.lower().endswith((".jpg", ".jpeg", ".png")):

            full_path = os.path.join(image_folder, file)

            try:
                embedding = extract_embedding(full_path, model)
                embeddings.append(embedding)
                filenames.append(full_path)

            except Exception as e:
                print("Skipping file:", file)

    embeddings = np.array(embeddings)

    print("Final embedding shape:", embeddings.shape)

    # Save embeddings
    np.save("embeddings.npy", embeddings)

    # Save filenames
    with open("filenames.pkl", "wb") as f:
        pickle.dump(filenames, f)

    print("Done. Embeddings saved.")


# -----------------------------
# RUN SCRIPT
# -----------------------------
if __name__ == "__main__":
    generate_embeddings("../../data/images")


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