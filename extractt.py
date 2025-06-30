import os
import numpy as np
import pickle
from tqdm import tqdm
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm

# Load pre-trained MobileNetV2 model (faster than ResNet50)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Add pooling layer
model = Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Get all valid image paths
image_folder = "images"
filenames = [os.path.join(image_folder, file)
             for file in os.listdir(image_folder)
             if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

def extract_features_batch(filenames, model, batch_size=32):
    features = []

    for i in tqdm(range(0, len(filenames), batch_size), desc="Extracting features"):
        batch_files = filenames[i:i+batch_size]
        batch_images = []

        for file in batch_files:
            img = image.load_img(file, target_size=(224, 224))
            img_array = image.img_to_array(img)
            batch_images.append(img_array)

        batch_images = np.array(batch_images)
        batch_images = preprocess_input(batch_images)

        # Predict batch
        batch_features = model.predict(batch_images)
        batch_features = batch_features.reshape(batch_features.shape[0], -1)

        # Normalize each feature vector
        batch_features = batch_features / norm(batch_features, axis=1, keepdims=True)
        features.extend(batch_features)

    return features

# Extract features for all images
feature_list = extract_features_batch(filenames, model, batch_size=32)

# Save features and filenames
with open("embeddings.pkl", "wb") as f:
    pickle.dump(feature_list, f)

with open("filenames.pkl", "wb") as f:
    pickle.dump(filenames, f)

print(f"✅ Done! Extracted features for {len(feature_list)} images.")
