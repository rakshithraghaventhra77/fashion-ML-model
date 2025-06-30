import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
import numpy as np
from tqdm import tqdm
import os
import pickle

model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3)) #imagenet -> athula model train aagum , antha images laam
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D() #remmove top layer and add this layer
])

# all 440000 images should go into the function and ella features um extract pannanum
def extract_features(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_image = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_image, verbose=0).flatten()
        normalized_result = result / norm(result)  # normalize pannanum
        return normalized_result
    except Exception as e:
        print(f"❌ Skipping {img_path} due to error: {e}")
        return None

# Pre-resolve file paths
image_dir = 'images'
filenames = [os.path.join(image_dir, file) for file in os.listdir(image_dir)]

feature_list = [] # 2d list --> ithuku ulla 202 features irukum ::: one 2d list for each image
valid_filenames = []

for file in tqdm(filenames, desc="Extracting features"):  # tqdm --> sees the progress of for loop--> kiila progress theriyum like mb/s and all
    features = extract_features(file, model)
    if features is not None:
        feature_list.append(features)
        valid_filenames.append(file)

# Save using pickle
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(valid_filenames, open('filenames.pkl', 'wb'))
