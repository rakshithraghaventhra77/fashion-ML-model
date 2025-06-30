import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import os
import pickle
import cv2

# If 'feature_list' is already saved via pickle, no need to import from app.py
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3)) #imagenet -> athula model train aagum , antha images laam
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D() #remmove top layer and add this layer
])

#all 440000 images should go into the function and ella features um extract pannanum
img = image.load_img('sample/sample1.jpeg', target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_image = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_image).flatten()
#normalizw pannanum:
normalized_result = result / norm(result) #it gives the new features

neighbours = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean') #take the nearest 5 images that are similar
neighbours.fit(feature_list)

dist, ind = neighbours.kneighbors([normalized_result]) #output vanthu --> filenames --> itha image ah convert panni output tharanum ippo

for file_index in ind[0][1:6]:
    temp_img = cv2.imread(filenames[file_index])
    temp_img = cv2.resize(temp_img, (512, 512))
    cv2.imshow('output', temp_img)
    cv2.waitKey(0)


