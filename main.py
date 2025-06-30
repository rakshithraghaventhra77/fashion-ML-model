import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

st.title("Fashion Recommender System")

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#feature extraction
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

#recommend
def recommend(features, feature_list):
    """
    neighbours = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')  # take the nearest 5 images that are similar
    neighbours.fit(feature_list)
    """
    n_neighbors = min(6, len(feature_list))  # Adjust dynamically based on available samples
    neighbours = NearestNeighbors(n_neighbors=n_neighbors)
    neighbours.fit(feature_list)

    dist, ind = neighbours.kneighbors([features])
    return ind

#save uploaded
def save_uplaoded(uploaded_file):
    try:
        os.makedirs("uploads", exist_ok=True)
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

#stepss --> file uplaod --> load the file --> extract features --> recommenstaiions
#show the recommendations --> avlothaan

uploaded_file = st.file_uploader("Choose an Image :")
if uploaded_file is not None:
    if save_uplaoded(uploaded_file):
        #file uplaoded --> display
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        #extract features --> recommenstaiions
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        st.text(features)

        ind = recommend(features, feature_list)
        #show
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(filenames[ind[0][1]])
        with col2:
            st.image(filenames[ind[0][2]])
        with col3:
            st.image(filenames[ind[0][3]])
        with col4:
            st.image(filenames[ind[0][4]])
        with col5:
            st.image(filenames[ind[0][5]])

    else:
        st.header("Error Occured")
