# 👗 Fashion Recommender System

A Streamlit app that provides fashion product recommendations by analyzing a user's uploaded image and displaying visually similar items using ResNet50 embeddings and a nearest neighbors approach.

---

## 📁 Dataset

This project uses the **Fashion Product Images Dataset** from Kaggle, created by **paramaggarwal**:

📥 **Dataset URL**:  
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset

➡️ The dataset contains several thousand clothing images, which are preprocessed and embedded into `embeddings.pkl` for fast similarity search.

---

## 🚀 Features

- Upload any fashion image (e.g., shirt, dress, shoes), and get 5 visually similar product recommendations.
- Powered by pre-trained **ResNet50** (`imagenet` weights) for feature extraction.
- Uses **K-Nearest Neighbors** to find closest matching items in the embedding space.
- Clean, dark-themed Streamlit interface with yellow accents and fade-in animations.

---

## 🛠️ Setup & Usage

### 1. Clone the repository

```bash
git clone https://github.com/rakshithraghaventhra77/fashion-ML-model.git
cd fashion-ML-model
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

*(If you don't have a `requirements.txt`, install the key libraries manually:)*

```bash
pip install streamlit tensorflow pillow numpy scikit-learn
```

### 3. Prepare the data

- Download the dataset from Kaggle.
- Precompute embeddings using the `.ipynb` notebook (or your Python script).
- Generate:
  - `embeddings.pkl` — NumPy array of image vectors
  - `filenames.pkl` — List of image paths matching embeddings

### 4. Run the app

```bash
streamlit run app.py
```

- Upload an image using the interface.
- View the uploaded image along with 5 similar fashion item suggestions.

---

## 🔧 File Overview

```
.
├── app.py                # Streamlit application code
├── embeddings.pkl        # Precomputed ResNet50 feature vectors
├── filenames.pkl         # Corresponding image file paths
├── uploads/              # Directory to store user-uploaded images
├── requirements.txt      # Python dependencies (optional)
└── README.md             # This file
```

---

## 👩‍💻 Customize & Expand

- Swap the pre-trained model (e.g., ResNet101, MobileNet) for feature extraction
- Adjust number of displayed recommendations
- Add user interface tweaks (e.g., theme toggle, image zoom)
- Deploy on [Streamlit Cloud](https://streamlit.io/cloud) for instant online access

---

## 🎯 Deployment (Optional)

1. Push the repo to GitHub.
2. Log in to [Streamlit Cloud](https://streamlit.io/cloud) and connect your GitHub.
3. Select this repository to deploy it live.
4. Specify `app.py` as the main file.
5. Add `embeddings.pkl` and `filenames.pkl` as part of the deployment assets.

---

## 📄 License

Dataset `Fashion Product Images Dataset` by paramaggarwal (Kaggle).  
This repository/code is licensed under MIT License. Feel free to use and modify.

---

## ❓ Questions?

Feel free to:
- Open an issue on GitHub
- Reach out via GitHub discussions or email

---

Thank you for checking out this project—happy coding and styling!

---

*Generated with 🧠 and ❤️ by your AI assistant.*
