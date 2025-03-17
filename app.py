import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
import faiss
from skimage.feature import hog
from PIL import Image
import streamlit as st

# Load VGG19 Model
@st.cache_resource
def load_model():
    base_model = VGG19(weights='imagenet', include_top=False, pooling='avg')
    model = Model(inputs=base_model.input, outputs=base_model.output)
    return model

model = load_model()

# Global Variables
index = None
image_paths = []

# Feature Extraction
def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    deep_features = model.predict(img).flatten()
    
    gray_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    gray_img = cv2.resize(gray_img, (128, 128))
    texture_features = hog(gray_img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)

    return np.concatenate((deep_features, texture_features))

# Load Dataset
def load_dataset(dataset_folder):
    global index, image_paths
    image_paths = []
    feature_vectors = []

    for file_name in os.listdir(dataset_folder):
        if file_name.lower().endswith((".jpg", ".png", ".jpeg", ".webp")):
            path = os.path.join(dataset_folder, file_name)
            features = extract_features(path)
            image_paths.append(path)
            feature_vectors.append(features)

    feature_vectors = np.array(feature_vectors, dtype='float32')
    
    # Build FAISS Index
    dimension = feature_vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(feature_vectors)

# Search Image
def search_image(query_image):
    if index is None:
        st.error("⚠️ Load the dataset first!")
        return None, 0
    
    query_features = extract_features(query_image)
    D, I = index.search(np.array([query_features]), k=1)
    
    matched_path = image_paths[I[0][0]]
    similarity_score = D[0][0] * 100
    
    return matched_path, similarity_score

# Streamlit App
st.set_page_config(page_title="🔎 Similar Image Search", layout="wide")

st.title("🔎 Similar Image Search")
st.write("Upload an image and search for similar images in the dataset.")

# Load Dataset Section
dataset_folder = st.text_input("📁 Enter path to dataset folder:")

if st.button("📂 Load Dataset"):
    if os.path.exists(dataset_folder):
        load_dataset(dataset_folder)
        st.success("✅ Dataset loaded successfully!")
    else:
        st.error("⚠️ Invalid path. Please check the dataset folder path.")

# Search Image Section
uploaded_file = st.file_uploader("🔎 Upload an image to search", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    query_image_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(query_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the uploaded image
    st.image(uploaded_file, caption="Query Image", width=200)
    
    if st.button("🔎 Search"):
        matched_path, similarity_score = search_image(query_image_path)
        
        if matched_path:
            # Display matched image
            matched_image = Image.open(matched_path)
            st.image(matched_image, caption=f"Matched Image ({similarity_score:.2f}% Similar)", width=200)
            
            # Show similarity score
            st.write(f"**Similarity Score:** {similarity_score:.2f}%")
            st.progress(similarity_score / 100)

