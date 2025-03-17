'''

import streamlit as st

# ✅ Set page config FIRST before any Streamlit command
st.set_page_config(page_title="🔎 Similar Image Search", layout="wide")

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

# Load VGG19 Model
@st.cache_resource
def load_model():
    base_model = VGG19(weights='imagenet', include_top=False, pooling='avg')
    model = Model(inputs=base_model.input, outputs=base_model.output)
    return model

model = load_model()

# Initialize session state
if "index" not in st.session_state:
    st.session_state.index = None
if "image_paths" not in st.session_state:
    st.session_state.image_paths = []

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
def load_dataset(uploaded_files):
    image_paths = []
    feature_vectors = []

    dataset_folder = "temp_dataset"
    os.makedirs(dataset_folder, exist_ok=True)

    for uploaded_file in uploaded_files:
        file_path = os.path.join(dataset_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extract features and store them
        features = extract_features(file_path)
        image_paths.append(file_path)
        feature_vectors.append(features)

    if feature_vectors:
        feature_vectors = np.array(feature_vectors, dtype="float32")
        dimension = feature_vectors.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(feature_vectors)

        # ✅ Store index and paths in session state
        st.session_state.index = index
        st.session_state.image_paths = image_paths
        st.success("✅ Dataset loaded successfully!")

# Search Image
def search_image(query_image):
    if st.session_state.index is None or not st.session_state.image_paths:
        st.error("⚠️ Load the dataset first!")
        return None, 0
    
    query_features = extract_features(query_image)
    D, I = st.session_state.index.search(np.array([query_features]), k=1)
    
    matched_path = st.session_state.image_paths[I[0][0]]
    similarity_score = D[0][0] * 100
    
    return matched_path, similarity_score

# Streamlit App
st.title("🔎 Similar Image Search")
st.write("Upload an image and search for similar images in the dataset.")

# Load Dataset Section
uploaded_files = st.file_uploader(
    "📂 Drag and drop your dataset files here (JPG, JPEG, PNG, WEBP)", 
    accept_multiple_files=True, 
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_files:
    if st.button("📂 Load Dataset"):
        load_dataset(uploaded_files)

# Search Image Section
uploaded_file = st.file_uploader(
    "🔎 Upload an image to search", 
    type=["jpg", "jpeg", "png", "webp"]
)

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
'''
'''
import streamlit as st

# ✅ Set page config FIRST before any Streamlit command
st.set_page_config(page_title="🔎 Similar Image Search", layout="wide")

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

# Load VGG19 Model
@st.cache_resource
def load_model():
    base_model = VGG19(weights="imagenet", include_top=False, pooling="avg")
    model = Model(inputs=base_model.input, outputs=base_model.output)
    return model

model = load_model()

# Initialize session state
if "index" not in st.session_state:
    st.session_state.index = None
if "image_paths" not in st.session_state:
    st.session_state.image_paths = []

# Feature Extraction
def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    deep_features = model.predict(img).flatten()

    gray_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    gray_img = cv2.resize(gray_img, (128, 128))
    texture_features = hog(
        gray_img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True
    )

    return np.concatenate((deep_features, texture_features))

# Load Dataset from Drag & Drop
def load_dataset_from_files(files):
    image_paths = []
    feature_vectors = []

    os.makedirs("temp_dataset", exist_ok=True)

    for uploaded_file in files:
        file_path = os.path.join("temp_dataset", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract features
        features = extract_features(file_path)
        image_paths.append(file_path)
        feature_vectors.append(features)

    if feature_vectors:
        feature_vectors = np.array(feature_vectors, dtype="float32")
        dimension = feature_vectors.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(feature_vectors)

        # ✅ Store index and paths in session state
        st.session_state.index = index
        st.session_state.image_paths = image_paths
        st.success(f"✅ {len(image_paths)} images loaded successfully!")

# Search Image
def search_image(query_image):
    if st.session_state.index is None or not st.session_state.image_paths:
        st.error("⚠️ Load the dataset first!")
        return None, 0
    
    query_features = extract_features(query_image)
    D, I = st.session_state.index.search(np.array([query_features]), k=1)
    
    matched_path = st.session_state.image_paths[I[0][0]]
    similarity_score = min(max(D[0][0] * 100, 0), 100)  # Ensure it's between 0-100
    
    return matched_path, similarity_score

# Streamlit App
st.title("🔎 Similar Image Search")
st.write("Upload images and search for similar ones.")

# ✅ Drag & Drop Multiple Files Section
uploaded_files = st.file_uploader(
    "📂 Drag and drop images here to load dataset",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("📂 Load Dataset"):
        load_dataset_from_files(uploaded_files)

# ✅ Search Image Section
uploaded_file = st.file_uploader(
    "🔎 Upload an image to search",
    type=["jpg", "jpeg", "png", "webp"]
)

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
            
            # ✅ Display similarity score in percentage
            st.write(f"**Similarity Score:** {similarity_score:.2f}%")
            st.progress(similarity_score / 100)  # ✅ Value between 0-1
'''


import streamlit as st

# ✅ Set page config FIRST before any Streamlit command
st.set_page_config(page_title="🔎 Similar Image Search", layout="wide")

import os
import numpy as np
import cv2
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
import faiss
from skimage.feature import hog
from PIL import Image

# ✅ Load VGG19 Model (Fine-tuned)
@st.cache_resource
def load_model():
    base_model = VGG19(weights="imagenet", include_top=False, pooling="avg")
    model = Model(inputs=base_model.input, outputs=base_model.output)
    return model

model = load_model()

# ✅ Initialize session state
if "index" not in st.session_state:
    st.session_state.index = None
if "image_paths" not in st.session_state:
    st.session_state.image_paths = []

# ✅ Settings
TEXTURE_WEIGHT = 1.8  # Increase texture weight to improve design-based similarity
SIMILARITY_THRESHOLD = 80.0  # Lower threshold for better cropped image matching

# ✅ Feature Extraction
def extract_features(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        deep_features = model.predict(img).flatten()
        deep_features /= np.linalg.norm(deep_features)  # Normalize

        gray_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
        gray_img = cv2.resize(gray_img, (128, 128))
        texture_features = hog(
            gray_img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True
        )
        texture_features /= np.linalg.norm(texture_features)  # Normalize

        # ✅ Combine deep and texture features with higher weight on texture
        combined_features = np.concatenate((deep_features, TEXTURE_WEIGHT * texture_features))
        return combined_features
    except Exception as e:
        st.error(f"❌ Error in feature extraction: {e}")
        return None

# ✅ Load Dataset from Drag & Drop
def load_dataset_from_files(files):
    image_paths = []
    feature_vectors = []

    os.makedirs("temp_dataset", exist_ok=True)

    for uploaded_file in files:
        file_path = os.path.join("temp_dataset", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # ✅ Extract features
        features = extract_features(file_path)
        if features is not None:
            image_paths.append(file_path)
            feature_vectors.append(features)

    if feature_vectors:
        feature_vectors = np.array(feature_vectors, dtype="float32")
        dimension = feature_vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)  # ✅ Use L2 for better pattern-based search
        index.add(feature_vectors)

        # ✅ Store index and paths in session state
        st.session_state.index = index
        st.session_state.image_paths = image_paths
        st.success(f"✅ {len(image_paths)} images loaded successfully!")
    else:
        st.error("❌ No valid images loaded. Check the format and try again.")

# ✅ Search Image
def search_image(query_image):
    if st.session_state.index is None or not st.session_state.image_paths:
        st.error("⚠️ Load the dataset first!")
        return None, 0
    
    query_features = extract_features(query_image)
    if query_features is None:
        return None, 0
    
    D, I = st.session_state.index.search(np.array([query_features]), k=1)

    similarity_score = (1 - (D[0][0] ** 0.5)) * 100
    similarity_score = round(similarity_score, 2)

    if similarity_score < SIMILARITY_THRESHOLD:
        return None, similarity_score
    
    # ✅ Ensure valid index before returning path
    if I[0][0] < len(st.session_state.image_paths):
        matched_path = st.session_state.image_paths[I[0][0]]
        if os.path.exists(matched_path):
            return matched_path, similarity_score
    
    return None, similarity_score

# ✅ Streamlit App
st.title("🔎 Similar Image Search")
st.write("Upload multiple gold jewelry images as a dataset and search for similar designs.")

# ✅ Drag & Drop Multiple Files Section
uploaded_files = st.file_uploader(
    "📂 Drag and drop images here to load dataset",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("📂 Load Dataset"):
        load_dataset_from_files(uploaded_files)

# ✅ Search Image Section
uploaded_file = st.file_uploader(
    "🔎 Upload an image to search",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    # ✅ Save the uploaded file temporarily
    query_image_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(query_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # ✅ Display the uploaded image
    st.image(uploaded_file, caption="Query Image", width=200)
    
    if st.button("🔎 Search"):
        matched_path, similarity_score = search_image(query_image_path)
        
        if matched_path:
            try:
                # ✅ Display matched image
                matched_image = Image.open(matched_path)
                st.image(matched_image, caption=f"Matched Image ({similarity_score:.2f}% Similar)", width=200)
                
                # ✅ Display similarity score in percentage
                st.write(f"**Similarity Score:** {similarity_score:.2f}%")
                st.progress(min(max(similarity_score / 100, 0), 1))
            except FileNotFoundError:
                st.error("❌ Matched image file not found. Try reloading the dataset.")
        else:
            st.warning("⚠️ No similar design found!")

# ✅ Clean up temporary files
import shutil
shutil.rmtree("temp", ignore_errors=True)
shutil.rmtree("temp_dataset", ignore_errors=True)
