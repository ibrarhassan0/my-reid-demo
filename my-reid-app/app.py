import streamlit as st
from PIL import Image
import os
import torch
from torchvision import models, transforms
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.title("Very Simple Thermal Animal Re-ID")
st.write("Upload any photo → see similar ones from your gallery")

# ---------------------------
# Load Model (cached)
# ---------------------------
@st.cache_resource
def get_model():
    model = models.resnet18(weights="IMAGENET1K_V1")
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model

model = get_model()

# ---------------------------
# Image Transform
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------------------
# Feature Extraction
# ---------------------------
def get_features(img):
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        feat = model(img).squeeze().numpy()
    return feat / np.linalg.norm(feat)

# ---------------------------
# Load Gallery Images
# ---------------------------
@st.cache_data
def load_my_photos():
    photos = []
    feats = []

    # Get folder where app.py exists
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    for file in os.listdir(BASE_DIR):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                path = os.path.join(BASE_DIR, file)
                im = Image.open(path).convert("RGB")
                f = get_features(im)
                photos.append((file, im))
                feats.append(f)
            except Exception:
                pass

    return photos, np.array(feats)

photos, features = load_my_photos()

st.write(f"I found {len(photos)} photos in this folder")

# ---------------------------
# Preview Gallery Images
# ---------------------------
if len(photos) > 0:
    st.write("Some of your gallery photos:")
    cols = st.columns(5)
    for i in range(min(10, len(photos))):
        with cols[i % 5]:
            st.image(photos[i][1], width=100)

# ---------------------------
# Upload Image
# ---------------------------
uploaded = st.file_uploader("Choose photo to compare",
                            type=["jpg", "jpeg", "png"])

if uploaded and len(features) > 0:
    query = Image.open(uploaded).convert("RGB")
    st.image(query, caption="Your photo", width=250)

    q_feat = get_features(query)

    sim = cosine_similarity([q_feat], features)[0]
    best = np.argsort(sim)[::-1][:5]  # top 5 matches

    st.write("Top 5 similar photos:")
    cols = st.columns(5)

    for i, idx in enumerate(best):
        name, img = photos[idx]
        score = sim[idx]
        with cols[i]:
            st.image(img, caption=f"{name}\nSim: {score:.2f}")
