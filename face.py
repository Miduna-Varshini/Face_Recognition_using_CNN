import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import gdown
import os

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Face Recognition App",
    page_icon="🧑‍💻",
    layout="centered"
)

# ================= BLACK THEME CSS =================
st.markdown("""
<style>
.stApp {
    background-color: #000000;
    color: white;
}

.card {
    background-color: #111111;
    padding: 25px;
    border-radius: 16px;
    margin-bottom: 20px;
}

.title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    color: white;
}

.subtitle {
    text-align: center;
    color: #cccccc;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

# ================= GOOGLE DRIVE MODEL =================
MODEL_ID = "1lFBCiEl_8dysIbnVH6sRbAAOfCZWORaX"   # 👈 your Drive file ID
MODEL_PATH = "facerecognition.h5"

@st.cache_resource
def load_face_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model from Google Drive..."):
            gdown.download(
                id=MODEL_ID,
                output=MODEL_PATH,
                quiet=False
            )
    model = load_model(MODEL_PATH)
    return model

model = load_face_model()

# ================= CLASS NAMES =================
class_names = [
    "Keerthivaasan",
    "Manohar",
    "Miduna Varshini"
]

# ================= HEADER =================
st.markdown("<div class='title'>🧑‍💻 Face Recognition System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload an image to identify the person</div>", unsafe_allow_html=True)

# ================= IMAGE INPUT =================
st.markdown("<div class='card'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png"]
)

camera_file = st.camera_input("OR Capture using Camera")

st.markdown("</div>", unsafe_allow_html=True)

# ================= IMAGE PROCESSING =================
img = None
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
elif camera_file:
    img = Image.open(camera_file).convert("RGB")

# ================= PREDICTION =================
if img:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.image(img, caption="Input Image", use_container_width=True)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    index = int(np.argmax(prediction))
    confidence = prediction[0][index] * 100

    st.markdown("</div>", unsafe_allow_html=True)

    # ================= RESULT =================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"## ✅ Identified Person: **{class_names[index]}**")
    st.markdown(f"### 🎯 Confidence: **{confidence:.2f}%**")
    st.markdown("</div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown(
    "<p style='text-align:center;color:#888;'>For educational purposes only</p>",
    unsafe_allow_html=True
)
