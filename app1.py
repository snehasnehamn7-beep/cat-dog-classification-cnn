import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱",
    layout="centered"
)

# Load model
model = tf.keras.models.load_model("cat_dog_mobilenet.h5")

# Title
st.markdown("<h1 style='text-align: center;'>Cat vs Dog Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image and let AI predict</p>", unsafe_allow_html=True)
st.divider()

# Uploadnet
uploaded_file = st.file_uploader(
    " Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button(" Predict"):
        with st.spinner("AI is thinking..."):
            prediction = model.predict(img_array)[0][0]

        confidence = prediction if prediction > 0.5 else 1 - prediction

        st.divider()
        if prediction > 0.5:
            st.success(f"**Dog Detected**  \nConfidence: {confidence*100:.2f}%")
        else:
            st.success(f" **Cat Detected**  \nConfidence: {confidence*100:.2f}%")

st.divider()
st.caption("Built with  using CNN & Streamlit")