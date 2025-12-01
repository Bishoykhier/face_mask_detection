
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("improved_mask_detector.h5")
img_size = 224

st.title("😷 Face Mask Detection App")
st.write("Upload an image to check if the person is wearing a mask.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img = img.resize((img_size, img_size))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "Mask" if prediction < 0.5 else "No Mask"
    confidence = (1 - prediction) if label == "Mask" else prediction

    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {confidence*100:.2f}%")
