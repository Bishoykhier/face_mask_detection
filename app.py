
# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.preprocessing import image
# from PIL import Image

# # Load model
# model = tf.keras.models.load_model("improved_mask_detector.h5")
# img_size = 224

# st.title("Face Mask Detection App")
# st.write("Upload an image to check if the person is wearing a mask.")

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     img = Image.open(uploaded_file).convert('RGB')
#     st.image(img, caption='Uploaded Image', use_column_width=True)

#     # Preprocess
#     img = img.resize((img_size, img_size))
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     # Predict
#     prediction = model.predict(img_array)[0][0]
#     label = "Mask" if prediction < 0.5 else "No Mask"
#     confidence = (1 - prediction) if label == "Mask" else prediction

#     st.write(f"**Prediction:** {label}")
#     st.write(f"**Confidence:** {confidence*100:.2f}%")


import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("improved_mask_detector.h5")
img_size = 224

st.title("😷 Face Mask Detection App")
st.write("Upload an image or use your webcam to check if the person is wearing a mask.")

# Sidebar options
option = st.sidebar.selectbox("Choose Mode", ["Upload Image", "Webcam Detection"])

# --- Image Upload Mode ---
if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Preprocess image
        img = img.resize((img_size, img_size))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)[0][0]
        label = "Mask" if prediction < 0.5 else "No Mask"
        confidence = (1 - prediction) if label == "Mask" else prediction

        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {confidence*100:.2f}%")

# --- Webcam Detection Mode ---
elif option == "Webcam Detection":
    st.write("Click **Start** to begin webcam detection.")
    run = st.checkbox("Start Webcam")

    if run:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face, (img_size, img_size))
                face = face.astype("float") / 255.0
                face = np.expand_dims(face, axis=0)

                pred = model.predict(face)[0][0]
                label = "Mask" if pred < 0.5 else "No Mask"
                confidence = (1 - pred) if label == "Mask" else pred
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Convert frame to RGB for Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB")

        cap.release()



# python -m streamlit run app.py