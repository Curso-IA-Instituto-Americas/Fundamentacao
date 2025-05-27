import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import cv2

# Carregar modelo com cache
@st.cache_resource
def load_model_cached():
    model = tf.keras.models.load_model('./models/xray_model.hdf5', compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

with st.spinner('Model is being loaded...'):
    model = load_model_cached()

class_names = ['NORMAL', 'PNEUMONIA']

st.title("Pneumonia Detection from Chest X-rays")
st.write("Upload a chest X-ray image to classify.")

file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def import_and_predict(image_data, model):
    size = (180, 180)
    image = ImageOps.fit(image_data, size, method=Image.Resampling.LANCZOS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
        
    return prediction
    
if file is None:
    st.text("Please upload an image file.")
else:
    image = Image.open(file)
    st.image(image, use_container_width=True)

    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])

    st.write("Predictions (raw):", predictions)
    st.write("Confidence Scores:", score.numpy())

    st.success(
        f"This image most likely belongs to **{class_names[np.argmax(score)]}** "
        f"with a **{100 * np.max(score):.2f}%** confidence."
    )