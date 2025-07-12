import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model("mnist_model.h5")

st.title("🧠 MNIST Digit Recognizer")
st.write("Upload a grayscale digit image (white digit on black background, 28x28 preferred).")

uploaded_file = st.file_uploader("Upload a digit image (PNG, JPG)...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Step 1: Load image and convert to grayscale
    image = Image.open(uploaded_file).convert("L")

    # Step 2: Resize to 28x28 (model input size)
    image = image.resize((28, 28), Image.LANCZOS)

    # Step 3: Show original
    st.image(image, caption="Original Image (grayscale)", width=150)

    # Step 4: Invert if background is white (check mean pixel value)
    mean_pixel = np.array(image).mean()
    if mean_pixel > 127:
        image = ImageOps.invert(image)

    # Step 5: Binarize (thresholding)
    image = image.point(lambda x: 255 if x > 100 else 0)

    # Step 6: Normalize and reshape
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Step 7: Display what the model actually sees
    st.image(img_array[0], caption="What the Model Sees", width=150, clamp=True)

    # Step 8: Predict
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    st.success(f"Predicted Digit: **{predicted_digit}**")
    st.bar_chart(prediction[0])
