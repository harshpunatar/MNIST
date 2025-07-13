# MNIST Digit Recognizer (Streamlit App)

A machine learning web app built using **Streamlit** and **TensorFlow** that recognizes digits (0–9) from uploaded images using a CNN trained on the 'MNIST dataset'.

---

## Features

- Upload a grayscale image of a digit (28x28 preferred)
- Automatically preprocesses and resizes the image
- Predicts digit using a trained model with 98.63% accuracy
- Shows prediction probabilities in a bar chart

---

## Files

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit application |
| `mnist_model.h5` | Trained CNN model |
| `README.md` | Project documentation |

---

## How to Run

1. Clone the repo**
```bash
git clone https://github.com/harshpunatar/MNIST.git
cd MNIST
```

2. Install the Dependencies
```bash
pip install streamlit tensorflow numpy pillow
```

3. Run the app
```bash
streamlit run app.py
```

---

## Model Training

The CNN model was trained on the **MNIST** dataset using TensorFlow and Keras.

### Dataset
- **MNIST**: 60,000 training images and 10,000 test images of handwritten digits (0–9)
- Each image is 28×28 pixels in grayscale

### Architecture
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

### Training
```python

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
```

---


### Result

Test Accuracy: 98.63%
<br>
Model saved as: mnist_model.h5

