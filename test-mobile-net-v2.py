import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

IMG_SIZE = (224, 224)
model = tf.keras.applications.MobileNetV2(weights='imagenet')

def preprocess_image(image_path):
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

def predict_food(image_path):
    img = preprocess_image(image_path)
    preds = model.predict(img)
    decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=3)[0]

    print("Top Predictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded_preds):
        print(f"{i+1}. {label} ({score*100:.2f}%)")

try:
    image_path = "mac-and-cheese.jpg"
    predict_food(image_path)
except Exception as e:
    print(f"Error: {str(e)}")
