import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

model = tf.keras.applications.MobileNetV2(weights='imagenet')

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
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

image_path = "mac-and-cheese.jpg"
predict_food(image_path)
