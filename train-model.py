import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
FINE_TUNE_EPOCHS = 5
LEARNING_RATE = 0.0001

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
])

def load_dataset():
    (train_ds, val_ds), dataset_info = tfds.load(
        "food101",
        split=["train[:80%]", "train[80%:]"],
        as_supervised=True, 
        with_info=True
    )
    class_names = dataset_info.features["label"].names
    return train_ds, val_ds, class_names

def preprocess_image(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.keras.layers.Rescaling(1./255)(image)
    image = data_augmentation(image)
    return image, label

def prepare_data(train_ds, val_ds):
    train_ds = train_ds.map(preprocess_image).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(preprocess_image).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    return train_ds, val_ds

def build_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Freeze base model

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def fine_tune_model(model, base_model):
    base_model.trainable = True
    for layer in base_model.layers[:100]:  # Keep first 100 layers frozen
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),  # Lower LR for fine-tuning
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_model(model, train_ds, val_ds, epochs):
    return model.fit(train_ds, validation_data=val_ds, epochs=epochs)

def predict_food(model, class_names, image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    preds = model.predict(img_array)
    predicted_label = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100

    print(f"Predicted: {predicted_label} ({confidence:.2f}%)")

if __name__ == "__main__":
    train_ds, val_ds, class_names = load_dataset()
    train_ds, val_ds = prepare_data(train_ds, val_ds)

    model = build_model(len(class_names))
    history = train_model(model, train_ds, val_ds, EPOCHS)

    model = fine_tune_model(model, model.layers[0])
    fine_tune_history = train_model(model, train_ds, val_ds, FINE_TUNE_EPOCHS)

    model.save("food101_mobilenetv2.h5")
    
    predict_food(model, class_names, "mac-and-cheese.jpg")
