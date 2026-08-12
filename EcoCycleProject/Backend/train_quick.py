import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Config
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
DATASET_DIR = "ecocycle"

# 2. Data generator with auto-validation split (80% train, 20% val)
gen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
    rescale=1./255,
    validation_split=0.2,
)

# 3. Load from your ecocycle folder
train_ds = gen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
)

val_ds = gen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
)

print("Classes found:", list(train_ds.class_indices.keys()))

# 4. Build MobileNetV2 transfer learning model
base = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet',
)

base.trainable = False

x = base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(3, activation='softmax')(x)

model = tf.keras.Model(inputs=base.input, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# 5. Train
print("Training model on your EcoCycle dataset...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)

# 6. Save model
model.save("mobilenetv2_ecocycle.h5")
print("Model saved as 'mobilenetv2_ecocycle.h5'")
print("Classes:", list(train_ds.class_indices.keys()))