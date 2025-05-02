import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import sys

IMG_PATH = sys.argv[1]  # e.g., python predict.py test.jpg
model = tf.keras.models.load_model("crack_detector_model.h5")

img = image.load_img(IMG_PATH, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)[0][0]
if pred > 0.5:
    print(f"🧱 Crack detected (Confidence: {pred:.2f})")
else:
    print(f"✅ No crack detected (Confidence: {1 - pred:.2f})")
