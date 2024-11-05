from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
model_path = 'dehaze_cnn_model.h5'
model = load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
print("Model loaded successfully.")
def subplot(img1, img2, title1, title2):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    img1 = (img1 + 1) / 2.0
    img2 = (img2 + 1) / 2.0
    axes[0].imshow(img1)
    axes[0].set_title(title1)
    axes[0].axis('off')
    axes[1].imshow(img2)
    axes[1].set_title(title2)
    axes[1].axis('off')
    plt.show()
file_path = 'hellohaze.webp'
if os.path.exists(file_path):
    hazy_image = cv2.imread(file_path)
    hazy_image = cv2.cvtColor(hazy_image, cv2.COLOR_BGR2RGB)
    hazy_image_resized = cv2.resize(hazy_image, (256, 256))
    print("Input image shape for model:", hazy_image_resized.shape)
    hazy_image_resized = np.expand_dims(hazy_image_resized, axis=0)
    print("Model input shape with batch size:", hazy_image_resized.shape)
    hazy_image_resized = hazy_image_resized.astype('float32') / 255.0
    dehazed_image = model.predict(hazy_image_resized)
    if dehazed_image.size > 0:
        subplot(hazy_image_resized[0], dehazed_image[0], 'Hazy Image', 'Dehazed Image')
    else:
        print("Model returned an empty prediction.")
else:
    print(f"File not found: {file_path}")
