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
    # Load and process the hazy image
    hazy_image = cv2.imread(file_path)
    hazy_image = cv2.cvtColor(hazy_image, cv2.COLOR_BGR2RGB)
    original_shape = hazy_image.shape[:2]
    
    # Resize to the model's input shape if necessary
    model_input_shape = (256, 256)
    hazy_image_resized = cv2.resize(hazy_image, model_input_shape)
    hazy_image_resized = np.expand_dims(hazy_image_resized, axis=0)
    hazy_image_resized = hazy_image_resized.astype('float32') / 255.0
    
    # Predict the dehazed image
    dehazed_image_resized = model.predict(hazy_image_resized)
    
    # Resize dehazed output back to original dimensions
    dehazed_image_resized = dehazed_image_resized[0]
    dehazed_image_original = cv2.resize(dehazed_image_resized, (original_shape[1], original_shape[0]))
    
    # Display the input and output images at original size
    subplot(hazy_image / 255.0, dehazed_image_original, 'Hazy Image', 'Dehazed Image')
else:
    print(f"File not found: {file_path}")
