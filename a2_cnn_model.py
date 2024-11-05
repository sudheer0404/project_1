import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

class DehazeCNN:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        try:
            model = load_model(self.model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def preprocess_image(self, file_path):
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        hazy_image = cv2.imread(file_path)
        hazy_image = cv2.cvtColor(hazy_image, cv2.COLOR_BGR2RGB)
        # Normalize the image without resizing
        hazy_image = hazy_image.astype('float32') / 255.0
        hazy_image = np.expand_dims(hazy_image, axis=0)  # Add batch dimension
        print("Preprocessed image shape (original size):", hazy_image.shape)
        return hazy_image

    def predict(self, preprocessed_image, original_size, save_path=None):
        if self.model is None:
            print("Model not loaded.")
            return None
        
        # Run model prediction
        dehazed_image = self.model.predict(preprocessed_image)
        
        if dehazed_image.size > 0:
            # Rescale and convert back to original image format
            dehazed_image = np.clip((dehazed_image[0] * 255), 0, 255).astype(np.uint8)
            dehazed_image = cv2.resize(dehazed_image, original_size)  # Resize to original size
            
            if save_path:
                dehazed_image_bgr = cv2.cvtColor(dehazed_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, dehazed_image_bgr)
                print(f"Dehazed image saved to: {save_path}")
            return dehazed_image  # CNN prediction with original size
        else:
            print("Model returned an empty prediction.")
            return None

# Example usage:
# model_path = 'dehaze_cnn_model.h5'
# dehaze_cnn = DehazeCNN(model_path)
# original_image_path = 'hellohaze.webp'
# save_path = 'dehazed_output.webp'
# original_image = cv2.imread(original_image_path)
# original_size = (original_image.shape[1], original_image.shape[0])  # (width, height)
# preprocessed_image = dehaze_cnn.preprocess_image(original_image_path)
# if preprocessed_image is not None:
#     dehazed_image = dehaze_cnn.predict(preprocessed_image, original_size, save_path=save_path)
