import os
import tensorflow as tf
import matplotlib.pyplot as plt
PATH_HAZE = 'dataset/'
MODEL_SAVE_PATH = 'dehaze_cnn_model.h5'
BATCH_SIZE = 4
image_shape = (256, 256, 3)
def load_image(image_path): #preprocessing
    img = tf.io.read_file(image_path)
    img = tf.io.decode_png(img, channels=3)
    img = tf.image.resize(img, [256, 256])
    img = (img / 127.5) - 1.0
    return img
def load_paired_data(real_image_path, fake_image_path): #loading the pairs
    real_image = load_image(real_image_path)
    fake_image = load_image(fake_image_path)
    return real_image, fake_image
def load_dataset(PATH, test_ds=False):
    haze_images_paths = sorted([os.path.join(PATH, 'hazy', item) for item in os.listdir(os.path.join(PATH, 'hazy'))])
    clear_images_paths = sorted([os.path.join(PATH, 'clear', item) for item in os.listdir(os.path.join(PATH, 'clear'))])
    dataset = tf.data.Dataset.from_tensor_slices((haze_images_paths, clear_images_paths))
    dataset = dataset.map(load_paired_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(1 if test_ds else BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset
def build_dehazing_model(): #cnn
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=image_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D((2, 2)),
        tf.keras.layers.Conv2D(3, (3, 3), activation='tanh', padding='same')
    ])
    return model
train_ds = load_dataset(os.path.join(PATH_HAZE, 'train/'))
test_ds = load_dataset(os.path.join(PATH_HAZE, 'test_moderate/'), test_ds=True)
model = build_dehazing_model()
model.compile(optimizer='adam', loss='mse')
epochs = 10
model.fit(train_ds, epochs=epochs)

model.save(MODEL_SAVE_PATH)
print(f"Model saved to '{MODEL_SAVE_PATH}'")
