from os import path, listdir
from PIL import Image
import numpy as np
from sklearn.metrics import f1_score

from tensorflow.keras.layers import GaussianNoise, Input, Rescaling, Reshape, Dropout, Flatten
from tensorflow import clip_by_value
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def get_data(directory: str) -> "tuple[np.ndarray, np.ndarray]":
    ima = listdir(path.join(directory, "PNEUMONIA"))
    nema = listdir(path.join(directory, "NORMAL"))
    size = len(ima) + len(nema)
    X = np.empty((size, 250, 350), dtype=np.uint8)
    y = np.empty((size), dtype=np.uint8)
    for i, file in enumerate(ima):
        X[i] = np.asarray(Image.open(path.join(directory, "PNEUMONIA", file)))
        y[i] = 1
    for i, file in enumerate(nema):
        X[len(ima) + i] = np.asarray(Image.open(path.join(directory, "NORMAL", file)))
        y[len(ima) + i] = 0
    return X , y


def make_model(input_shape, learning_rate):
    input_layer = Input(input_shape)
    rs = Rescaling(scale=1/255)(input_layer)
    gauss = GaussianNoise(2/255)(rs)
    clip_input = clip_by_value(gauss, 0, 1)
    reshape_input = Reshape((input_shape[0], input_shape[1], 1))(clip_input)

    c1 = Conv2D(8, (3,3), activation="relu")(reshape_input)
    mp1 = MaxPool2D((3,3))(c1)
    c2 = Conv2D(16, (4,4), activation="relu")(mp1)
    mp2 = MaxPool2D((4,4))(c2)
    c3 = Conv2D(32, (5,5), activation="relu")(mp2)
    mp3 = MaxPool2D((5,5))(c3)
    drop1 = Dropout(0.2)(mp3) 
    flat = Flatten()(drop1)
    
    d1 = Dense(256, activation="relu")(flat)
    d2 = Dense(128, activation="relu")(d1)
    
    drop2 = Dropout(0.3)(d2)
    output_layer = Dense(1, activation="sigmoid")(drop2)
    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=learning_rate), metrics=["accuracy"])
    return model


if __name__ == "__main__":
    directory_train = "../chest_xray_norm/train"
    x_train, y_train = get_data(directory_train)

    directory_val = "../chest_xray_norm/val"
    x_val, y_val = get_data(directory_val)
    
    model = make_model(x_train[0].shape, 0.01)
    model.summary()

    model.fit(x_train, y_train, epochs=20, batch_size=0, verbose=1, shuffle=True, validation_data=(x_val, y_val))
