from tensorflow.keras.layers import GaussianNoise, Input, Reshape, Dropout, Flatten
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow import clip_by_value
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall

def make_model_1(input_shape):
    learning_rate = 0.001
    input_layer = Input(input_shape)
    rs = Rescaling(scale=1/255.0)(input_layer)
    gauss = GaussianNoise(1.5/255.0)(rs)
    clip_input = clip_by_value(gauss, 0, 1)
    reshape_input = Reshape((input_shape[0], input_shape[1], 1))(clip_input)

    c1 = Conv2D(16, (3,3), activation="relu")(reshape_input)
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
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy',Precision(), Recall()])
    return model


def make_model_2(input_shape):
    learning_rate = 0.002
    input_layer = Input(input_shape)
    rs = Rescaling(scale=1/255.0)(input_layer)
    gauss = GaussianNoise(0.75/255.0)(rs)
    clip_input = clip_by_value(gauss, 0, 1)
    reshape_input = Reshape((input_shape[0], input_shape[1], 1))(clip_input)

    c1 = Conv2D(8, (3,3), activation="relu")(reshape_input)
    mp1 = MaxPool2D((3,3))(c1)
    c2 = Conv2D(12, (4,4), activation="relu")(mp1)
    mp2 = MaxPool2D((3,3))(c2)
    c3 = Conv2D(16, (5,5), activation="relu")(mp2)
    mp3 = MaxPool2D((3,3))(c3)
    c4 = Conv2D(20, (5,5), activation="relu")(mp3)
    drop1 = Dropout(0.2)(c4) 
    flat = Flatten()(drop1)
    
    d1 = Dense(256, activation="relu")(flat)
    d2 = Dense(128, activation="relu")(d1)
    
    drop2 = Dropout(0.3)(d2)
    output_layer = Dense(1, activation="sigmoid")(drop2)
    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy',Precision(), Recall()])
    return model