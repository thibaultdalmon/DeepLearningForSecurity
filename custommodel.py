from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate, Dropout, Conv2D, LSTM, Lambda
from keras.models import Model, load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import xception
from keras import backend as K
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.utils import Sequence
import numpy as np


class CustomModel:

    def __init__(self, nb_img=5, img_width=299, img_height=299, reuse=False):

        self.nb_img = 5
        self.img_width = 299
        self.img_height = 299

        # This removes all operations linked to training the model
        K.set_learning_phase(0)

        model_dict = {}
        # load model. include_top=False means that we do not load the last
        # fully connected layer(s) necessary for classification
        for i in range(nb_img):
            model_dict[f'model_{i}'] = xception.Xception(
                weights='imagenet', include_top=False, input_shape=(
                    img_width, img_height, 3))
            # Freeze the layers
            for layer in model_dict[f'model_{i}'].layers[:]:
                layer.trainable = False
                layer.name = str(f"X{i}_") + layer.name

        conv_1 = Conv2D(
            16,
            1,
            strides=(
                1,
                1),
            padding='same',
            activation="relu")
        conv_2 = Conv2D(
            16,
            4,
            strides=(
                4,
                4),
            padding='same',
            activation="relu")
        flatten = Flatten()
        dropout = Dropout(0.3)
        dense = Dense(16, activation="relu")
        expand_dims = Lambda(lambda x: K.expand_dims(x, axis=1))

        for i in range(nb_img):
            x = model_dict[f'model_{i}'].output
            x_2 = conv_1(x)
            x_3 = conv_2(x_2)
            x_4 = flatten(x_3)
            x_5 = dropout(x_4)
            y = dense(x_5)
            model_dict[f'model_{i}'] = Model(
                inputs=model_dict[f'model_{i}'].input, outputs=y)

        # Adding custom Layers
        x = Concatenate(axis=1)(
            [expand_dims(model_dict[f'model_{i}'].output) for i in range(nb_img)])
        #x = Flatten()(x)
        x = Dropout(0.3)(x)
        x = LSTM(256, activation="relu", return_sequences=True)(x)
        x = Dropout(0.3)(x)
        x = LSTM(256, activation="relu")(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation="relu")(x)
        x = Dropout(0.3)(x)
        predictions = Dense(1, activation="sigmoid")(x)

        # creating the final model
        self.model = Model(
            inputs=[
                model_dict[f'model_{i}'].input for i in range(nb_img)],
            outputs=predictions)

        # compile the model
        self.model.compile(loss="binary_crossentropy",
                           optimizer=optimizers.Adam(),
                           metrics=["accuracy"])
        if reuse:
            self.model = load_model('V1.h5')

    def train(self, training_generator, validation_generator, save=True):

        self.model.fit_generator(generator=training_generator,
                                 validation_data=validation_generator,
                                 use_multiprocessing=True,
                                 workers=12, epochs=1,
                                 shuffle=True)
        if save:
            self.model.save('V1.h5')

    def predict(self, X):
        
        y = self.model.predict(X)
        return y
