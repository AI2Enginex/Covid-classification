import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint


class Load_Data:

    def __init__(self, model_name, patience):

        self.datagen = ImageDataGenerator(1/255.0)
        self.model_checkpoints = ModelCheckpoint(
            model_name, monitor='val_accuracy', verbose=2, mode='max')
        self.callback = EarlyStopping(
            monitor='val_loss', patience=patience, verbose=2, mode='min')

    def training_set(self, directory, size, color_mode, class_mode, batch):

        self.training = self.datagen.flow_from_directory(
            directory=directory,
            target_size=(size, size),
            color_mode=color_mode,
            class_mode=class_mode,
            batch_size=batch,
            seed=42)

        return self.training

    def testing_set(self, directory, size, color_mode, class_mode, batch):

        self.testing = self.datagen.flow_from_directory(

            directory=directory,
            target_size=(size, size),
            color_mode=color_mode,
            class_mode=class_mode,
            batch_size=batch,
            seed=42)
        return self.testing


class Build_Model:

    def __init__(self, model_name, patience, directory1, directory2, size, color_mode, class_mode, batch):

        data = Load_Data(model_name, patience)

        self.image_dim = size
        self.train = data.training_set(
            directory1, size, color_mode, class_mode, batch)
        self.test = data.testing_set(
            directory2, size, color_mode, class_mode, batch)
        self.checkpoints = data.model_checkpoints
        self.early_stopping = data.callback

    def train_model(self, filter_count, channel, class_count, activation_func, padding_value, layer_acf, dropout_rate):

        self.datagen_model = keras.models.Sequential([

            keras.layers.Conv2D(filters=filter_count, activation=layer_acf, kernel_size=(
                3, 3), padding=padding_value, input_shape=(self.image_dim, self.image_dim, channel)),
            keras.layers.BatchNormalization(),

            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Dropout(dropout_rate),

            keras.layers.Conv2D(filters=filter_count, activation=layer_acf,
                                padding=padding_value, kernel_size=(3, 3)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Dropout(dropout_rate),

            keras.layers.Conv2D(filters=filter_count, activation=layer_acf,
                                padding=padding_value, kernel_size=(3, 3)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Dropout(dropout_rate),

            keras.layers.Conv2D(filters=filter_count, activation=layer_acf,
                                padding=padding_value, kernel_size=(3, 3)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Dropout(dropout_rate),

            keras.layers.Conv2D(filters=filter_count, activation=layer_acf,
                                padding=padding_value, kernel_size=(3, 3)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Dropout(dropout_rate),



            keras.layers.Flatten(input_shape=(
                self.image_dim, self.image_dim, channel)),
            keras.layers.Dense(800, activation=layer_acf),
            keras.layers.Dense(700, activation=layer_acf),
            keras.layers.Dense(600, activation=layer_acf),
            keras.layers.Dense(class_count, activation=activation_func)
        ])

    def model_fitting(self, filter_count, channel, class_count, activation_func, padding_value, layer_acf, dropout_rate, epochs, loss_func, steps):

        self.train_model(filter_count, channel, class_count,
                         activation_func, padding_value, layer_acf, dropout_rate)
        self.datagen_model.compile(optimizer=tf.optimizers.Adam(
            learning_rate=0.001), loss=loss_func, metrics=["accuracy"])
        self.datagen_model.fit_generator(self.train, epochs=epochs, validation_data=(
            self.test), steps_per_epoch=len(self.train)//steps, callbacks=[self.early_stopping, self.checkpoints], verbose=2)


if __name__ == '__main__':

    keras_model = Build_Model(model_name="2020_covid.h5", patience=15, directory1="D:/CNN_Projects/Covid/train",
                              directory2="D:/CNN_Projects/Covid/test", size=48, color_mode="rgb",
                              class_mode="categorical", batch=16)
    keras_model.model_fitting(filter_count=64, channel=3, class_count=1, activation_func='sigmoid', padding_value='same', layer_acf='relu', dropout_rate=0.2, epochs=30,
                              loss_func='binary_crossentropy', steps=16)
