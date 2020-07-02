"""
COVID19 Cough Segmentation.

This program is used to train a CNN-LSTM for COVID19 coughs diagnosis.
"""
import datetime
import os

import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split

import utils

FILE_DIR = "/home/fhueto/code/covid_lstm"
DB_DIR_COVID = f"{FILE_DIR}/dataset"
MODEL_DIR = f"{FILE_DIR}/models"
RAND_STATE = 5

def import_dataset(var="gender"):
    """Select dataset based on string variable."""
    X, Y = utils.get_dataset_from_pickle(var)
    X, _ = utils.normalize_dataset(X, X)
    return (X, Y)


def reshape_timesteps(X, width=4, stride=2):
    """Reshape timesteps so as to have them overlap"""
    new_X = []
    for x in X:
        new_x = [np.concatenate(x[0:2]), np.concatenate(x[1:3]), np.concatenate(x[2:4]), np.concatenate(x[3:5])]
        new_X.append(new_x)

    return np.asarray(new_X)


def choose_model(input_shape, classes=30, model_name="local"):
    """Choose model from model_name string."""
    input_size = input_shape[1:]
    if model_name == "VGG16":
        model = tf.keras.applications.VGG16(include_top=False,
                                            weights=None,
                                            input_shape=input_size,
                                            pooling=None,
                                            classes=classes)
    elif model_name == "DenseNet201":
        model = tf.keras.applications.DenseNet201(include_top=False,
                                                  weights=None,
                                                  input_shape=input_size,
                                                  pooling=None,
                                                  classes=classes)
    elif model_name == "ResNet50":
        model = tf.keras.applications.ResNet50V2(include_top=False,
                                                 weights=None,
                                                 input_shape=input_size,
                                                 pooling=None,
                                                 classes=classes)
    elif model_name == "DenseNet201_trained":
        return tf.keras.models.load_model(f'{MODEL_DIR}models/DenseNet_1.h5')
    elif model_name == "ResNet50_trained":
        return tf.keras.models.load_model(f'{MODEL_DIR}models/ResNet50_1.h5')
    else:
        return None

    model = define_lstm_model(model, input_shape, classes=classes)
    return model


def define_lstm_model(model, input_shape, classes=3):
    """Add LSTM layer at the end of a ConvNet."""
    # Define model layers
    input_layer = keras.layers.Input(shape=input_shape)
    pooling = tf.keras.layers.GlobalAveragePooling2D()
    dense_1 = tf.keras.layers.Dense(256, activation='relu')
    dropout_1 = tf.keras.layers.Dropout(0.3)
    lstm_1 = tf.keras.layers.LSTM(units=64, return_state=True, dropout=0.3)
    # lstm_2 = tf.keras.layers.LSTM(units=128, return_state=False)
    dense_output = tf.keras.layers.Dense(classes, activation='softmax')

    # Define time distributed CNN model
    x = model.output
    x = pooling(x)
    x = dense_1(x)
    x = dropout_1(x)
    CNN_model = tf.keras.models.Model(inputs=model.inputs, outputs=x)
    CNN_model_distributed = tf.keras.layers.TimeDistributed(CNN_model)(input_layer)

    # Define LSTM model
    x, state_h, state_c = lstm_1(CNN_model_distributed)
    # x = lstm_2(x, initial_state=(state_h, state_c))
    output = dense_output(x)
    LSTM_model = tf.keras.models.Model(inputs=input_layer, outputs=output)
    return LSTM_model


def train_model(model, data, hyperparams):
    """Train provided model with speech training dataset."""
    # Compile model with loss function
    model.compile(hyperparams["optimizer"],
                  loss=hyperparams["loss"],
                  metrics=hyperparams["metrics"])

    chk = keras.callbacks.ModelCheckpoint(hyperparams["save_file"],
                                          monitor='val_accuracy',
                                          verbose=1,
                                          save_best_only=True,
                                          save_weights_only=False,
                                          mode='auto',
                                          period=1)
    early = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                          min_delta=0,
                                          patience=8,
                                          verbose=1,
                                          mode='auto')
    log_dir = f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                            histogram_freq=1,
                                            profile_batch=0)
    cb = [chk, early, tboard]
    model.fit(data["X_train"],
              data["y_train"],
              validation_data=(data["X_test"], data["y_test"]),
              epochs=hyperparams["epochs"],
              batch_size=hyperparams["batch_size"],
              verbose=1,
              callbacks=cb)
    return model


def main():
    """Train LSTM model."""
    # Import data
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    mirrored_strategy = tf.distribute.MirroredStrategy(["/gpu:0", "/gpu:1"])
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        X, Y = import_dataset("gender")
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            Y,
                                                            test_size=0.3,
                                                            shuffle=True,
                                                            random_state=RAND_STATE)

        X_train, y_train = utils.augment_dataset(X_train, y_train)

        data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
        }

        # Define model parameters
        model_type = "ResNet50"
        input_shape = (None, X_train.shape[2], X_train.shape[3], X_train.shape[4])

        filename = f"{model_type}_complete_gender_vFinal2"
        batch = 128
        hyperparameters = {
            "epochs": 1000,
            "batch_size": batch,
            "loss": "categorical_crossentropy",
            "optimizer": keras.optimizers.Adam(),
            "metrics": ["accuracy"],
            "save_file": f"models/{filename}_b{batch}.h5"
        }

        #
        try:
            classes = y_train.shape[1]
        except:
            classes = 1

        # Choose and train model
        model = choose_model(input_shape, classes, model_type)
        train_model(model, data, hyperparameters)

    return None


if __name__ == '__main__':
    main()
