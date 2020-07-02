import os
import random
import datetime

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
FILE_DIR = ""
DB_DIR_COVID = f"{FILE_DIR}/"
MODEL_DIR = f"{FILE_DIR}/"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
RAND_STATE = 5
TIMESTEPS = 6
ALPHA = 0.01

import utils
import visdom_utils

def CAS(input_shape):
    """Collaborative Attention Segmentation model."""
    # Define Encode Layers
    input = tf.keras.layers.Input(input_shape)
    conv1 = tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=input_shape, padding='same')
    pool1 = tf.keras.layers.MaxPooling2D((5, 5))
    conv2 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding='same')
    pool2 = tf.keras.layers.MaxPooling2D((5, 5))
    conv3 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding='same')

    # Define decoder layers
    convt1 = tf.keras.layers.Conv2DTranspose(64, 3, activation='relu', padding='same')
    upsampling1 = tf.keras.layers.UpSampling2D((5, 5))
    convt2 = tf.keras.layers.Conv2DTranspose(64, 3, activation='relu', padding='same')
    upsampling2 = tf.keras.layers.UpSampling2D((5, 5))
    convt3 = tf.keras.layers.Conv2DTranspose(32, 3, padding='same', activation='tanh')
    reshape_output = tf.keras.layers.Conv2DTranspose(1, 3, padding='same', activation=step_sigmoid)

    # Encoder
    x = conv1(input)
    x = pool1(x)
    x = conv2(x)
    x = pool2(x)
    x = conv3(x)

    # Decoder
    x = convt1(x)
    x = upsampling1(x)
    x = convt2(x)
    x = upsampling2(x)
    x = convt3(x)
    x = reshape_output(x)

    model = tf.keras.models.Model(input, x)

    return model


def step_sigmoid(x):
    """Approximate a continuous step function."""
    k = -10.0
    exponential = tf.math.exp(tf.multiply(k,x-.5))
    output = tf.math.divide(1.0, 1.0 + exponential)
    return output


@tf.function
def CAS_loss(img, mask):
    """CAS loss, average of all values in mask"""
    return ALPHA*tf.reduce_mean(mask)


@tf.function
def CAS_crossentropy(img, mask):
    """Personalized crossentropy loss for DiscoveryRNN model."""
    rnn_shape = (tf.shape(mask)[0], TIMESTEPS, 100, 50, 1)

    # Reshape inputs for RNN
    img_reshaped = tf.reshape(img, (rnn_shape))
    mask_reshaped = tf.reshape(mask, (rnn_shape))

    # Generate new image from output mask
    gen_img_reshaped = tf.math.multiply(img_reshaped, mask_reshaped)

    # Generate logits/labels from original and generated images
    labels = img_reshaped
    logits = gen_img_reshaped
    for i in range(len(TARGET_MODEL.layers)):
        layer = TARGET_MODEL.layers[i]
        if layer.name == "dense_1":
            labels = layer(labels[0])
            logits = layer(logits[0])
        else:
            labels = layer(labels)
            logits = layer(logits)

    # Get binary classification for labels
    labels = step_sigmoid(labels)

    # Calculate loss
    return -tf.reduce_sum(labels*tf.math.log(tf.clip_by_value(logits,1e-10,1.0)), axis=1)

@tf.function
def custom_loss(img, mask):
    """Add both custom losses"""
    cas_loss = CAS_loss(img, mask)
    crossentropy_loss = CAS_crossentropy(img, mask)
    total_loss = tf.add(cas_loss, crossentropy_loss)
    return total_loss


def generate_batches(data, batch_size):
    """Generate shuffled batches."""
    random.shuffle(data)
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]


def import_target_dataset():
    """Import cough dataset."""
    X, Y = utils.get_dataset_from_pickle(EXP_TYPE)
    X, X = utils.normalize_dataset(X, X)
    if EXP_TYPE == "country":
        X, Y = utils.switch_to_binary(X, Y)
    return X, Y


def reshape_to_cnn(X):
    """Reshape input with size [None, timesteps, time, mfcc_size, channels] to
    [None, time*timesteps, mfcc_size, channels]"""
    return X.reshape([X.shape[0], X.shape[2]*X.shape[1], X.shape[3], X.shape[4]])


def plot_img_square(imgs, filename, title):
    """Plot square of generated images."""
    r, c = 4, 4
    imgs = np.squeeze(imgs)
    fig, axs = plt.subplots(r, c)
    cnt = 0
    fig.suptitle(title)
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(imgs[cnt, :,:], cmap='gray', aspect="auto")
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(filename, format='png', dpi=1200)
    plt.close()


def plot_imgs(model, X, epoch, batch_size, binary=False):
    """Plot 16 masks, generated images and reference images."""
    test_samples = len(X)-len(X)%batch_size
    X = X[0:test_samples]
    masks = model.predict(X)
    masks = np.squeeze(masks[0:16])
    ref_imgs = np.squeeze(X[0:16])

    gen_imgs = np.multiply(masks, ref_imgs)

    save_dir = f"{FILE_DIR}/imgs/{EXP_TYPE}/alpha_{ALPHA}"

    utils.check_if_dir_exists(save_dir)
    plot_img_square(ref_imgs,
                    f"{save_dir}/ref_img_wAct3c.png",
                    "Reference Images")
    plot_img_square(masks,
                    f"{save_dir}/mask_epoch_{epoch}_wAct3c.png",
                    f"Masks for '{EXP_TYPE}' model with a={ALPHA}")
    plot_img_square(gen_imgs,
                    f"{save_dir}/gen_img_epoch_{epoch}_wAct3c.png",
                    f"Generated images for '{EXP_TYPE}' model with a={ALPHA}")



def train_model(model, data, hyperparams):
    """Train provided model with speech training dataset."""
    # Compile model with loss function
    model.compile(hyperparams["optimizer"],
                  loss=hyperparams["loss"],
                  metrics=hyperparams["metrics"]
                  )
    batches_per_epoch = data["X_train"].shape[0]//hyperparams["batch_size"]
    for epoch in range(hyperparams["epochs"]):
        print(f"\nEPOCH {epoch} for ALPHA {ALPHA}:\n")
        batches = generate_batches(data["X_train"], hyperparams["batch_size"])
        batch_count = 0
        model.fit(data["X_train"],
                  data["X_train"],
                  validation_data=[data["X_test"], data["X_test"]],
                  epochs=1)

        # Print results
        if epoch%hyperparams["save_interval"] is 0:
            plot_imgs(model, data["X_test"], epoch, hyperparams["batch_size"])

    return model

def train_CAS():
    """Train LSTM model."""

    global EXP_TYPE
    EXP_TYPE = "country"

    global TARGET_MODEL
    TARGET_MODEL = tf.keras.models.load_model(f"{MODEL_DIR}/ResNet50_complete_countries_vFinal2_b128.h5")
    # Import data
    mirrored_strategy = tf.distribute.MirroredStrategy(["/gpu:0", "/gpu:1"])
    with mirrored_strategy.scope():
        X, Y = import_target_dataset()
        X_cnn = reshape_to_cnn(X)

        X_train, X_test, y_train, y_test = train_test_split(X_cnn,
                                                            Y,
                                                            test_size=0.3,
                                                            shuffle=True,
                                                            random_state=RAND_STATE)

        #X_train, y_train = utils.augment_dataset(X_train, y_train)

        data = {
            'X_train': X_train,
            'X_test': X_test[0:200],
            'y_train': y_train,
            'y_test': y_test,
        }

        # Define model parameters
        model_type = "CAS_CNN"
        input_shape = X_cnn.shape[1:]

        # Define callbacks
        log_dir = f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        tboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                histogram_freq=1,
                                                profile_batch=0)

        filename = f"{model_type}_CAS_{EXP_TYPE}_v1"
        batch = 128
        hyperparameters = {
            "epochs": 4,
            "batch_size": batch,
            "loss": custom_loss,
            "optimizer": tf.keras.optimizers.Adam(),
            "metrics": [CAS_crossentropy, CAS_loss],
            "callbacks": [tboard],
            "save_file": f"models/{filename}_b{batch}.h5",
            "save_interval": 1
        }

        # Choose and train model
        CASmodel = CAS(input_shape=input_shape)

        train_model(CASmodel, data, hyperparameters)

    return None


def main():
    """Executed code on file call."""
    train_CAS()
    return None


if __name__ == '__main__':
    main()
