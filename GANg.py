import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.losses import mean_squared_error as mse
from keras.losses import binary_crossentropy  #原来的低版本里的from keras.objectives import binary_crossentropy不可以 

from keras.layers import Dense, Activation, Input, Lambda, Layer
from keras.regularizers import l2
from keras.optimizers import Adam

from keras import backend as K
from tensorflow.keras.backend import clear_session
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from imblearn.over_sampling import SMOTE
np.random.seed(123)
from sklearn.ensemble import RandomForestClassifier

def GANtest(traindata):
    baddata = traindata
    def make_latent_samples(n_samples, sample_size):
        return np.random.normal(loc=0, scale=1, size=(n_samples, sample_size))
    def make_GAN(sample_size,
                g_hidden_size,
                d_hidden_size,
                g_learning_rate,
                d_learning_rate):
        K.clear_session()

        generator = Sequential([
            Dense(g_hidden_size, input_shape=(sample_size,)),
            Activation('relu'),
    #         Dense(10),
            Dense(2),
            Activation('sigmoid')
        ], name='generator')

        discriminator = Sequential([
            Dense(d_hidden_size, input_shape=(2,)),
            Activation('relu'),
    #         Dense(10),
            Dense(1),
            Activation('sigmoid')
        ], name='discriminator')

        gan = Sequential([generator,discriminator])

        discriminator.compile(optimizer=Adam(learning_rate=d_learning_rate), loss='mse')
        gan.compile(optimizer=Adam(learning_rate=g_learning_rate), loss='mse')

        return gan, generator, discriminator


    def make_trainable(model, trainable):
        for layer in model.layers:
            layer.trainable = trainable
    def make_labels(size):
        return np.ones([size, 1]), np.zeros([size, 1])

    # Hyperparameters
    # Hyperparameters
    sample_size     = 10    # latent sample size
    g_hidden_size   = 6
    d_hidden_size   = 6
    g_learning_rate = 0.0001  # learning rate for the generator
    d_learning_rate = 0.0001   # learning rate for the discriminator

    epochs          = 100
    batch_size      = 1     # train batch size
    eval_size       = 1     # evaluate size
    smooth          = 0.1

    # Make labels for the batch size and the test size
    y_train_real, y_train_fake = make_labels(batch_size)
    y_eval_real,  y_eval_fake  = make_labels(eval_size)


    gan, generator, discriminator = make_GAN(
        sample_size,
        g_hidden_size,
        d_hidden_size,
        g_learning_rate,
        d_learning_rate)

    losses = []
    for e in range(epochs):
        for i in range(len(baddata.values)//batch_size):
            # Real data (minority class)
            X_batch_real = baddata.values[i*batch_size:(i+1)*batch_size]

            # Latent samples
            latent_samples = make_latent_samples(batch_size, sample_size)
            # Fake data (on minibatches)
            X_batch_fake = generator.predict_on_batch(latent_samples)

            # Train the discriminator
            make_trainable(discriminator, True)
            discriminator.train_on_batch(X_batch_real, y_train_real * (1 - smooth))
            discriminator.train_on_batch(X_batch_fake, y_train_fake)

            # Train the generator (the discriminator is fixed)
            make_trainable(discriminator, False)
            gan.train_on_batch(latent_samples, y_train_real)

        # Evaluate
        X_eval_real = baddata.values[np.random.choice(len(baddata.values), eval_size, replace=False)]

        latent_samples = make_latent_samples(eval_size, sample_size)
        X_eval_fake = generator.predict_on_batch(latent_samples)

        d_loss  = discriminator.test_on_batch(X_eval_real, y_eval_real)
        d_loss += discriminator.test_on_batch(X_eval_fake, y_eval_fake)
        g_loss  = gan.test_on_batch(latent_samples, y_eval_real)

        losses.append((d_loss, g_loss))

        print("Epoch: {:>3}/{} Discriminator Loss: {:>6.4f} Generator Loss: {:>6.4f}".format(
            e+1, epochs, d_loss, g_loss))
        
    # Generate data to fill baddata
    gen_num=500####意思要生成1000个数据
    latent_samples = make_latent_samples(gen_num, sample_size) 
    generated_data = generator.predict(latent_samples)#####重新生成虚拟样本
    generated_data = pd.DataFrame(generated_data)####

    return generated_data