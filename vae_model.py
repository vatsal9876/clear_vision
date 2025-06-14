# vae_model.py
import tensorflow as tf
from tensorflow.keras import layers, Model
from keras.saving import register_keras_serializable

latent_dim = 128

def build_encoder(input_shape=(64, 64, 3)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(inputs)
    x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    return tf.keras.Model(inputs, [z_mean, z_log_var], name='encoder')

def build_decoder():
    decoder_input = layers.Input(shape=(latent_dim,))
    x = layers.Dense(16*16*64, activation='relu')(decoder_input)
    x = layers.Reshape((16, 16, 64))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)
    output = layers.Conv2D(3, 3, activation='sigmoid', padding='same')(x)
    return Model(decoder_input, output, name='decoder')

def latent_sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

@register_keras_serializable()
class VAEModel(Model):
    def __init__(self, encoder, decoder):
        super(VAEModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = latent_sampling([z_mean, z_log_var])
        return self.decoder(z)
