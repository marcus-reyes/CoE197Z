# autoenc
import keras

from keras.models import Model

from keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose, Activation, Dropout, BatchNormalization

from keras import backend as K

from keras import optimizers
from keras.optimizers import Adam, SGD
import numpy as np
import skimage
###Code from https://github.com/baudm/ee298z
from keras.datasets import mnist
from hw2.benchmark_keras import test_model
import matplotlib.pyplot as plt
from PIL import Image
(x_train, y_train), (x_test, y_test) = mnist.load_data()
def corrupt_mnist_img(rng, img, value):
    """Corrupt a single MNIST image.
    Note that the image itself is MODIFIED.
    :param rng: instance of numpy.random.RandomState
    :param img: image to modify. ndarray or compatible
    :param value: pixel value to use for corrupting the image
    :return: modified image
    """
    # Choose square size
    s = rng.randint(7, 15)
    # Choose top-left corner position
    x = rng.randint(0, 29 - s)
    y = rng.randint(0, 29 - s)
    # Draw square
    img[..., y:y + s, x:x + s] = value
    # Return object for convenience
    return img
def corrupt_mnist_copy(x_train, value=255, seed=0):
    """Create a corrupted copy of the MNIST dataset
    :param x_train: ndarray of images. Shape: (N, ..., H, W)
    :param value: pixel value to use for corrupting the image
    :param seed: seed to use for the random number generator
    :return: ndarray of corrupted images
    :return: ndarray of corrupted images
    """
    rng = np.random.RandomState(seed)
    corrupted_x_train = x_train.copy()
    for img in corrupted_x_train:
        corrupt_mnist_img(rng, img, value)
    return corrupted_x_train
x_train_corrupted = corrupt_mnist_copy(x_train)
x_test_corrupted = corrupt_mnist_copy(x_test)
###End Code Chunk

# Scale to [0, 1]
x_train = x_train.astype('float32') / 255.
x_train_corrupted = x_train_corrupted.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_test_corrupted = x_test_corrupted.astype('float32') / 255.
print(x_train_corrupted[1].shape)
# plt.imshow(x_train_corrupted[1])
# plt.show()


filters = 64
strides = 2
kernel_size = 3
activation = 'relu'
dropout = 0
latent_dim = 1000 #Arbitrary
#Based on https://keras.io/examples/mnist_denoising_autoencoder/

#Encoder
encoder_input = Input(shape = (28,28,1))
x = encoder_input
for i in range(2):
    x = Conv2D(filters = filters*(i+1),
                kernel_size = kernel_size,
                activation = activation,
                strides = strides,
                padding = "same")(x)
    x = BatchNormalization()(x)
pre_flatten_shape = K.int_shape(x)

print(pre_flatten_shape)
x = Flatten()(x)
print(K.int_shape(x))
x_latent = Dense(latent_dim)(x)
encoder = Model(encoder_input, x_latent)

#Decoder
decoder_input = Input(shape = (latent_dim,))
x = decoder_input

#Get the shape just before Conv2D
x = Dense(pre_flatten_shape[1]*pre_flatten_shape[2]*pre_flatten_shape[3])(decoder_input)
x = Reshape((pre_flatten_shape[1],pre_flatten_shape[2],pre_flatten_shape[3]))(x)
#reverse the Conv2D
print("PreConv2DTranspose", K.int_shape(x))
for i in range(2):
    
    x = Conv2DTranspose(filters = filters*(i+1),
                kernel_size = kernel_size,
                activation = activation,
                strides = strides,
                padding = "same")(x)
    x = BatchNormalization()(x)

x = Conv2DTranspose(filters = 1,
            kernel_size = kernel_size,
            strides = 1,
            #activation = activation why does adding this prevent training. Explore in the future
            padding = "same")(x)
decoder_output = Activation('sigmoid')(x)

decoder = Model(decoder_input, decoder_output, name = 'decoder')

autoenc = Model(encoder_input, decoder(encoder(encoder_input)))
encoder.summary()
decoder.summary()
autoenc.summary()



#
x_train = np.reshape(x_train, [-1,28,28,1])
x_train_corrupted = np.reshape(x_train_corrupted, [-1,28,28,1])

x_test = np.reshape(x_test, [-1,28,28,1])
x_test_corrupted = np.reshape(x_test_corrupted, [-1,28,28,1])


adam = optimizers.Adam(lr=0.003, beta_1=0.9, beta_2=0.999, amsgrad=False)
autoenc.compile(loss = 'mean_squared_error',optimizer = adam)

pltx_train = np.reshape(x_train, [-1,28,28])
pltx_train_corrupted = np.reshape(x_train_corrupted, [-1,28,28])

# plt.figure(1)
# plt.subplot(131)
# plt.imshow(pltx_train_corrupted[1])
# plt.subplot(132)
# plt.imshow(pltx_train[1])
# plt.show()

autoenc.fit(x_train_corrupted, x_train,
            validation_data = (x_test_corrupted, x_test),
            epochs = 15,
            batch_size = 128)
            
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Scale to [0, 1]
x_test = x_test.astype(K.floatx()) / 255.

# model is your Keras model
# DO NOT convert y_test, i.e. don't use keras.utils.to_categorical()
test_model(autoenc, x_test, y_test, batch_size=100)
autoenc.save_weights('autoencmodel100epochs.h5')
###Chunk
x_test = np.reshape(x_test, [-1,28,28,1])
x_test_corrupted = np.reshape(x_test_corrupted, [-1,28,28,1])

x_decoded = autoenc.predict(x_test_corrupted)
x_decoded = x_decoded.astype('float32')
x_test = np.reshape(x_test, [-1,28,28])
x_test_corrupted = np.reshape(x_test_corrupted, [-1,28,28])
x_decoded = np.reshape(x_decoded, [-1,28,28])
plt.figure(1)
plt.subplot(331)
plt.imshow(x_test[1],cmap = 'gray', vmin = 0, vmax = 1)
plt.subplot(332)
plt.imshow(x_test_corrupted[1],cmap = 'gray', vmin = 0, vmax = 1)
plt.subplot(333)
plt.imshow(x_decoded[1],cmap = 'gray', vmin = 0, vmax = 1)

plt.subplot(334)
plt.imshow(x_test[4],cmap = 'gray', vmin = 0, vmax = 1)
plt.subplot(335)
plt.imshow(x_test_corrupted[4],cmap = 'gray', vmin = 0, vmax = 1)
plt.subplot(336)
plt.imshow(x_decoded[4],cmap = 'gray', vmin = 0, vmax = 1)

plt.subplot(337)
plt.imshow(x_test[5],cmap = 'gray', vmin = 0, vmax = 1)
plt.subplot(338)
plt.imshow(x_test_corrupted[5],cmap = 'gray', vmin = 0, vmax = 1)
plt.subplot(339)
plt.imshow(x_decoded[5],cmap = 'gray', vmin = 0, vmax = 1)

plt.show()