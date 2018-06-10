from keras.layers import UpSampling2D, Conv2DTranspose, BatchNormalization, Activation, Reshape
from keras.models import Sequential

def decoder_block(outfilters, kernel_size, upsample, d_block, first = False):
    n_filter = outfilters / 4

    row_kernel = (kernel_size, 1)
    col_kernel = (1, kernel_size)

    if first:
        d_block.add(Conv2DTranspose(filters=n_filter, kernel_size=(1, 1), strides=(1, 1), padding='same',input_shape=(45, 60, 256)))
    else:
        d_block.add(Conv2DTranspose(filters=n_filter, kernel_size=(1, 1), strides=(1, 1), padding='same'))

    d_block.add(BatchNormalization())
    d_block.add(Activation('relu'))

    d_block.add(Conv2DTranspose(filters=n_filter, kernel_size=row_kernel, strides=(1, 1), padding='same'))
    d_block.add(Conv2DTranspose(filters=n_filter, kernel_size=col_kernel, strides=(1, 1), padding='same'))

    d_block.add(BatchNormalization())
    d_block.add(Activation('relu'))

    d_block.add(Conv2DTranspose(filters=outfilters, kernel_size=(1, 1), strides=(1, 1), padding='same'))
    d_block.add(BatchNormalization())
    d_block.add(Activation('relu'))

    if upsample:
        d_block.add(UpSampling2D(size=(2, 2)))

    return d_block

def gen_decoder():
    model = Sequential()
    model = decoder_block(128, 3, True,  model, True) # i:(45  x 60  x 256)| o:(90  x 120 x 128)
    model = decoder_block(64,  3, True,  model)       # i:(90  x 120 x 128)| o:(180 x 240 x 64)
    model = decoder_block(32,  3, True,  model)       # i:(180 x 240 x 64) | o:(360 x 480 x 32)
    model = decoder_block(16,  3, True,  model)       # i:(360 x 480 x 32) | o:(360 x 480 x 16)
    model = decoder_block(3,   1, False, model)       # i:(360 x 480 x 16) | o:(360 x 480 x 3)

    model.add(Activation('softmax'))
    return model
