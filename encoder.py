from keras.models import Sequential
from keras.layers import Conv2D, Dropout, BatchNormalization, Activation, MaxPooling2D, Reshape

def separable_block(outfilters, kernel_size, downsample, dropout, s_block, first = False):
    n_filter = outfilters / 4

    row_kernel = (kernel_size, 1)
    col_kernel = (1, kernel_size)
    
    if first:
        s_block.add(Conv2D(filters=n_filter, kernel_size=(1, 1), strides=(1, 1), padding='same', input_shape=(360, 480, 3)))
    else:
        s_block.add(Conv2D(filters=n_filter, kernel_size=(1, 1), strides=(1, 1), padding='same'))

    s_block.add(BatchNormalization())
    s_block.add(Activation('relu'))

    s_block.add(Conv2D(filters=n_filter, kernel_size=row_kernel, strides=(1, 1), padding='same'))
    s_block.add(Conv2D(filters=n_filter, kernel_size=col_kernel, strides=(1, 1), padding='same'))

    s_block.add(BatchNormalization())
    s_block.add(Activation('relu'))

    s_block.add(Conv2D(filters=outfilters, kernel_size=(1, 1), strides=(1, 1), padding='same'))

    if downsample:
        s_block.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    s_block.add(BatchNormalization())
    s_block.add(Activation('relu'))

    if dropout:
        s_block.add(Dropout(dropout))

    return s_block

def gen_encoder():
    model = Sequential()
    model = separable_block(16,  3, True,    0, model, True) # i:(360 x 480 x 3)  | o:(180 x 240 x 16)
    model = separable_block(16,  3, False, 0.2, model)       # i:(180 x 240 x 16) | o:(180 x 240 x 16) 
    model = separable_block(32,  3, False, 0.2, model)       # i:(180 x 240 x 16) | o:(180 x 240 x 32)
    model = separable_block(32,  3, False, 0.5, model)       # i:(180 x 240 x 32) | o:(180 x 240 x 32)
    model = separable_block(64,  3, True,    0, model)       # i:(180 x 240 x 32) | o:(90  x 120 x 64)
    model = separable_block(64,  3, False, 0.5, model)       # i:(90  x 120 x 64) | o:(90  x 120 x 64)
    model = separable_block(128, 3, False, 0.2, model)       # i:(90  x 120 x 64) | o:(90  x 120 x 128)
    mdoel = separable_block(128, 3, False, 0.5, model)       # i:(90  x 120 x 128)| o:(90  x 120 x 128)
    model = separable_block(256, 3, True,    0, model)       # i:(90  x 120 x 128)| o:(45  x 60  x 256)
    model = separable_block(256, 3, False, 0.5, model)       # i:(45  x 60  x 256)| o:(45  x 60  x 256)

    return model
