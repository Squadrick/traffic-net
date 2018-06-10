import numpy as np
from keras.models import Sequential
from data_preprocess import prep_data
from encoder import gen_encoder
from decoder import gen_decoder
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

encoder = gen_encoder()
decoder = gen_decoder()
model = Sequential()
model.add(encoder)
model.add(decoder)

print "Model Created"
model.summary()

train_data, train_labels = prep_data('train')

print "Data preprocessed"
print "Training data: ", train_data.shape
print "Training Labels: ", train_labels.shape

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
print "Model compiled"

class_weights = [0.00776, 1.0, 1.38170]

# freq = pix_freq(class)/no_of_images_containing_pixels(class)
# weight = median(for all classes [freq]) / freq(class)

path = "weights/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_lr=0.00001)
tensorboard = TensorBoard(log_dir="Graph", histogram_freq=0, write_graph=True, write_images=True)
callbacks_list = [checkpoint, reduce_lr, tensorboard]

model.fit(x=train_data, y=train_labels, batch_size=6, epochs=2000, callbacks=callbacks_list, validation_split=0.2, class_weight=class_weights, shuffle=True)
