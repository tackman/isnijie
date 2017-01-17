import fnmatch
import os
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
from datetime import datetime as dt

train_dir = "/home/takuma/datasets/nijigen_train"
validation_dir = "/home/takuma/datasets/nijigen_validation"

tstr = dt.now()


img_channels  = 3
img_rows = 61
img_cols = 61
batch_size = 32


train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        color_mode='rgb',
        target_size=(img_rows, img_cols),
        batch_size=batch_size)

validation_generator = train_datagen.flow_from_directory(
        validation_dir,
        color_mode='rgb',
        target_size=(img_rows, img_cols),
        batch_size=batch_size)




model = Sequential()

model.add(Convolution2D(21, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(21, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))


model.add(Convolution2D(42, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(42, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint('./prod_models/cnn_model{epoch:02d}-loss{loss:.2f}-acc{acc:.2f}-vloss{val_loss:.4f}-vacc{val_acc:.4f}.hdf5')



json_string = model.to_json()
open('nijigen_model_arrch-' + tstr.strftime('%Y-%m-%d-%H:%M') + '.json', 'w').write(json_string)



model.fit_generator(
        train_generator,
        samples_per_epoch= batch_size*32*3,
        nb_epoch=100,
        validation_data=validation_generator,
        nb_val_samples=1000,
        callbacks=[model_checkpoint])


model.save_weights('nijigen_model_weights-' + tstr.strftime('%Y-%m-%d-%H:%M') + '.h5')
