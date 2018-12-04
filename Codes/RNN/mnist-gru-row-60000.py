from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, TimeDistributed, GRU
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

batch_size = 32
num_classes = 10
epochs = 15

img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print (str(x_train.shape) + str(y_train.shape) )

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols)
input_shape = (img_rows, img_cols)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# x = np.transpose(x_train, axes=(0,2,1))
# xt = np.transpose(x_test, axes=(0,2,1))
# x_train = np.concatenate([x, x_train], axis=1)
# x_test = np.concatenate([xt, x_test], axis=1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_train.shape[1:])

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(TimeDistributed(Dropout(0), input_shape=input_shape))
model.add(LSTM(128))
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.50))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
          
score = model.evaluate(x_test, y_test, verbose=0)

plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()
plt.savefig("model_accuracy.png")

plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()
plt.savefig("model_loss.png")

model.summary()
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print(history.history['val_acc'])
print(history.history['acc'])

model.save('gru_row.h5')