import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, TimeDistributed, GRU
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import RMSprop
from keras.models import load_model

batch_size = 32
num_classes = 10
epochs = 15

img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols)
input_shape = (img_rows, img_cols)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

x_trainT = np.transpose(x_train, axes=(0,2,1))
x_testT = np.transpose(x_test, axes=(0,2,1))

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Row Model
print('Row 1 Model')
model_row = load_model('saved_model/lstm_row.h5')
score_row = model_row.evaluate(x_test, y_test, verbose=0)
model_row.summary()
print('Row Test accuracy:', score_row[1])

print('Row 2 Model')
model_row2 = load_model('saved_model/gru_row.h5')
score_row2 = model_row2.evaluate(x_test, y_test, verbose=0)
model_row2.summary()
print('Row Test accuracy:', score_row2[1])

# Col Model
print('Col 1 Model')
model_col = load_model('saved_model/lstm_col.h5')
score_col = model_col.evaluate(x_testT, y_test, verbose=0)
model_col.summary()
print('Col Test accuracy:', score_col[1])

print('Col 2 Model')
model_col2 = load_model('saved_model/gru_col.h5')
score_col2 = model_col2.evaluate(x_testT, y_test, verbose=0)
model_col2.summary()
print('Col2 Test accuracy:', score_col2[1])




# Ensambled using both row and column Model
pred1 = model_row.predict(x_train)
pred2 = model_col.predict(x_trainT)
pred3 = model_col2.predict(x_trainT)
pred4 = model_row2.predict(x_train)



pred1t = model_row.predict(x_test)
pred4t = model_row2.predict(x_test)
pred2t = model_col.predict(x_testT)
pred3t = model_col2.predict(x_testT)


p_train = np.zeros((60000,40))
# print(pred1[1],pred2[1],pred3[1])
p_train = np.concatenate([pred1, pred2, pred3, pred4],  axis=1)

p_test = np.zeros((60000,40))
# print(pred1[1],pred2[1],pred3[1])
p_test = np.concatenate([pred1t, pred2t, pred3t, pred4t],  axis=1)

print(p_train.shape)

batch_size = 32
num_classes = 10
epochs = 5
# print(p_train[1])
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(40,)))
# model.add(Dropout(0.1))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.05))
model.add(Dense(num_classes, activation='softmax'))



model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(p_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=0,
                    validation_data=(p_test, y_test))

score = model.evaluate(p_test, y_test, verbose=0)

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

print('NN Based Ensambled Model')
model.summary()
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print(history.history['val_acc'])
print(history.history['acc'])

model.save('rnn_ensamble.h5')




# predf = (pred1+pred2+pred3)/3.0
# y_pred = np.argmax(pred, axis=1)
# yt = np.argmax(y_train, axis=1)
# c = np.count_nonzero(yt-y_pred)
# print('Ensambled Accuracy:', 1-c/yt.shape[0])
