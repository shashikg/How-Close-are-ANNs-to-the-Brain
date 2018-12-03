import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import RMSprop
from keras.models import load_model

batch_size = 32
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = []

for i in range(4):
    m = load_model('saved_model/' + 'cnn'+str(i)+'.h5')
    print('cnn '+str(i))
    m.summary()
    s = m.evaluate(x_test, y_test, verbose=0)

    print('Test accuracy:', s[1])

    model.append(m)

p_tr = []
p_te = []

for i, m in enumerate(model):
    p = m.predict(x_train)
    pt = m.predict(x_test)
    p_tr.append(p)
    p_te.append(pt)

print(len(p_te[1]))

p_train = np.zeros((60000,40))
p_test = np.zeros((10000,40))
for i, p in enumerate(p_tr):
    p_train[:,10*i:10*(i+1)] = p

for i, p in enumerate(p_te):
    p_test[:,10*i:10*(i+1)] = p

print(p_train.shape, p_test.shape)

modele = load_model('saved_model/cnnensemble.h5')

score = modele.evaluate(p_test, y_test, verbose=0)

print('NN Based Ensambled Model')
print('Test loss:', score[0])
print('Test accuracy:', score[1])
