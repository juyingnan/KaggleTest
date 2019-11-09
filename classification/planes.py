import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Dropout, Flatten
from tensorflow.compat.v2.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.regularizers import l2, l1
from skimage import io
import math
import os

input_dir = r'C:\Users\bunny\Desktop\planesnet'
print(os.listdir(input_dir))

data = pd.read_json(os.path.join(input_dir, 'planesnet.json'))
data.head()

x = []
for d in data['data']:
    d = np.array(d)
    x.append(d.reshape((3, 20 * 20)).T.reshape((20, 20, 3)))
x = np.array(x)
y = np.array(data['labels'])
print(x.shape)
print(y.shape)

# splitting the data into training ans test sets
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.20)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.50)
# Normalizing the data
scalar = MinMaxScaler()
scalar.fit(x_train.reshape(x_train.shape[0], -1))

x_train = scalar.transform(x_train.reshape(x_train.shape[0], -1)).reshape(x_train.shape)
x_val = scalar.transform(x_val.reshape(x_val.shape[0], -1)).reshape(x_val.shape)
x_test = scalar.transform(x_test.reshape(x_test.shape[0], -1)).reshape(x_test.shape)

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
print(x_test.shape)
print(y_test.shape)


# creating the convolutional model

def create_cnn(data_shape):
    kernel_size = 3

    model = Sequential()

    model.add(Conv2D(16, (kernel_size), strides=(1, 1), padding='valid',
                     input_shape=(data_shape[1], data_shape[2], data_shape[3])))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #     model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(32, (kernel_size), strides=(1, 1), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #     model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64, (kernel_size), strides=(1, 1), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #     model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64, (kernel_size), strides=(1, 1), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (kernel_size), strides=(1, 1), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    #     model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    return model


cnn_model = create_cnn(x_train.shape)
print(cnn_model.summary())


# training the model
def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    #    if epoch:
    #        lrate = initial_lrate/np.sqrt(epoch)
    #    else:
    #        return initial_lrate
    return lrate


opt = Adam(lr=0.0001)
lrate = LearningRateScheduler(step_decay)
cnn_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
history = cnn_model.fit(x_train, y_train, batch_size=64, epochs=50, shuffle=True, verbose=2,
                        validation_data=(x_val, y_val), callbacks=[lrate],
                        )

y_pred = cnn_model.predict(x_test)
y_pred[y_pred > 0.5] = 1
y_pred[y_pred < 0.5] = 0
y_pred_bool = np.asarray(y_pred, dtype=bool)
target_names = ['No Plane', 'Plane']
print(classification_report(y_test, y_pred_bool, target_names=target_names))
print('Accuracy:', accuracy_score(y_test, y_pred_bool))

for i in range(len(y_pred)):
    pred = int(y_pred[i])
    print(pred)
    if pred == 0 and y_test[i] == 1:
        io.imsave(r'C:\Users\bunny\Desktop\test\01\{}.png'.format(i), x_test[i])
    if pred == 1 and y_test[i] == 0:
        io.imsave(r'C:\Users\bunny\Desktop\test\10\{}.png'.format(i), x_test[i])
    if pred == 0 and y_test[i] == 0:
        io.imsave(r'C:\Users\bunny\Desktop\test\00\{}.png'.format(i), x_test[i])
    if pred == 1 and y_test[i] == 1:
        io.imsave(r'C:\Users\bunny\Desktop\test\11\{}.png'.format(i), x_test[i])

# # plotting the learning curves.
# fig, ax = plt.subplots(1,2)
# fig.set_size_inches((15,5))
# ax[0].plot(range(1,51), history.history['loss'], c='blue', label='Training loss')
# ax[0].plot(range(1,51), history.history['val_loss'], c='red', label='Validation loss')
# ax[0].legend()
# ax[0].set_xlabel('epochs')
#
# ax[1].plot(range(1,51), history.history['acc'], c='blue', label='Training accuracy')
# ax[1].plot(range(1,51), history.history['val_acc'], c='red', label='Validation accuracy')
# ax[1].legend()
# ax[1].set_xlabel('epochs')
