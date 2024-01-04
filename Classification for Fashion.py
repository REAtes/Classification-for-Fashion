import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


## Data Exploration

# the 10 classes decoding is as follows:
# 0 => T-shirt/top
# 1 => Trouser
# 2 => Pullover
# 3 => Dress
# 4 => Coat
# 5 => Sandal
# 6 => Shirt
# 7 => Sneaker
# 8 => Bag
# 9 => Ankle boot

train_df = pd.read_csv(".datasets/fashion-mnist_train.csv", sep=",")
test_df = pd.read_csv(".datasets/fashion-mnist_test.csv", sep=",")


## Data Visualization

### Create training and testing arrays
training = np.array(train_df, dtype='float32')
testing = np.array(test_df, dtype='float32')

i = random.randint(1, 60000)  # select any random index from 1 to 60,000
plt.imshow(training[i, 1:].reshape((28, 28)))  # reshape and plot the image
plt.imshow(training[i, 1:].reshape((28, 28)), cmap='gray')  # recolored and plot the image

### Let's view more images in a grid format
W_grid = 15
L_grid = 15
fig, axes = plt.subplots(L_grid, W_grid, figsize=(17, 17))
axes = axes.ravel()  # flatten the 15 x 15 matrix into 225 array
n_training = len(training)  # get the length of the training dataset
#### Select a random number from 0 to n_training
for i in np.arange(0, W_grid * L_grid):  # create evenly spaces variables
    # Select a random number
    index = np.random.randint(0, n_training)
    # read and display an image with the selected index
    axes[i].imshow(training[index, 1:].reshape((28, 28)))
    axes[i].set_title(training[index, 0], fontsize=8)
    axes[i].axis('off')
plt.subplots_adjust(hspace=0.4)


## Training The Model

X_train = training[:, 1:]/255  # normalization
y_train = training[:, 0]

X_test = testing[:, 1:]/255  # normalization
y_test = testing[:, 0]

X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2, random_state=123)

### * unpack the tuple
X_train = X_train.reshape(X_train.shape[0], *(28, 28, 1))
X_test = X_test.reshape(X_test.shape[0], *(28, 28, 1))
X_validate = X_validate.reshape(X_validate.shape[0], *(28, 28, 1))

### 32 filters
cnn_model32 = Sequential()
cnn_model32.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
cnn_model32.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model32.add(Dropout(0.25))

cnn_model32.add(Flatten())
cnn_model32.add(Dense(units=32, activation='relu'))
cnn_model32.add(Dense(units=10, activation='sigmoid'))

cnn_model32.compile(loss='sparse_categorical_crossentropy',
                    optimizer=Adam(lr=0.001),
                    metrics=['accuracy'])

history32 = cnn_model32.fit(X_train,
                            y_train,
                            batch_size=512,
                            epochs=50,
                            verbose=1,
                            validation_data=(X_validate, y_validate))
#### Results:
# loss: 0.1421 - accuracy: 0.9482 - val_loss: 0.2511 - val_accuracy: 0.9165

### 64 filters
cnn_model64 = Sequential()
cnn_model64.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1), activation='relu'))
cnn_model64.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model64.add(Dropout(0.25))

cnn_model64.add(Flatten())
cnn_model64.add(Dense(units=32, activation='relu'))
cnn_model64.add(Dense(units=10, activation='sigmoid'))

cnn_model64.compile(loss='sparse_categorical_crossentropy',
                    optimizer=Adam(lr=0.001),
                    metrics=['accuracy'])

history64 = cnn_model64.fit(X_train,
                            y_train,
                            batch_size=512,
                            epochs=50,
                            verbose=1,
                            validation_data=(X_validate, y_validate))
#### Results:
# loss: 0.1114 - accuracy: 0.9586 - val_loss: 0.2710 - val_accuracy: 0.9119


### Lets make a new model with MaxPooling
new_model = Sequential()
new_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
new_model.add(MaxPooling2D(pool_size=(2, 2)))

new_model.add(Flatten())
new_model.add(Dense(128, activation='relu'))
new_model.add(Dense(10, activation='softmax'))

new_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

### Convert the labels to one-hot coding
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_validate = to_categorical(y_validate)

new_model.fit(X_train,
              y_train,
              batch_size=512,
              epochs=50,
              validation_data=(X_validate, y_validate))
#### Results:
# loss: 0.0156 - accuracy: 0.9963 - val_loss: 0.4081 - val_accuracy: 0.9152


## Hyperparameter Tuning

from keras.optimizers import RMSprop
optimizer = RMSprop(lr=0.001)
new_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
new_model.fit(X_train,
              y_train,
              batch_size=32,
              epochs=50,
              validation_data=(X_validate, y_validate))
#### Results
# loss: 0.0048 - accuracy: 0.9986 - val_loss: 1.3150 - val_accuracy: 0.9067



predicted_probabilities = new_model.predict(X_test)
predicted_classes = np.argmax(predicted_probabilities, axis=-1)

L = 5
W = 5
fig, axes = plt.subplots(L, W, figsize=(12, 12))
axes = axes.ravel()
for i in np.arange(0, L * W):
    axes[i].imshow(X_test[i].reshape(28, 28))
    axes[i].set_title("Prediction Class = {:0.1f}\n True Class = {:0.1f}".format(predicted_classes[i], y_test[i]))
    axes[i].axis('off')
plt.subplots_adjust(wspace=0.5)

### Sum the diagonal element to get the total true correct values
num_classes = 10
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_test, predicted_classes, target_names=target_names))