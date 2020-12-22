import os
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

#Load data and visualize it
raw_list_x = []
raw_list_y = []
for filename in os.listdir("data"):
    if filename.endswith(".mat"):
        str = filename[0 : -4]
        m = loadmat("data/" + filename)
        mat = np.array(m["connectivity"])
        plt.matshow(mat)
        plt.colorbar()
        plt.savefig("img/" + filename[0 : -4] + ".png")
        plt.figure()
        mat = mat / mat.max()
        raw_list_x.append(mat)
        if str[0 : 3] == "nkd":
            raw_list_y.append(0)
        else:
            raw_list_y.append(1)
data_x = np.array(raw_list_x)
data_y = np.array(raw_list_y)

from sklearn.model_selection import train_test_split

#Split data into train and test
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y)

import tensorflow as tf

#Preprocess the data
train_x = train_x.reshape(-1, 120, 120, 1)
test_x = test_x.reshape(-1, 120, 120, 1)
train_x = train_x.astype("float32")
test_x = test_x.astype("float32")
train_y = tf.keras.utils.to_categorical(train_y)
test_y = tf.keras.utils.to_categorical(test_y)

#Build model
tf.compat.v1.disable_eager_execution()
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), padding = "same", input_shape = (120, 120, 1)))
model.add(tf.keras.layers.LeakyReLU(0.1))
model.add(tf.keras.layers.MaxPool2D((2, 2), padding = "same"))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Conv2D(64, (3, 3), padding = "same"))
model.add(tf.keras.layers.LeakyReLU(0.1))
model.add(tf.keras.layers.MaxPool2D((2, 2), padding = "same"))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Conv2D(128, (3, 3), padding = "same"))
model.add(tf.keras.layers.LeakyReLU(0.1))
model.add(tf.keras.layers.MaxPool2D((2, 2), padding = "same"))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation = "linear"))
model.add(tf.keras.layers.LeakyReLU(0.1))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(2, activation = "softmax"))

#Compile the model
model.compile(loss = tf.keras.losses.categorical_crossentropy, optimizer = tf.keras.optimizers.Adam(), metrics = ["accuracy"])

#Train the model
model_train = model.fit(train_x, train_y, epochs = 50, validation_data = (test_x, test_y))

#Plot the model's history
accuracy = model_train.history["accuracy"]
validation_accuracy = model_train.history["val_accuracy"]
loss = model_train.history["loss"]
validation_loss = model_train.history["val_loss"]
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, "bo", label = "Training accuracy")
plt.plot(epochs, validation_accuracy, "b", label = "Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.savefig("plots/training_and_validation_accuracy.png")
plt.figure()
plt.plot(epochs, loss, "bo", label = "Training loss")
plt.plot(epochs, validation_loss, "b", label = "Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.savefig("plots/training_and_validation_loss.png")
plt.figure()

#Set up and preprocess the data for CAM visualization
test_nkd = np.array(loadmat("data/nkd1_1.mat")["connectivity"])
test_scz = np.array(loadmat("data/scz1_1.mat")["connectivity"])
test_nkd = test_nkd.reshape((1, 120, 120, 1))
test_scz = test_scz.reshape((1, 120, 120, 1))
test_nkd = test_nkd / test_nkd.max()
test_scz = test_scz / test_scz.max()
test_nkd = test_nkd.astype("float32")
test_scz = test_scz.astype("float32")

#Predict
pred_nkd = model.predict(test_nkd)
pred_scz = model.predict(test_scz)

import keras.backend as K

#Create heatmap for nkd
nkd_output = model.output[:, 0]
last_conv_layer = model.layers[8]
grads = K.gradients(nkd_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis = (0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([test_nkd])
for i in range(128):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.imshow(heatmap)
plt.colorbar()
plt.savefig("img/nkd.png")
plt.figure()

#Create heatmap for scz
scz_output = model.output[:, 1]
last_conv_layer = model.layers[8]
grads = K.gradients(scz_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis = (0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([test_scz])
for i in range(128):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.imshow(heatmap)
plt.colorbar()
plt.savefig("img/scz.png")