from __future__ import absolute_import, division, print_function
import pathlib
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# download Auto MPG dataset
dataset_path = keras.utils.get_file("auto-mpg.data",
                                    "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

# import the dataset using pandas
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment='\t', sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy();

# printing the dataset
# print(dataset.tail());

# cleaning the data
dataset = dataset.dropna();


# print(dataset.isna().sum());

# convert the categorical "Origin" column to a one-hot.
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0
# print(dataset.tail())

# split the data into training and testing datasets.
trainingData = dataset.sample(frac=0.8, random_state=0)
testingData = dataset.drop(trainingData.index)

# plot the data
#sns.pairplot(trainingData[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
# plt.show()

# look at the statistics
train_stats = trainingData.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
# print(train_stats)

# separate the label (which we are trying to predict) from the rest of the features.
train_labels = trainingData.pop('MPG')
test_labels = testingData.pop('MPG')


# create method to normalize data passed into it used the mean and standard deviation.
def normalize(data):
    return (data - train_stats['mean']) / train_stats['std']


# normalize the training and testing data and store the normalized data in variables.
normTrainingData = normalize(trainingData)
normTestingData = normalize(testingData)


# new method which can be called to build the model
def buildModel():
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(trainingData.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model


# build the model and store it in a variable.
model = buildModel()

# inspect the model.
#print(model.summary())

# test the model on a batch of 10 examples from the training data.
example = normTrainingData[:10]
exampleResult = model.predict(example)
# print(exampleResult)

# create a dot printer to display the training progress by printing a dot for each completed epoch.
class DotPrinter(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if (epoch % 100) == 0:
            print('')
        print('.', end='')


# store number of epochs in a variable
EPOCHS = 1000

# model is trained for 1000 epochs and the training and validation accuracy is stored in a variable.
history = model.fit(
    normTrainingData, train_labels,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    # callbacks=[DotPrinter()]
)

# visualize models training progress by printing stats stored in the history variable.
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

#print()
#print(hist.tail()

# create method to plot stored stats in graph.
def plotHistory(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    # plt.figure()
    # plt.xlabel('EPOCH')
    # plt.ylabel('Mean absolute error [MPG]')
    # plt.plot(hist['epoch'], hist['mean_absolute_error'], label = 'Train error')
    # plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label = 'Val error')
    # plt.legend()
    # plt.ylim([0,5])
    #
    # plt.figure()
    # plt.xlabel('EPOCH')
    # plt.ylabel('Mean square error [$MPG^2$]')
    # plt.plot(hist['epoch'], hist['mean_squared_error'], label = 'Train error')
    # plt.plot(hist['epoch'], hist['val_mean_squared_error'], label = 'Val error')
    # plt.legend()
    # plt.ylim([0,20])

# plot the stats stored in the history variable in a graph.
# plotHistory(history)
# plt.show()

# update model.fit to stop training when improvement of the validation score stops.
model = buildModel()
earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(normTrainingData, train_labels, epochs=EPOCHS, validation_split= 0.2, verbose=0, callbacks=[earlyStop, DotPrinter()])
plotHistory(history)
# plt.show()
print()

# use test set to see how well the model generalizes.
loss, meanAbsoluteError, meanSquaredError = model.evaluate(normTestingData, test_labels, verbose = 0)
print("Testing set mean absolute error: {:5.2f} MPG".format(meanAbsoluteError))

predictions = model.predict(normTestingData).flatten()
plt.scatter(test_labels, predictions)
plt.xlabel('True values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()][1])
_ = plt.plot([-100, 100], [-100, 100])

plt.show()
