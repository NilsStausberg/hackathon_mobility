import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import model_from_json

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

class NN():

    npTrainingFeatures = None
    npTrainingTargets = None
    npValidationFeatures = None
    npValidationTargets = None
    npTestFeatures = None
    npTestTargets = None
    model = None

    def __init__(self, data=None, loadPathModel=None, loadPathWeights=None):
        if data is not None:
            trainingData, validationData, testData = self.__splitData(data, 60, 1, 20, True)

            self.npTrainingFeatures, self.npTrainingTargets = self.__convertData(trainingData)
            self.npValidationFeatures, self.npValidationTargets = self.__convertData(validationData)
            self.npTestFeatures, self.npTestTargets = self.__convertData(testData)
        elif loadPathModel is not None and loadPathWeights is not None:
            self.model = self.loadModel(loadPathModel, loadPathWeights)
        else:
            print("ERROR!")
            assert(0)


    def __splitData(self, data, ratioTraining=60, ratioValidation=30, ratioTest=10, shuffle=True):
        """
        Splits the data (panda data frame) into smaller data frames for training,
        validation and testing.
        """
        ratioTraining = ratioTraining / 100
        ratioValidation = ratioValidation / 100
        ratioTest = ratioTest / 100

        if shuffle:
           data = data.reindex(np.random.permutation(data.index))

        N = data.shape[0]
        testData = data.head(int(N * ratioTest))
        dataTemp = data.tail(int(N * (1 - ratioTest)))

        trainingData = dataTemp.head(int(N * ratioTraining))
        validationData = dataTemp.tail(int(N * ratioValidation))

        return trainingData, validationData, testData

    def __convertData(self, data):
        # Convert data to numpy format + seperate features and self.targets
        #targetDioxid = data[["Stickstoffdioxid (NO2)[µg/m³]","Stickstoffmonoxid (NO)[µg/m³]"]];
        targetDioxid = data[["Stickstoffmonoxid (NO)[µg/m³]"]];
        dataTemp = data.drop(["Stickstoffdioxid (NO2)[µg/m³]","Stickstoffmonoxid (NO)[µg/m³]"], axis=1);
        npFeatures = dataTemp.values
        npTargets = targetDioxid.values

        return npFeatures, npTargets;

    def build_model(self, train_data):
        model = keras.Sequential([keras.layers.Dense(100, activation=tf.nn.relu,
                                  input_shape=(train_data.shape[1],)),
                                  keras.layers.Dropout(0.2),
                                  keras.layers.Dense(50, activation=tf.nn.relu),
                                  keras.layers.Dropout(0.2),
                                  keras.layers.Dense(10, activation=tf.nn.relu),
                                  keras.layers.Dense(1)])


        model.compile(loss='mse', optimizer="adam", metrics=['mae'])

        return model

    def plot_history(self, history):
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error')
        plt.plot(history.epoch, np.array(history.history['mean_absolute_error']), label='Train Loss')
        plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']), label = 'Val loss')
        plt.legend()

    def train(self):
        train_data = self.npTrainingFeatures
        train_labels = self.npTrainingTargets

        test_data = self.npTestFeatures
        test_labels = self.npTestTargets

        self.model = self.build_model(train_data)
        self.model.summary()
        EPOCHS = 100

        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
        history = self.model.fit(train_data, train_labels, epochs=EPOCHS,
                            validation_split=0.2, verbose=0,
                            callbacks=[PrintDot()])

        self.plot_history(history)

        [loss, mae] = self.model.evaluate(test_data, test_labels, verbose=0)

        test_predictions = self.model.predict(test_data).flatten()

        model_json = self.model.to_json()
        with open("modelNN.json", "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        self.model.save_weights("modelNN.h5")
        print("Saved model to disk")

        plt.figure()
        plt.scatter(test_labels, test_predictions)
        plt.xlabel('True Values / MSE')
        plt.ylabel('Predictions / MSE')

        print("Testing set Mean Abs Error: ${:7.2f}".format(mae))

        print("Plot prediction and target")
        plt.figure()
        plt.plot(test_labels, label="Target")
        plt.plot(test_predictions, label="Prediction")
        plt.legend()

        print("Plot error")
       # plt.figure()
        error = test_predictions - test_labels
        print(test_predictions.shape)
        print(test_labels.shape)
       # plt.hist(error, bins = 50)
       # plt.xlabel("Prediction Error")
       # plt.ylabel("Count")

        plt.show()

    def loadModel(self, path="modelNN.json", pathWeights="modelNN.h5"):
        json_file = open(path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        loaded_model.load_weights(pathWeights)

        return loaded_model

    def predict(self, data):
        predictions = self.model.predict(data).flatten()

        return predictions
