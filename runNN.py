from finalNN import NN
import pandas as pd
import numpy as np
from helperFunctions import add_datepart

dfN = pd.read_csv("./Data_prep3.csv", delimiter=",")
dfN = dfN.fillna(0)
add_datepart(dfN, "Timestamp", drop=False, time=False)
#print(dfN.columns)
dfN = dfN.drop(["Unnamed: 0", "Timestamp","TimestampElapsed"], axis=1);
dfN.to_csv("Data_prep_weekdays.csv", sep="\t")
#dfN = dfN.dropna()

targetDioxid = dfN[["Stickstoffmonoxid (NO)[µg/m³]"]];
dataTemp = dfN.drop(["Stickstoffdioxid (NO2)[µg/m³]","Stickstoffmonoxid (NO)[µg/m³]"], axis=1);
npFeatures = dataTemp.values
npTargets = targetDioxid.values

feature = npFeatures[100:101]
target = npTargets[100:101]

neuralNetwork = NN(loadPathModel="./modelNN.json", loadPathWeights="./modelNN.h5")
prediction = neuralNetwork.predict(feature)
