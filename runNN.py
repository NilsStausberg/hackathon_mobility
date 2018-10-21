from finalNN import NN
import pandas as pd
import numpy as np
from helper_function import reduce_cars_in_distance
from helper_function import select_day
from helper_function import add_datepart
import matplotlib.pyplot as plt

df = pd.read_csv("./Data_prep3.csv", delimiter=",")
df = df.fillna(0)

add_datepart(df, "Timestamp", drop=False, time=False)

df = df.drop(["Unnamed: 0","TimestampElapsed"], axis=1);
#df = df.drop(["Unnamed: 0"], axis=1);

neuralNetwork = NN(loadPathModel="./modelNN.json", loadPathWeights="./modelNN.h5")

dfDayTEMP = select_day(df, month="09", day="30")

dfDayTEMP = dfDayTEMP.drop("Timestamp", axis=1);

dfDay = dfDayTEMP.copy()
dfDayReduced = reduce_cars_in_distance(dfDayTEMP, 0.5, 10000000)


targetDay = dfDay[["Stickstoffmonoxid (NO)[µg/m³]"]];
targetDay = targetDay.reset_index()
targetDayReduced = dfDayReduced[["Stickstoffmonoxid (NO)[µg/m³]"]];
dataDay = dfDay.drop(["Stickstoffdioxid (NO2)[µg/m³]","Stickstoffmonoxid (NO)[µg/m³]"], axis=1);
dataDayReduced = dfDayReduced.drop(["Stickstoffdioxid (NO2)[µg/m³]","Stickstoffmonoxid (NO)[µg/m³]"], axis=1);

npFeaturesDay = dataDay.values
npFeaturesDayReduced = dataDayReduced.values
npTargetsDay = targetDay.values
npTargetsDayReduced = targetDayReduced.values

predictionDay = neuralNetwork.predict(npFeaturesDay)
predictionDayReduced = neuralNetwork.predict(npFeaturesDayReduced)


plt.plot(predictionDay, label="100%")
plt.plot(predictionDayReduced, label="50%")
plt.plot(targetDay["Stickstoffmonoxid (NO)[µg/m³]"], label="Target")
plt.legend()
plt.show()
