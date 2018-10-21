#!/usr/bin/env python
# coding: utf-8

# # Visualization

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd

from helper_function import drop_loops_far_away

import tilemapbase
tilemapbase.start_logging()
tilemapbase.init(create=True)
t = tilemapbase.tiles.OSM


# Function which takes a data frame containing the locations of each IT as a data frame. A plot is created showing the station on a map.

# In[5]:


def plotLocation(df, df2 = None):
    # Sensor locations
    latSensor = [50.12565556, 50.10290556, 50.12691389]
    lonSensor = [8.69305556, 8.54222222, 8.74861111]

    # Create a basic map using OpenStreetMap
    centerPoint = (8.69305556, 50.12565556)
    degree_range = 0.1
    extent = tilemapbase.Extent.from_lonlat(centerPoint[0] - 1.6*degree_range, centerPoint[0] + degree_range,centerPoint[1] - degree_range, centerPoint[1] + degree_range)
    extent = extent.to_aspect(1.0)

    plotter = tilemapbase.Plotter(extent, t, width=600)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    plotter.plot(ax, t)

    plotxSensor = []
    plotySensor = []
    for i in range(len(latSensor)):
        setTEMP = (lonSensor[i], latSensor[i])
        x, y = tilemapbase.project(*setTEMP)

        plotxSensor.append(x)
        plotySensor.append(y)

    lonTEMP = df["Lon"]
    latTEMP = df["Lat"]

    plotx = []
    ploty = []
    for i in range(len(lonTEMP)):
        setTEMP = (lonTEMP.iloc[i] ,latTEMP.iloc[i])
        x, y = tilemapbase.project(*setTEMP)

        plotx.append(x)
        ploty.append(y)

    ax.scatter(plotx, ploty, label="Loops")
    ax.scatter(plotxSensor, plotySensor, label="Sensor")

    plotx_add = []
    ploty_add = []
    if df2 is not None:
        colList = [df2.columns[i] for i in range(1,df2.columns.get_loc("Stickstoffmonoxid (NO)[µg/m³]"))]
        for column in colList:
            location = df[df["ElemUID"] == int(column)]
            setTEMP = (location["Lon"].values[0], location["Lat"].values[0])
            x, y = tilemapbase.project(*setTEMP)

            plotx_add.append(x)
            ploty_add.append(y)

        ax.scatter(plotx_add, ploty_add, c="green", label="Reduced Loops")

    plt.legend()
    plt.show()

def plotReducedLoops(data, R, coordinates):
    file = "Mobility/dfLocation.csv"
    dfLocation = pd.read_csv(file, delimiter="\t")

    dfDropedLoops = drop_loops_far_away(data, R, coordinates[0], coordinates[1])

    plotLocation(dfLocation, dfDropedLoops)
