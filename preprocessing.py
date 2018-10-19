#!/usr/bin/env python
# coding: utf-8

# # Preprocessing of data

# In[29]:


import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

pathDynamicData = "Mobility/dynamische\ Verkehrsdaten/FFM_DZG_180701"


# ## Dynamic countrates
# As a first step, preprocess the dynamic data by loading an additional xml-file to identify the location corresponding to the coles. Further a new data frame is created, containing the latitude, longitude for each cole. 

# In[43]:


def prepareDynData(pathAndFilename,
                   pathToXML="./Mobility/dynamische Verkehrsdaten/Statische Detektordaten.xml"):
    """
    Read and preprocess data from dynamic countrates.
        Input: 
            pathAndFilename - Path to file containing the data
            pathToXML - Path to xml file
        Returns: 
            df - data frame of the filtered data
            dfLocation - data frame of the filtered data (ID as index)
    """
    # Open data as data frame
    df = pd.read_csv(pathAndFilename, encoding='latin1', low_memory=False, sep="\t")
    df = df.fillna(0) # NaN corresponds to 0 count rates
    df = df[df["Number"] != "########"]

    # Open xml file
    xmlTree = ET.parse(pathToXML)
    root = xmlTree.getroot()

    # Get from records the identification ID and map these to location 
    IDList = []
    lat = {}
    lon = {}

    for record in root.findall(".//{http://datex2.eu/schema/2/2_0}measurementSiteRecord"):
        identification = record.findall(".//{http://datex2.eu/schema/2/2_0}measurementSiteIdentification");
        if len(identification) is not 1:
            print("More IDs per site. Take first one.")

        coordinates = record.find(".//{http://datex2.eu/schema/2/2_0}pointCoordinates")

        #ID.append(identification[0].text)
        ID = identification[0].text.split("[")[0]
        IDList.append(int(ID))
        lat[int(ID)] = float(coordinates.find(".//{http://datex2.eu/schema/2/2_0}latitude").text)
        lon[int(ID)] = float(coordinates.find(".//{http://datex2.eu/schema/2/2_0}longitude").text)

    # Only consider those coles, where location information is available
    df = df[df["ElemUID"].isin(IDList)]

    # Create data frame with location info
    dfLocation = pd.DataFrame()
    dfLocation["ElemUID"] = df["ElemUID"]
    dfLocation = dfLocation.drop_duplicates("ElemUID")
    
    dfLocation["Lat"] = dfLocation["ElemUID"]
    dfLocation["Lon"] = dfLocation["ElemUID"]
    
    dfLocation.set_index("ElemUID", inplace=True)
  
    dfLocation = dfLocation.replace({"Lat": lat})
    dfLocation = dfLocation.replace({"Lon": lon})
    
    return df, dfLocation


# ## Run the preprocessing and save data (TODO)

# In[51]:


file = "Mobility/dynamische Verkehrsdaten/FFM_DZG_180701/FFM_DEZ_180701.csv"
df, dfLocation = prepareDynData(file)


# In[ ]:




