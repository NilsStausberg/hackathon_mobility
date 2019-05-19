
# coding: utf-8

# # Preprocessing of data

# In[1]:


import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import datetime
import matplotlib.pyplot as plt

pathDynamicData = "Mobility/dynamische_Verkehrsdaten/"


# ## Dynamic countrates
# As a first step, preprocess the dynamic data by loading an additional xml-file to identify the location corresponding to the induction loops. Further a new data frame is created, containing the latitude, longitude for each cole.

# In[2]:


def prepareDynData(pathAndFilename,
                   pathToXML=f"{pathDynamicData}Statische_Detektordaten.xml"):
    """
    Read and preprocess data from dynamic countrates.
        Input:
            pathAndFilename - Path to file containing the data
            pathToXML - Path to xml file
        Returns:
            df_filtered - data frame of the filtered data
            dfLocation - data frame of the filtered data (ID as index)
    """
    # Open data as data frame
    df = pd.read_csv(pathAndFilename, sep="\t", encoding='latin1', low_memory=False
                     , parse_dates=["DaySecFrom(UTC)", "DaySecTo(UTC)"])
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


# We have given the data every 1-2 minutes for every induction loop. However, we only want a granularity of 30 minutes, because the air polution data is only given in this granularity. The result will be a data frame containing for each timestamp (granularity: 30 min)
#  the amount of cars at each induction loop, respectively.

# In[3]:


def round_up_date_to_half_hours(dt):
    if ((dt.minute == 30 or dt.minute == 0) and dt.second == 0):
        return str(dt)
    if (dt.minute < 30):
        return datetime.datetime(dt.year, dt.month, dt.day, dt.hour, 30).isoformat(' ')
    if (dt.hour < 23):
        return datetime.datetime(dt.year, dt.month, dt.day, dt.hour + 1).isoformat(' ')
    if (getattr(dt, 'is_year_end')):
        return datetime.datetime(dt.year+1, 1, 1).isoformat(' ')
    if (getattr(dt, 'is_month_end')):
        return datetime.datetime(dt.year, dt.month + 1, 1).isoformat(' ')
    else:
        return datetime.datetime(dt.year, dt.month, dt.day + 1).isoformat(' ')

def get_amount_cars_per_30_min(df):
    """
    We only want a granularity of 30 minutes, because the air polution data
    is only given in this granularity.
        Input:
            df - data frame of the filtered data
        Returns:
            df_amount_cars - data frame containing for each timestamp (granularity: 30 min)
                             the amount of cars at each induction loop, respectively
    """

    df.ElemUID = pd.to_numeric(df.ElemUID, downcast='integer')
    # NaN corresponds to 0 count rates
    df.Number = pd.to_numeric(df.Number).fillna(0).astype(int)

    # Number of cars is given in cars/hour. We want to have cars/period.
    df['Period'] = df['DaySecTo(UTC)'] - df['DaySecFrom(UTC)']
    df.Period = (df.Period.dt.seconds / 60).astype(int)
    df.Number = (df.Number * df.Period / 60).astype(int)

    # Add 1 hour, because date in UTC+0 and air polution data is in MEZ.
    # TODO: summer/winter time
    df['Timestamp'] = df['DaySecTo(UTC)'] + datetime.timedelta(hours=1)
    df = df.drop(['ElemName', 'Kind', 'DaySecFrom(UTC)', 'DaySecTo(UTC)', 'Period'], axis=1)

    df.Timestamp = df.Timestamp.apply(lambda dt: round_up_date_to_half_hours(dt))
    df = df.groupby(['ElemUID', 'Timestamp'])['Number'].sum().reset_index(name = 'Total_Cars')

    return df.pivot(index='Timestamp', columns='ElemUID', values='Total_Cars')


# ## Run the preprocessing

# In[4]:


file = f"{pathDynamicData}FFM_DZG_180701/FFM_DEZ_180701.csv"


# In[6]:


df_before.ElemUID = pd.to_numeric(df_before.ElemUID, downcast='integer').apply(str)
df_before.describe(include='all')


# Note that in the original dataframe we about 3.5 M data points for 2315 different UIDs.

# In[7]:


df_filtered, dfLocation = prepareDynData(file)


# In[8]:


print(df_filtered.shape)
df_filtered.head()


# In[9]:


print(dfLocation.shape)
dfLocation.head()


# In[10]:


df_amount_cars = get_amount_cars_per_30_min(df_filtered)


# In[11]:


print(df_amount_cars.shape)
df_amount_cars.head()


# After preprocessing, we are left with 719 induction loops (for about 70% we have no location given).
