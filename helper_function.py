import pandas as pd
import numpy as np
import re
import datetime

# https://en.wikipedia.org/wiki/Haversine_formula
def distFrom(lat1, lng1, lat2, lng2):
    earthRadius = 6371000 #meters
    dLat = np.radians(lat2-lat1)
    dLng = np.radians(lng2-lng1)
    a = np.sin(dLat/2) ** 2 + np.cos(np.radians(lat1)) * np.cos(
        np.radians(lat2)) * np.sin(dLng/2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    dist = earthRadius * c
    return dist

def compute_dist_to_Sensor(point, sensor_lat, sensor_lon):
    dfLocation = pd.read_csv('Mobility/dfLocation.csv', sep="\t", index_col='ElemUID')
    return distFrom(dfLocation.loc[int(point)].Lat, dfLocation.loc[int(point)].Lon, sensor_lat, sensor_lon)

def drop_loops_far_away(df, max_dist=3000, latSensor = 50.12565556, lonSensor = 8.69305556):
    """
    Drops data (i.e. #cars) of all induction loops which are more than {max_dist} meters away
        Input:
            df - data frame with all 719 inducion loops in the 2nd to the 720th column
        Returns:
            df_nearest_loops - data frame that only contains the nearest loops
    """
    col_to_drop = [df.columns[i] for i in range(1,720) if
              (compute_dist_to_Sensor(df.columns[i], latSensor, lonSensor) > max_dist)]
    df_nearest_loops = df.drop(col_to_drop, axis=1)
    return df_nearest_loops

def reduce_cars_in_distance(df, percetage = 1, dist=100, lat = 50.12565556, lon = 8.69305556):
    """
    Drops data (i.e. #cars) of all induction loops which are more than {max_dist} meters away
        Input:
            df - data frame with all 719 inducion loops in the 2nd to the 720th column
            dist - distance in meters
        Returns:
            df_nearest_loops - data frame that only contains the nearest loops
    """
    print(df.columns)
    col_to_reduce = [df.columns[i] for i in range(1,696) if (compute_dist_to_Sensor(df.columns[i], lat, lon) < dist)]
    for column in col_to_reduce:
        df[column] = df[column].multiply(percetage)

    return df

def select_day(df, month="09", day="30"):
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    dateStart = "2018-"+month+"-"+day+" 00:30:00"
    dateEnd = "2018-"+month+"-"+day+" 23:30:00"

    index_min = df[df["Timestamp"] == datetime.datetime.strptime(dateStart, "%Y-%m-%d %H:%M:%S")].index.values[0]
    index_max = df[df["Timestamp"] == datetime.datetime.strptime(dateEnd, "%Y-%m-%d %H:%M:%S")].index.values[0] + 1

    return df[index_min:index_max]

def add_datepart(df, fldname, drop=True, time=False):
    """add_datepart converts a column of df from a datetime64 to many columns containing
    the information from the date. This applies changes inplace.

    Parameters:
    -----------
    df: A pandas data frame. df gain several new columns.
    fldname: A string that is the name of the date column you wish to expand.
        If it is not a datetime64 series, it will be converted to one with pd.to_datetime.
    drop: If true then the original date column will be removed.
    time: If true time features: Hour, Minute, Second will be added.

    Examples:
    ---------

    >>> df = pd.DataFrame({ 'A' : pd.to_datetime(['3/11/2000', '3/12/2000', '3/13/2000'], infer_datetime_format=False) })
    >>> df

        A
    0   2000-03-11
    1   2000-03-12
    2   2000-03-13

    >>> add_datepart(df, 'A')
    >>> df

        AYear AMonth AWeek ADay ADayofweek ADayofyear AIs_month_end AIs_month_start AIs_quarter_end AIs_quarter_start AIs_year_end AIs_year_start AElapsed
    0   2000  3      10    11   5          71         False         False           False           False             False        False          952732800
    1   2000  3      10    12   6          72         False         False           False           False             False        False          952819200
    2   2000  3      11    13   0          73         False         False           False           False             False        False          952905600
    """
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Dayofweek']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)

def is_date(x): return np.issubdtype(x.dtype, np.datetime64)

