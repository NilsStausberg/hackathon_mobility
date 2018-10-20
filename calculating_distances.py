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
    return distFrom(dfLocation.loc[point].Lat, dfLocation.loc[point].Lon
                    , sensor_lat, sensor_lon)

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