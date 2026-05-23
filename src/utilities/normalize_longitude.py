def normalize_longitude(lon):
    """
    Normalize longitude values to be within [-180, 180] range
    """
    while lon > 180:
        lon -= 360
    while lon < -180:
        lon += 360
    return lon
