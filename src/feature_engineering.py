import pandas as pd


def create_features(df):

    """
    Create engineered features from raw sensor data.
    """

    # Identify sensor columns
    sensor_cols = [col for col in df.columns if "sensor" in col]

    # Rolling Mean
    for sensor in sensor_cols:
        df[f"{sensor}_rolling_mean"] = (
            df.groupby("engine_id")[sensor]
            .rolling(window=5)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # Rolling Standard Deviation
    for sensor in sensor_cols:
        df[f"{sensor}_rolling_std"] = (
            df.groupby("engine_id")[sensor]
            .rolling(window=5)
            .std()
            .reset_index(level=0, drop=True)
        )

    # Delta (cycle difference)
    for sensor in sensor_cols:
        df[f"{sensor}_delta"] = df.groupby("engine_id")[sensor].diff()

    # Fill missing values
    df = df.fillna(0)

    return df

def select_features(df):

    keep_cols = [
        'op1','op2','op3',
        'sensor_2','sensor_3','sensor_4','sensor_6','sensor_7',
        'sensor_8','sensor_9','sensor_11','sensor_12','sensor_13',
        'sensor_14','sensor_15','sensor_17','sensor_20','sensor_21'
    ]

    return df[keep_cols]
