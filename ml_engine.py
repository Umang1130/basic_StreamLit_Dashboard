import pandas as pd
from sklearn.ensemble import IsolationForest

def analyze_vitals(df):
    """
    Fits an IsolationForest on the historical dataframe and returns the same dataframe
    with an 'anomaly' column (-1 for anomaly, 1 for normal).
    """
    if len(df) < 5:
        # Not enough data for meaningful ML
        df['anomaly'] = 1 
        return df
    
    # Only use numeric features
    features = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'sleep_hours', 'steps']
    X = df[features].fillna(df[features].mean())
    
    # Isolation Forest for Anomaly Detection
    # Contamination is the expected proportion of outliers.
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X)
    
    # Predict (-1 is anomaly, 1 is normal)
    predictions = model.predict(X)
    
    # Adding results back to the dataframe
    df_result = df.copy()
    df_result['anomaly'] = predictions
    return df_result
