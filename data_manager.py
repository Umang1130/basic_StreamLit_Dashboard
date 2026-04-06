import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

DATA_FILE = "health_data.csv"

def get_dummy_data():
    np.random.seed(42)
    dates = [datetime.now() - timedelta(days=i) for i in range(30, -1, -1)]
    
    # Healthy baselines with some noise
    heart_rate = np.random.normal(loc=72, scale=5, size=len(dates))
    systolic_bp = np.random.normal(loc=120, scale=8, size=len(dates))
    diastolic_bp = np.random.normal(loc=80, scale=5, size=len(dates))
    sleep_hours = np.random.normal(loc=7.5, scale=1, size=len(dates))
    steps = np.random.normal(loc=8000, scale=2000, size=len(dates))
    
    # Introduce an anomaly in the past to ensure model sees it
    heart_rate[10] = 110
    systolic_bp[10] = 160
    
    df = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "heart_rate": heart_rate,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "sleep_hours": sleep_hours,
        "steps": steps
    })
    return df

def load_data():
    if not os.path.exists(DATA_FILE):
        df = get_dummy_data()
        df.to_csv(DATA_FILE, index=False)
        return df
    
    return pd.read_csv(DATA_FILE)

def add_entry(entry_dict):
    df = load_data()
    # check if date exists, if so overwrite, else append
    date_str = entry_dict['date']
    
    new_row = pd.DataFrame([entry_dict])
    if date_str in df['date'].values:
        # replace the existing row
        for col in entry_dict.keys():
            df.loc[df['date'] == date_str, col] = new_row.iloc[0][col]
    else:
        # append
        df = pd.concat([df, new_row], ignore_index=True)
        
    # Sort by date
    df = df.sort_values(by="date")
    df.to_csv(DATA_FILE, index=False)
    return df
