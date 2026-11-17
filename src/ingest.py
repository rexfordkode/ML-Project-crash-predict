# Data Ingestion Module
import pandas as pd
import os

def load_raw_crss_data(year, raw_dir='data/raw'):
    """
    Loads raw accident, vehicle, and person CSV files for a given year 
    with proper encoding and data types.
    """
    per_path = os.path.join(raw_dir, f'person_{year}.csv')
    veh_path = os.path.join(raw_dir, f'vehicle_{year}.csv')
    acc_path = os.path.join(raw_dir, f'accident_{year}.csv')
    
    if not (os.path.exists(per_path) and os.path.exists(veh_path) and os.path.exists(acc_path)):
        print(f"Warning: Missing files for year {year} in {raw_dir}")
        return None, None, None
        
    print(f"Ingesting raw data for year {year}...")
    
    # Load with dtype=str to preserve codes and IDs safely
    person_df = pd.read_csv(per_path, encoding='latin-1', dtype=str, low_memory=False)
    vehicle_df = pd.read_csv(veh_path, encoding='latin-1', dtype=str, low_memory=False)
    accident_df = pd.read_csv(acc_path, encoding='latin-1', dtype=str, low_memory=False)
    
    # Standardize column names to uppercase
    person_df.columns = [c.upper() for c in person_df.columns]
    vehicle_df.columns = [c.upper() for c in vehicle_df.columns]
    accident_df.columns = [c.upper() for c in accident_df.columns]
    
    return person_df, vehicle_df, accident_df

if __name__ == "__main__":
    # Quick test execution
    p, v, a = load_raw_crss_data(2018)
    if p is not None:
        print(f"Successfully ingested 2016 data. Person rows: {len(p)}")