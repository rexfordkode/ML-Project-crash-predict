import os
import pandas as pd

def clean_crss_data(raw_dir='data/raw', processed_dir='data/processed'):
    """
    Loads raw CRSS files year by year (2016-2023), dynamically handles schema variations 
    for person types (including PER_TYP and PERSONRFNAME), merges tables cleanly, 
    drops survey weights & leakage variables, and exports a clean master dataset.
    """
    os.makedirs(processed_dir, exist_ok=True)
    all_cleaned_years = []

    for year in range(2016, 2024):
        per_path = os.path.join(raw_dir, f'person_{year}.csv')
        veh_path = os.path.join(raw_dir, f'vehicle_{year}.csv')
        acc_path = os.path.join(raw_dir, f'accident_{year}.csv')
        
        if not (os.path.exists(per_path) and os.path.exists(veh_path) and os.path.exists(acc_path)):
            print(f"Skipping year {year}: Missing one or more raw files.")
            continue
            
        print(f"Preprocessing year {year}...")
        
        # Load tables safely with string dtypes and latin-1 encoding
        person_df = pd.read_csv(per_path, encoding='latin-1', dtype=str, low_memory=False)
        vehicle_df = pd.read_csv(veh_path, encoding='latin-1', dtype=str, low_memory=False)
        accident_df = pd.read_csv(acc_path, encoding='latin-1', dtype=str, low_memory=False)
        
        # Standardize columns to uppercase
        person_df.columns = [c.upper() for c in person_df.columns]
        vehicle_df.columns = [c.upper() for c in vehicle_df.columns]
        accident_df.columns = [c.upper() for c in accident_df.columns]
        
       # Flexible check for person type across all years
        if 'PER_TYP' in person_df.columns:
            person_df['PER_TYP'] = pd.to_numeric(person_df['PER_TYP'], errors='coerce')
            # Filter for pedestrians (5) and cyclists (6)
            ped_cyc_df = person_df[person_df['PER_TYP'].isin([5, 6])]
        else:
            # Fallback if PER_TYP is named differently
            alt_col = next((c for c in person_df.columns if 'PER' in c and 'TYP' in c), None)
            if alt_col:
                person_df[alt_col] = pd.to_numeric(person_df[alt_col], errors='coerce')
                ped_cyc_df = person_df[person_df[alt_col].isin([5, 6])]
            else:
                print(f"Warning: Could not isolate pedestrian/cyclist column for {year}.")
                continue
                
        if ped_cyc_df.empty:
            print(f"No records matched for {year}.")
            continue

        # Merge tables on CASENUM using explicit suffixes to prevent column name collisions
        merged_df = ped_cyc_df.merge(vehicle_df, on='CASENUM', how='inner', suffixes=('_per', '_veh'))
        merged_df = merged_df.merge(accident_df, on='CASENUM', how='inner', suffixes=('', '_acc'))
        
        # Drop survey weights (STRATUM, WEIGHT) and post-injury leakage columns
        leakage_cols = ['STRATUM', 'WEIGHT', 'HOSP_RC', 'DEATH_TM', 'INJ_SEV']
        cols_to_drop = [col for col in leakage_cols if col in merged_df.columns]
        cleaned_df = merged_df.drop(columns=cols_to_drop)
        
        cleaned_df['YEAR'] = year
        all_cleaned_years.append(cleaned_df)

    if all_cleaned_years:
        master_df = pd.concat(all_cleaned_years, ignore_index=True)
        
        output_path = os.path.join(processed_dir, 'clean_ped_cyc_crashes.csv')
        master_df.to_csv(output_path, index=False)
        print(f"\nPreprocessing complete! Saved to {output_path} with shape: {master_df.shape}")
    else:
        print("No data was processed successfully.")

if __name__ == "__main__":
    clean_crss_data()