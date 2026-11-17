import os
import pandas as pd
from ingest import load_raw_crss_data

def clean_crss_data(raw_dir='data/raw', processed_dir='data/processed'):
    """
    Cleans and merges CRSS data from 2016 to 2023, filtering for 
    pedestrians/cyclists and removing leakage variables and survey weights.
    """
    os.makedirs(processed_dir, exist_ok=True)
    all_cleaned_years = []

    # Process years 2016 to 2023 as specified in your project proposal
    for year in range(2016, 2024):
        person_df, vehicle_df, accident_df = load_raw_crss_data(year, raw_dir)
        
        if person_df is None:
            continue
            
        print(f"Preprocessing year {year}...")
        
        # 1. Identify and filter for pedestrians and cyclists (Codes 5 and 6 typically)
        possible_col_names = ['PER_TYP', 'PER_TYPE', 'PERSON_TYPE', 'PTYPE']
        per_typ_col = next((col for col in possible_col_names if col in person_df.columns), None)
        
        if not per_typ_col:
            print(f"Warning: Person type column not found for {year}.")
            continue
            
        person_df[per_typ_col] = pd.to_numeric(person_df[per_typ_col], errors='coerce')
        ped_cyc_df = person_df[person_df[per_typ_col].isin([5, 6])]
        
        if ped_cyc_df.empty:
            print(f"No pedestrian/cyclist records found for {year}.")
            continue

        # 2. Merge tables on CASENUM
        merged_df = ped_cyc_df.merge(vehicle_df, on='CASENUM', how='inner', suffixes=('_per', '_veh'))
        merged_df = merged_df.merge(accident_df, on='CASENUM', how='inner', suffixes=('', '_acc'))

        # 3. Drop survey weights (STRATUM, WEIGHT) and post-injury leakage columns[cite: 1]
        leakage_cols = ['STRATUM', 'WEIGHT', 'HOSP_RC', 'DEATH_TM', 'INJ_SEV']
        cols_to_drop = [col for col in leakage_cols if col in merged_df.columns]
        cleaned_df = merged_df.drop(columns=cols_to_drop)
        
        cleaned_df['YEAR'] = year
        all_cleaned_years.append(cleaned_df)

    if all_cleaned_years:
        master_df = pd.concat(all_cleaned_years, ignore_index=True)
        
        # Save final processed dataset
        output_path = os.path.join(processed_dir, 'clean_ped_cyc_crashes.csv')
        master_df.to_csv(output_path, index=False)
        print(f"\nPreprocessing complete! Saved to {output_path} with shape: {master_df.shape}")
    else:
        print("No data was processed.")

if __name__ == "__main__":
    clean_crss_data()