import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
import logging
import multiprocessing
import time
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("MR_PROJECT_ROOT", SCRIPT_DIR.parent))
DATA_DIR = os.environ.get("MR_RADIOMICS_DATA_DIR", os.path.join(PROJECT_ROOT, "data", "radiomics_nifti"))
OUTPUT_FILE = os.environ.get("MR_RADIOMICS_OUTPUT_FILE", os.path.join(PROJECT_ROOT, "outputs", "radiomics_features_raw.csv"))
LOG_FILE = os.environ.get("MR_RADIOMICS_LOG_FILE", os.path.join(PROJECT_ROOT, "outputs", "radiomics_extraction.log"))
PARAMS_FILE = os.environ.get("MR_RADIOMICS_PARAMS_FILE", str(PROJECT_ROOT / "configs" / "radiomics.yaml"))
N_JOBS = int(os.environ.get("MR_RADIOMICS_N_JOBS", "10"))

Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

# Setup Logging
# Suppress PyRadiomics info/warnings to console
radiomics.logger.setLevel(logging.ERROR)
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def get_extractor():
    return featureextractor.RadiomicsFeatureExtractor(PARAMS_FILE)

def extract_features(patient_id):
    try:
        patient_dir = os.path.join(DATA_DIR, patient_id)
        phases = ['AP', 'PVP', 'DP']
        
        patient_features = {'MRI_id': patient_id}
        extractor = get_extractor()
        
        has_data = False
        for phase in phases:
            image_path = os.path.join(patient_dir, f"{phase}.nii.gz")
            mask_path = os.path.join(patient_dir, f"{phase}_mask.nii.gz")
            
            if not os.path.exists(image_path) or not os.path.exists(mask_path):
                logging.error(f"Missing {phase} image or mask for {patient_id}")
                return None
                
            try:
                result = extractor.execute(image_path, mask_path)
                for key, val in result.items():
                    if not key.startswith("diagnostics_"):
                        patient_features[f"{phase}_{key}"] = val
                has_data = True
            except Exception as e:
                logging.error(f"Error extracting {phase} for {patient_id}: {e}")
                return None
        
        if has_data:
            return patient_features
        else:
            return None
    except Exception as e:
        logging.error(f"Critical error for {patient_id}: {e}")
        return None

def main():
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    patients = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    # Check for existing file to resume
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            df_exist = pd.read_csv(OUTPUT_FILE)
            if 'MRI_id' in df_exist.columns:
                processed_ids = set(df_exist['MRI_id'].astype(str))
                print(f"Resuming... Found {len(processed_ids)} already processed.")
        except:
            print("Error reading existing file, starting over.")
            
    patients_to_process = [p for p in patients if p not in processed_ids]
    print(f"Found {len(patients)} total. Processing {len(patients_to_process)} remaining with {N_JOBS} cores...")
    
    if not patients_to_process:
        print("All patients processed.")
        return

    # Process in chunks to save incrementally
    chunk_size = 50
    for i in range(0, len(patients_to_process), chunk_size):
        chunk = patients_to_process[i:i+chunk_size]
        print(f"Processing chunk {i//chunk_size + 1} ({len(chunk)} patients)...")
        
        with multiprocessing.Pool(processes=N_JOBS) as pool:
            results = pool.map(extract_features, chunk)
        
        valid_results = [r for r in results if r is not None]
        
        if valid_results:
            df_chunk = pd.DataFrame(valid_results)
            # Append to CSV
            if not os.path.exists(OUTPUT_FILE):
                df_chunk.to_csv(OUTPUT_FILE, index=False)
            else:
                df_chunk.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
            print(f"Saved {len(valid_results)} patients to {OUTPUT_FILE}")
            
    print("Extraction complete.")

if __name__ == '__main__':
    main()
