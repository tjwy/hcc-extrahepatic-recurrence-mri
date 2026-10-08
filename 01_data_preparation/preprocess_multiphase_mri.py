import os
import sys
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom, label
import warnings
from tqdm import tqdm

# Filter warnings
warnings.filterwarnings('ignore')

# Config constants (Same as config_phase2.py)
TARGET_SPACING = (1.0, 1.0, 1.0)
ROI_MARGIN = 10 
TARGET_SIZE = (32, 128, 128)
ROI_MODE_TO_MARGIN = {
    'tumor_plus_10mm': 10,
    'tumor_only': 0,
    'tumor_plus_5mm': 5,
}
SEQUENCES = {
    'AP':  {'img': 'AP.nii.gz',  'mask': 'AP_mask.nii.gz'},
    'PVP': {'img': 'PVP.nii.gz', 'mask': 'PVP_mask.nii.gz'},
    'DP':  {'img': 'DP.nii.gz',  'mask': 'DP_mask.nii.gz'}
}

# ==========================================
# Helper Functions (Adapted from 01_preprocessing_pipeline.py)
# ==========================================

def get_flexible_path(base_dir, filename):
    """Find file allowing .nii / .nii.gz variations and case-insensitivity."""
    if not os.path.exists(base_dir): return None
    
    # 1. Try exact match and simple extensions
    exts = ['', '.gz'] if not filename.endswith('.gz') else ['', '']
    basename = filename.replace('.gz', '') if filename.endswith('.gz') else filename
    basename = basename.replace('.nii', '') if basename.endswith('.nii') else basename
    
    # Priority candidates
    candidates = [
        filename,
        basename + ".nii.gz",
        basename + ".nii",
        basename.upper() + ".nii.gz",
        basename.lower() + ".nii.gz",
        basename.upper() + ".nii",
        basename.lower() + ".nii"
    ]
    
    for cand in candidates:
        path = os.path.join(base_dir, cand)
        if os.path.exists(path): return path
            
    # 2. Search in directory (Case insensitive and substring)
    try:
        files = os.listdir(base_dir)
        # Search for something containing "AP" and ".nii" etc.
        search_key = basename.upper()
        for f in files:
            f_upper = f.upper()
            if search_key in f_upper and (".NII" in f_upper):
                # Extra check for mask vs image
                if "MASK" in search_key:
                    if "MASK" in f_upper: return os.path.join(base_dir, f)
                else:
                    if "MASK" not in f_upper: return os.path.join(base_dir, f)
    except:
        pass
            
    return None

def load_nifti(path):
    if path is None or not os.path.exists(path): return None, None
    import SimpleITK as sitk
    try:
        # SimpleITK is very robust for "fake" extensions and various NIfTI versions
        itk_img = sitk.ReadImage(path)
        data = sitk.GetArrayFromImage(itk_img).astype(np.float32)
        # SimpleITK arrays are (Z, Y, X), NIfTI convention is (X, Y, Z)
        data = data.transpose(2, 1, 0)
        
        # We still need the affine for nibabel-based processing downstream.
        # Try to get it via nibabel if possible (for canonical affine)
        try:
            img = nib.load(path)
            affine = img.affine
        except:
            # Fallback: Construct affine from sitk metadata if nibabel fails
            spacing = itk_img.GetSpacing()
            origin = itk_img.GetOrigin()
            direction = itk_img.GetDirection()
            # Construction of 4x4 affine from sitk is complex but we can approximate 
            # for most diagonal cases.
            affine = np.eye(4)
            affine[0, 0] = spacing[0] * direction[0]
            affine[1, 1] = spacing[1] * direction[4]
            affine[2, 2] = spacing[2] * direction[8]
            affine[:3, 3] = origin
            
        return data, affine
    except Exception as e:
        # print(f"Error loading {path}: {e}")
        return None, None

def keep_largest_connected_component(mask):
    """Keep only the largest connected component of a binary mask."""
    if mask is None: return None
    labeled_mask, num_features = label(mask)
    if num_features <= 1:
        return mask
    
    max_volume = 0
    best_component = 0
    
    # Check volume for each component
    for i in range(1, num_features + 1):
        volume = np.sum(labeled_mask == i)
        if volume > max_volume:
            max_volume = volume
            best_component = i
            
    return (labeled_mask == best_component).astype(np.float32)

def process_mask_labels(mask_data):
    """
    Extract tumor mask.
    Logic:
    1. If Label 2 exists (standard for tumor in this dataset), select it.
    2. Else, select the label with the SMALLEST volume (heuristic for single label scenarios).
    3. Finally, keep only the LARGEST connected component to avoid outlier noise or multifocality issues.
    """
    if mask_data is None: return None
    unique_labels = np.unique(mask_data)
    unique_labels = unique_labels[unique_labels != 0]
    
    if len(unique_labels) == 0: return mask_data
    
    target_mask = None
    
    # Priority 1: Check for Label 2
    if 2 in unique_labels:
        target_mask = (mask_data == 2).astype(np.float32)
    
    # Priority 2: Fallback to original heuristic (Single label or weird labels)
    elif len(unique_labels) == 1:
        target_mask = (mask_data > 0).astype(np.float32)
    else:
        # Heuristic: Choose smallest volume (assuming liver > tumor)
        best_label = None
        min_volume = float('inf')
        for l in unique_labels:
            vol = np.sum(mask_data == l)
            if vol < min_volume:
                min_volume = vol
                best_label = l
        target_mask = (mask_data == best_label).astype(np.float32)
        
    # Post-processing: Keep Largest Connected Component
    final_mask = keep_largest_connected_component(target_mask)
    
    return final_mask

def resample_image_volume(data, affine, target_spacing=TARGET_SPACING, is_mask=False):
    if data is None: return None, None
    
    # Logic for mask extraction before resampling
    if is_mask:
        data = process_mask_labels(data)
        
    current_spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    zoom_factors = current_spacing / target_spacing
    
    order = 0 if is_mask else 1
    
    data_resampled = zoom(data, zoom_factors, order=order)
    
    scale_mat = np.diag([1/zoom_factors[0], 1/zoom_factors[1], 1/zoom_factors[2], 1.0])
    new_affine = np.dot(affine, scale_mat)
    
    return data_resampled, new_affine

def crop_roi(data, mask, affine, margin=ROI_MARGIN):
    if mask is None or data is None: return None, None
    
    coords = np.argwhere(mask > 0)
    if len(coords) == 0: return None, None
        
    x_min, y_min, z_min = coords.min(axis=0)
    x_max, y_max, z_max = coords.max(axis=0)
    
    # Add margin
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    z_min = max(0, z_min - margin)
    
    x_max = min(data.shape[0], x_max + margin)
    y_max = min(data.shape[1], y_max + margin)
    z_max = min(data.shape[2], z_max + margin)
    
    cropped_data = data[x_min:x_max, y_min:y_max, z_min:z_max]
    
    # Update affine origin
    new_origin = np.dot(affine, np.array([x_min, y_min, z_min, 1]))
    new_affine = np.copy(affine)
    new_affine[:3, 3] = new_origin[:3]
    
    return cropped_data, new_affine

def resize_volume(data, target_size=TARGET_SIZE):
    if data is None: return None
    # 3D NIfTI usually loads as (X, Y, Z) / (Width, Height, Slices) in LPS orientation.
    # The models expect (Depth, Height, Width) where Depth is Slices.
    # To map Z->32, Y->128, X->128, we transpose (X, Y, Z) to (Z, Y, X).
    data_transposed = data.transpose(2, 1, 0)
    zoom_factors = np.array(target_size) / np.array(data_transposed.shape)
    data_resized = zoom(data_transposed, zoom_factors, order=1)
    return data_resized

def normalize_and_clip(data):
    """Percentile clipping + Z-score normalization."""
    if data is None: return None
    
    # 1. Percentile Clipping (Robust against metal artifacts/outliers)
    p05 = np.percentile(data, 0.5)
    p995 = np.percentile(data, 99.5)
    data_clipped = np.clip(data, p05, p995)
    
    # 2. Z-score
    mean = np.mean(data_clipped)
    std = np.std(data_clipped)
    if std == 0: return data_clipped
    return (data_clipped - mean) / std

def save_nifti(data, affine, path):
    if data is None: return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, path)

# ==========================================
# Main Processing Function
# ==========================================

def process_case(mri_id, case_path, dl_root, rad_root, roi_margin=ROI_MARGIN):
    """
    Process one case using specific path.
    Returns: status string ("Success", "Fail", etc.)
    """
    
    # 1. Output dirs
    case_rad_dir = os.path.join(rad_root, mri_id)
    case_dl_path = os.path.join(dl_root, f"{mri_id}.npy")
    
    # Check if already done
    if os.path.exists(case_dl_path):
        return "Skipped"

    # 2. Process per sequence
    patient_arrays_dl = {}
    
    for seq_name, file_info in SEQUENCES.items():
        img_name = file_info['img']
        mask_name = file_info['mask']
        
        img_path = get_flexible_path(case_path, img_name)
        mask_path = get_flexible_path(case_path, mask_name)
        
        if not img_path or not mask_path:
            return f"Missing {seq_name}"
            
        # Load
        img_data, img_aff = load_nifti(img_path)
        mask_data, mask_aff = load_nifti(mask_path)
        
        if img_data is None or mask_data is None:
            return f"Load Fail {seq_name}"
            
        # Do not silently overwrite segmentation geometry.
        if mask_data.shape != img_data.shape or not np.allclose(mask_aff, img_aff, atol=1e-4):
            return f"Image/mask geometry mismatch {seq_name}"
            
        # Resample
        img_res, img_aff_res = resample_image_volume(img_data, img_aff, is_mask=False)
        mask_res, mask_aff_res = resample_image_volume(mask_data, mask_aff, is_mask=True)
        
        # Crop (Ensure shapes match first)
        min_shape = np.minimum(img_res.shape, mask_res.shape)
        img_res = img_res[:min_shape[0], :min_shape[1], :min_shape[2]]
        mask_res = mask_res[:min_shape[0], :min_shape[1], :min_shape[2]]
        
        # Check alignment logic - assuming aligned. If mask is from different space, this might fail.
        # But for this dataset, we assume they are registered.
        
        try:
            img_crop, crop_aff = crop_roi(img_res, mask_res, img_aff_res, margin=roi_margin)
            mask_crop, _ = crop_roi(mask_res, mask_res, mask_aff_res, margin=roi_margin)
        except Exception as e:
            return f"Crop Error {seq_name}: {e}"
        
        if img_crop is None or img_crop.size == 0:
            return f"Empty Crop {seq_name}"
            
        # --- Save Radiomics (Cropped, Original Spacing/Values preserved implicitly via Resample? No, Resampled 1mm) ---
        # Note: The 'radiomics' files here are resampled to 1x1x1mm and cropped.
        save_nifti(img_crop, crop_aff, os.path.join(case_rad_dir, f"{seq_name}.nii.gz"))
        save_nifti(mask_crop, crop_aff, os.path.join(case_rad_dir, f"{seq_name}_mask.nii.gz"))
        
        # --- Deep Learning (Resize & Norm) ---
        img_resize = resize_volume(img_crop)
        img_norm = normalize_and_clip(img_resize)
        patient_arrays_dl[seq_name] = img_norm
        
    # Stack
    try:
        stack = np.stack([patient_arrays_dl['AP'], patient_arrays_dl['PVP'], patient_arrays_dl['DP']], axis=0)
        np.save(case_dl_path, stack)
        return "Success"
    except KeyError as e:
        return f"Missing Sequence for Stack: {e}"
    except Exception as e:
        return f"Stack Error: {e}"

def parse_args():
    parser = argparse.ArgumentParser(description="Batch preprocessing for DL and radiomics inputs with explicit ROI strategies.")
    parser.add_argument(
        '--roi-mode',
        default='tumor_plus_5mm',
        choices=sorted(ROI_MODE_TO_MARGIN.keys()),
        help='ROI crop strategy. tumor_plus_5mm is the manuscript workflow.',
    )
    parser.add_argument(
        '--output-dl-dir',
        default=None,
        help='Optional absolute or project-root-relative output directory for DL .npy files.',
    )
    parser.add_argument(
        '--output-rad-dir',
        default=None,
        help='Optional absolute or project-root-relative output directory for radiomics NIfTI files.',
    )
    parser.add_argument(
        '--list-file',
        default=None,
        help='CSV containing MRI_id and Path. Defaults to data/processing_list.csv.',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Optional number of cases to process for smoke testing.',
    )
    return parser.parse_args()


def resolve_output_dir(base_dir, arg_value, default_name):
    if arg_value is None or str(arg_value).strip() == '':
        return os.path.join(base_dir, default_name)
    if os.path.isabs(arg_value):
        return arg_value
    return os.path.join(base_dir, arg_value)


def main():
    args = parse_args()
    # Repository root is the parent of this stage directory.)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    roi_margin = ROI_MODE_TO_MARGIN[args.roi_mode]
    
    # Inputs
    LIST_FILE = args.list_file or os.path.join(BASE_DIR, 'data', 'processing_list.csv')
    OUTPUT_DL = resolve_output_dir(BASE_DIR, args.output_dl_dir, 'data/dl_tumor_plus5mm')
    OUTPUT_RAD = resolve_output_dir(BASE_DIR, args.output_rad_dir, 'data/radiomics_nifti')
    
    if not os.path.exists(LIST_FILE):
        raise FileNotFoundError(f"List file not found at: {LIST_FILE}")
        
    df = pd.read_csv(LIST_FILE)
    if not {'MRI_id', 'Path'}.issubset(df.columns):
        raise ValueError("Processing list requires MRI_id and Path columns")
    if df['MRI_id'].isna().any() or df['MRI_id'].duplicated().any():
        raise ValueError("Processing-list identifiers must be present and unique")
    if args.limit is not None:
        df = df.head(args.limit).copy()
    print(f"Loaded {len(df)} cases to process.")
    print(f"ROI mode: {args.roi_mode} | margin={roi_margin} mm")
    print(f"DL output: {OUTPUT_DL}")
    print(f"Radiomics output: {OUTPUT_RAD}")
    
    os.makedirs(OUTPUT_DL, exist_ok=True)
    os.makedirs(OUTPUT_RAD, exist_ok=True)
    
    results = []
    
    pbar = tqdm(df.iterrows(), total=len(df))
    for idx, row in pbar:
        uid = str(row['MRI_id'])
        path = os.path.expanduser(str(row['Path']))
        
        status = process_case(uid, path, OUTPUT_DL, OUTPUT_RAD, roi_margin=roi_margin)
        
        pbar.set_description(f"Processing {uid}")
        
        if status not in ["Success", "Skipped"]:
            results.append({'MRI_id': uid, 'Status': status})
            
    print("\nProcessing Complete.")
    if results:
        err_df = pd.DataFrame(results)
        err_file = os.path.join(OUTPUT_DL, 'preprocessing_errors.csv')
        err_df.to_csv(err_file, index=False)
        print(f"Errors saved to {err_file}")
        print(f"Total Errors: {len(err_df)}")
        raise RuntimeError("Preprocessing failed for one or more cases; see the error report")
    else:
        print("No errors reported!")

if __name__ == "__main__":
    main()
