import os
import shutil
from totalsegmentator.python_api import totalsegmentator
import SimpleITK as sitk

def convert_dicom_to_nifti(dicom_dir, output_dir, scan_name=None):
    """
    Convert DICOM series to NIfTI format with robust slice ordering.
    
    Handles cases where GetGDCMSeriesFileNames returns incorrect order
    by explicitly sorting slices by ImagePositionPatient Z-coordinate.
    """
    reader = sitk.ImageSeriesReader()
    
    # Get all series IDs in the directory
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    
    if len(series_ids) == 0:
        raise RuntimeError(f"No DICOM series found in {dicom_dir}")
    
    print(f"\nDICOM Image Info:")
    print(f"  Found {len(series_ids)} series in directory")
    
    # If multiple series, try to find the largest one (usually the CT volume)
    if len(series_ids) > 1:
        print(f"  Series IDs: {series_ids}")
        best_series = None
        max_slices = 0
        for sid in series_ids:
            files = reader.GetGDCMSeriesFileNames(dicom_dir, sid)
            if len(files) > max_slices:
                max_slices = len(files)
                best_series = sid
        series_id = best_series
        print(f"  Selected series with most slices: {series_id} ({max_slices} slices)")
    else:
        series_id = series_ids[0]
    
    # Get file names for the selected series
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_id)
    
    if len(dicom_names) == 0:
        raise RuntimeError(f"No DICOM files found in {dicom_dir}")
    
    print(f"  Found {len(dicom_names)} DICOM files")
    
    # Read slice positions to ensure correct ordering
    # This fixes issues where GetGDCMSeriesFileNames returns wrong order
    slice_info = []
    for fname in dicom_names:
        file_reader = sitk.ImageFileReader()
        file_reader.SetFileName(fname)
        file_reader.ReadImageInformation()
        
        try:
            # Try to get ImagePositionPatient (most reliable for Z-ordering)
            position = file_reader.GetMetaData("0020|0032")  # ImagePositionPatient
            z_pos = float(position.split("\\")[2])
        except:
            try:
                # Fallback to SliceLocation
                z_pos = float(file_reader.GetMetaData("0020|1041"))  # SliceLocation
            except:
                try:
                    # Fallback to InstanceNumber
                    z_pos = float(file_reader.GetMetaData("0020|0013"))  # InstanceNumber
                except:
                    # Last resort: use file index
                    z_pos = dicom_names.index(fname)
        
        slice_info.append((fname, z_pos))
    
    # Sort by Z position
    slice_info.sort(key=lambda x: x[1])
    sorted_dicom_names = [s[0] for s in slice_info]
    
    # Check if sorting changed the order (indicates original was wrong)
    if sorted_dicom_names != list(dicom_names):
        print(f"  WARNING: Slices were re-ordered by Z-position for correct reconstruction")
    
    reader.SetFileNames(sorted_dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    
    image = reader.Execute()
    
    print(f"  Size: {image.GetSize()}")
    print(f"  Spacing: {image.GetSpacing()}")
    print(f"  Origin: {image.GetOrigin()}")
    
    array = sitk.GetArrayFromImage(image)
    print(f"  Intensity range: [{array.min()}, {array.max()}]")
    print(f"  Array shape (Z,Y,X): {array.shape}")
    
    if scan_name is None:
        scan_name = os.path.basename(dicom_dir.rstrip('/'))
    
    nifti_path = os.path.join(output_dir, f"{scan_name}.nii.gz")
    os.makedirs(output_dir, exist_ok=True)
    sitk.WriteImage(image, nifti_path)
    print(f"Converted DICOM to {nifti_path}")
    
    return nifti_path

def convert_mhd_to_nifti(mhd_path, output_dir):
    image = sitk.ReadImage(mhd_path)

    print(f"\nMHD Image Info:")
    print(f"  Size: {image.GetSize()}")
    print(f"  Spacing: {image.GetSpacing()}")
    print(f"  Origin: {image.GetOrigin()}")

    array = sitk.GetArrayFromImage(image)
    print(f"  Intensity range: [{array.min()}, {array.max()}]")
    print(f"  Array shape (Z,Y,X): {array.shape}")

    base_name = os.path.splitext(os.path.basename(mhd_path))[0]
    nifti_path = os.path.join(output_dir, f"{base_name}.nii.gz")

    os.makedirs(output_dir, exist_ok=True)
    sitk.WriteImage(image, nifti_path)
    print(f"Converted {mhd_path} to {nifti_path}")

    return nifti_path

def segment_airwayfull_from_mhd(mhd_path, output_dir, fast=False, device="gpu"):
    """
    Segment airways from MHD file or DICOM directory.
    
    Args:
        mhd_path: Path to MHD file OR DICOM directory
        output_dir: Output directory for results
        fast: Use fast segmentation mode
        device: "gpu" or "cpu" (default: "gpu")
    
    Returns:
        Path to segmented airway mask
    """
    os.makedirs(output_dir, exist_ok=True)

    # Detect if input is DICOM directory or MHD file
    if os.path.isdir(mhd_path):
        print("\n=== 1) Converting DICOM → NIfTI (for airway segmentation) ===")
        scan_name = os.path.basename(mhd_path.rstrip('/'))
        nifti_path = convert_dicom_to_nifti(mhd_path, output_dir, scan_name)
        base = scan_name
    else:
        print("\n=== 1) Converting MHD → NIfTI (for airway segmentation) ===")
        nifti_path = convert_mhd_to_nifti(mhd_path, output_dir)
        base = os.path.splitext(os.path.basename(mhd_path))[0]

    print(f"\n=== 2) Airway Segmentation with TotalSegmentator (device={device}) ===")

    totalsegmentator(
        nifti_path,
        output_dir,
        task="lung_vessels",
        fast=fast,
        device=device,
    )

    airway_src = os.path.join(output_dir, "lung_trachea_bronchia.nii.gz")
    if not os.path.exists(airway_src):
        raise RuntimeError(
            "ERROR: TotalSegmentator did not generate lung_trachea_bronchia.nii.gz"
        )

    airway_dst = os.path.join(output_dir, f"{base}_airwayfull.nii.gz")

    shutil.move(airway_src, airway_dst)

    print(f"\n✓ Airway segmentation complete: {airway_dst}")
    return airway_dst