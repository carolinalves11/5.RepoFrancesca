'''
REtrieved from: https://gist.github.com/jizhang02/6e395880c085f7c9884d9cec5490c710

-----------------------------------------------
File Name: convert_dcm_to_mhd$
Description: convert dicom files into mhd file
Author: Jing$
Date: 16/11/2022$
-----------------------------------------------
'''

import os
import SimpleITK as sitk

def convert(input_path, output_path):
    '''
    Convert DICOM series to MHD format using SimpleITK's robust DICOM reader.
    
    :param input_path: the path pattern of dicom files (e.g., "folder/*.dcm")
    :param output_path: the output folder path
    :return: True if successful, False otherwise
    '''
    
    # Extract directory from input_path pattern
    input_dir = os.path.dirname(input_path)
    
    if not os.path.exists(input_dir):
        print(f"  ✗ Error: Input directory {input_dir} does not exist")
        return False
    
    # Use SimpleITK's ImageSeriesReader for robust DICOM reading
    reader = sitk.ImageSeriesReader()
    
    try:
        dicom_names = reader.GetGDCMSeriesFileNames(input_dir)
    except Exception as e:
        print(f"  ✗ Error reading DICOM directory: {e}")
        return False
    
    if len(dicom_names) == 0:
        print(f"  ✗ No DICOM files found in {input_dir}")
        return False
    
    print(f"  Found {len(dicom_names)} DICOM files")
    
    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    
    try:
        image = reader.Execute()
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Save as MHD
        output_file = os.path.join(output_path, os.path.basename(input_dir) + ".mhd")
        sitk.WriteImage(image, output_file)
        
        print(f"  ✓ Converted successfully")
        print(f"    Size: {image.GetSize()}, Spacing: {image.GetSpacing()}")
        return True
        
    except RuntimeError as e:
        error_msg = str(e)
        if "Corrupt JPEG" in error_msg or "bad Huffman" in error_msg or "Transfer Syntax" in error_msg:
            print(f"  ✗ Corrupted DICOM files detected - skipping")
        else:
            print(f"  ✗ RuntimeError: {error_msg[:100]}...")
        return False
        
    except Exception as e:
        print(f"  ✗ Conversion failed: {str(e)[:100]}...")
        return False

########

# 1. define output path
output_dir = "/media/carolinalves11/Disk1TB/5.RepoFrancesca/1.Inputs/OSICS_train"
input_dir = "/media/carolinalves11/Disk1TB/1.Dados/OSICS/train"

# 2. loops through all the folders in input path, and convert dicom files into mhd files
if os.path.exists(input_dir):
    folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    print(f"\nBatch DICOM to MHD Conversion")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Found {len(folders)} folders to process\n")
    
    success_count = 0
    fail_count = 0
    failed_folders = []
    
    for idx, folder in enumerate(folders, 1):
        print(f"[{idx}/{len(folders)}] {folder}")
        input_path = os.path.join(input_dir, folder, "*.dcm")
        output_folder = os.path.join(output_dir, folder)  # Create folder-specific output
        os.makedirs(output_folder, exist_ok=True)
        
        success = convert(input_path=input_path, output_path=output_folder)
        
        if success:
            success_count += 1
        else:
            fail_count += 1
            failed_folders.append(folder)
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"✓ Success: {success_count}/{len(folders)}")
    if fail_count > 0:
        print(f"✗ Failed: {fail_count}/{len(folders)}")
        print(f"\nFailed folders:")
        for folder in failed_folders:
            print(f"  - {folder}")
    print(f"{'='*60}")
else:
    print(f"Error: Input directory not found: {input_dir}")

          
#convert(input_path="/media/carolinalves11/Disk1TB/1.Dados/OSICS/test/ID00419637202311204720264/*.dcm",output_path="/media/carolinalves11/Disk1TB/1.Dados/OSICS/test_mhd/ID00419637202311204720264")



