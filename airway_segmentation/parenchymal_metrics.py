


import os
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from totalsegmentator.python_api import totalsegmentator
    TOTALSEGMENTATOR_AVAILABLE = True
except ImportError:
    TOTALSEGMENTATOR_AVAILABLE = False
    print("Warning: TotalSegmentator not available for parenchymal metrics")



def segment_lungs_totalsegmentator(ct_path, output_dir, fast=True, device="gpu"):


    if not TOTALSEGMENTATOR_AVAILABLE:
        raise ImportError("TotalSegmentator is not installed. Cannot segment lungs.")

    print(f"  Segmenting lungs with TotalSegmentator (device={device})...")

    if isinstance(ct_path, (str, Path)):
        ct_image = sitk.ReadImage(str(ct_path))
    else:
        ct_image = ct_path

    nifti_path = output_dir / "ct_temp.nii.gz"
    sitk.WriteImage(ct_image, str(nifti_path))

    print(f"    Running TotalSegmentator (task='total', device={device})...")
    totalsegmentator(
        str(nifti_path),
        str(output_dir),
        task="total",
        roi_subset=["lung_upper_lobe_left", "lung_lower_lobe_left", 
                    "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"],
        fast=fast,
        ml=False,
        device=device
    )

    # save lung masks of each lobe with its tag, in a single nifty file (10-LUL, 11-LLL, 12-LUR, 13-LML, 14-LLR)
    tag_map = {
        "lung_upper_lobe_left": 10,
        "lung_lower_lobe_left": 11,
        "lung_upper_lobe_right": 12,
        "lung_middle_lobe_right": 13,
        "lung_lower_lobe_right": 14
    }

    combined_mask = np.zeros_like(sitk.GetArrayFromImage(ct_image), dtype=np.uint8)
    lung_mask = np.zeros_like(combined_mask, dtype=np.uint8)
    found_any_lobe = False

    for part_name, tag in tag_map.items():
        part_path = output_dir / f"{part_name}.nii.gz"
        if part_path.exists():
            part_image = sitk.ReadImage(str(part_path))
            part_array = sitk.GetArrayFromImage(part_image)
            lobe_voxels = part_array > 0
            combined_mask[lobe_voxels] = tag
            lung_mask[lobe_voxels] = 1
            found_any_lobe = True

    if not found_any_lobe:
        raise RuntimeError("TotalSegmentator did not generate lung lobe masks")

    combined_mask_image = sitk.GetImageFromArray(combined_mask)
    combined_mask_image.CopyInformation(ct_image)
    sitk.WriteImage(combined_mask_image, str(output_dir / "total_segmentator_0000.nii.gz"))


    # then, combine them
    lung_parts = [
        "lung_upper_lobe_left.nii.gz",
        "lung_lower_lobe_left.nii.gz",
        "lung_upper_lobe_right.nii.gz",
        "lung_middle_lobe_right.nii.gz",
        "lung_lower_lobe_right.nii.gz"
    ]

    lung_mask = None

    for part_name in lung_parts:
        part_path = output_dir / part_name
        if part_path.exists():
            part_image = sitk.ReadImage(str(part_path))
            part_array = sitk.GetArrayFromImage(part_image)

            if lung_mask is None:
                lung_mask = (part_array > 0).astype(np.uint8)
            else:
                lung_mask = np.logical_or(lung_mask, part_array > 0).astype(np.uint8)

            part_path.unlink(missing_ok=True)

    if lung_mask is None:
        raise RuntimeError("TotalSegmentator did not generate lung lobe masks")

    print(f"    Lung volume: {np.sum(lung_mask)} voxels")

    nifti_path.unlink(missing_ok=True)



    return lung_mask, combined_mask



class ParenchymalMetricsComputer:

    def __init__(self, ct_array, spacing, lung_mask, verbose=True):
        self.ct_array = ct_array
        self.spacing = spacing
        self.lung_mask = lung_mask
        self.verbose = verbose

        self.lung_hu = ct_array[lung_mask > 0]

        self.metrics = {}


    def compute_all_metrics(self):
        if self.verbose:
            print("\n" + "="*60)
            print("COMPUTING PARENCHYMAL METRICS")
            print("="*60)

        self.compute_density_metrics()
        self.compute_histogram_features()

        if self.verbose:
            print("\n" + "="*60)
            print("PARENCHYMAL METRICS COMPLETE")
            print("="*60)

        return self.metrics


    def compute_density_metrics(self):
        if self.verbose:
            print("\n[1/2] Computing Mean Lung Density (HU)...")

        lung_hu = self.lung_hu


        mean_density = float(np.mean(lung_hu))

        self.metrics['mean_lung_density_HU'] = mean_density

        if self.verbose:
            print(f"    Mean lung density: {mean_density:.1f} HU")
            print(f"    Formula: (1/{len(lung_hu)}) * Σ(HU_i)")


    def compute_histogram_features(self):
        if self.verbose:
            print("\n[2/2] Computing Histogram Entropy...")

        lung_hu = self.lung_hu

        hist, bin_edges = np.histogram(lung_hu, bins=100, range=(-1024, 100))

        hist_normalized = hist / np.sum(hist)

        hist_nonzero = hist_normalized[hist_normalized > 0]

        entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))

        self.metrics['histogram_entropy'] = float(entropy)

        if self.verbose:
            print(f"    Histogram entropy: {entropy:.3f}")
            print(f"    Formula: -Σ p_j * log₂(p_j)")
            print(f"    Bins: 100 from -1024 to 100 HU")



def integrate_parenchymal_metrics(mhd_path, output_dir, fast_segmentation=True, verbose=True, device="gpu"):
    """
    Compute parenchymal metrics from MHD file or DICOM directory.
    
    Args:
        mhd_path: Path to MHD file OR DICOM directory
        output_dir: Output directory
        fast_segmentation: Use fast mode
        verbose: Print detailed information
        device: "gpu" or "cpu" (default: "gpu")
    """

    if verbose:
        print("\n" + "="*80)
        print("STEP 5: PARENCHYMAL METRICS COMPUTATION")
        print("="*80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if verbose:
            print(f"\nLoading CT scan from: {mhd_path}")

        # Handle DICOM directories
        if os.path.isdir(str(mhd_path)):
            if verbose:
                print("  Detected DICOM directory, converting to image...")
            reader = sitk.ImageSeriesReader()
            
            # Get series IDs and select the one with most slices
            series_ids = reader.GetGDCMSeriesIDs(str(mhd_path))
            if len(series_ids) == 0:
                raise RuntimeError(f"No DICOM series found in {mhd_path}")
            
            if len(series_ids) > 1:
                best_series = None
                max_slices = 0
                for sid in series_ids:
                    files = reader.GetGDCMSeriesFileNames(str(mhd_path), sid)
                    if len(files) > max_slices:
                        max_slices = len(files)
                        best_series = sid
                series_id = best_series
            else:
                series_id = series_ids[0]
            
            dicom_names = reader.GetGDCMSeriesFileNames(str(mhd_path), series_id)
            
            # Sort slices by Z position to fix reconstruction issues
            slice_info = []
            for fname in dicom_names:
                file_reader = sitk.ImageFileReader()
                file_reader.SetFileName(fname)
                file_reader.ReadImageInformation()
                
                try:
                    position = file_reader.GetMetaData("0020|0032")
                    z_pos = float(position.split("\\")[2])
                except:
                    try:
                        z_pos = float(file_reader.GetMetaData("0020|1041"))
                    except:
                        try:
                            z_pos = float(file_reader.GetMetaData("0020|0013"))
                        except:
                            z_pos = list(dicom_names).index(fname)
                
                slice_info.append((fname, z_pos))
            
            slice_info.sort(key=lambda x: x[1])
            sorted_dicom_names = [s[0] for s in slice_info]
            
            reader.SetFileNames(sorted_dicom_names)
            reader.MetaDataDictionaryArrayUpdateOn()
            reader.LoadPrivateTagsOn()
            ct_image = reader.Execute()
        else:
            ct_image = sitk.ReadImage(str(mhd_path))
        
        ct_array = sitk.GetArrayFromImage(ct_image)
        spacing = ct_image.GetSpacing()

        if verbose:
            print(f"  Shape: {ct_array.shape}")
            print(f"  Spacing: {spacing} mm")
            print(f"  HU range: [{ct_array.min():.0f}, {ct_array.max():.0f}]")

        if verbose:
            print(f"\nSegmenting lungs with TotalSegmentator...")

        segmentation_dir = output_dir / "segmentation_temp"
        segmentation_dir.mkdir(parents=True, exist_ok=True)

        lung_mask, lobe_multilabel_mask = segment_lungs_totalsegmentator(
            ct_image,
            segmentation_dir,
            fast=fast_segmentation,
            device=device
        )

        if verbose:
            print(f"\nComputing parenchymal metrics...")

        computer = ParenchymalMetricsComputer(ct_array, spacing, lung_mask, verbose=verbose)
        metrics = computer.compute_all_metrics()


        metrics_json_path = output_dir / "parenchymal_metrics.json"
        with open(metrics_json_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        if verbose:
            print(f"\n✓ Metrics saved to: {metrics_json_path}")

        mask_path = output_dir / "lung_mask.nii.gz"
        mask_sitk = sitk.GetImageFromArray(lung_mask.astype(np.uint8))
        mask_sitk.CopyInformation(ct_image)
        sitk.WriteImage(mask_sitk, str(mask_path))

        lobe_mask_path = output_dir / "lung_lobes_multilabel_0000.nii.gz"
        lobe_mask_sitk = sitk.GetImageFromArray(lobe_multilabel_mask.astype(np.uint8))
        lobe_mask_sitk.CopyInformation(ct_image)
        sitk.WriteImage(lobe_mask_sitk, str(lobe_mask_path))

        if verbose:
            print(f"✓ Lung mask saved to: {mask_path}")
            print(f"✓ Lobe multilabel mask saved to: {lobe_mask_path}")

        report_path = output_dir / "parenchymal_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("PARENCHYMAL METRICS REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"CT scan: {mhd_path}\n\n")

            f.write("PARENCHYMAL METRICS (2 KEY METRICS)\n")
            f.write("-"*80 + "\n")
            f.write(f"  1. Mean Lung Density: {metrics.get('mean_lung_density_HU', np.nan):.1f} HU\n")
            f.write(f"     Formula: (1/N) * Σ(HU_i) where N = {len(computer.lung_hu)} voxels\n\n")

            f.write(f"  2. Histogram Entropy: {metrics.get('histogram_entropy', np.nan):.3f}\n")
            f.write(f"     Formula: -Σ p_j * log₂(p_j)\n")
            f.write(f"     Bins: 100 from -1024 to 100 HU\n\n")

            f.write("INTERPRETATION:\n")
            f.write("-"*80 + "\n")
            f.write("  • Higher Mean Density = Denser lung tissue (fibrosis)\n")
            f.write("  • Higher Entropy = More heterogeneous tissue patterns\n\n")

            f.write("OUTPUT MASKS:\n")
            f.write("-"*80 + "\n")
            f.write(f"  • Binary lung mask: {mask_path}\n")
            f.write(f"  • Lobe multilabel mask: {lobe_mask_path}\n")
            f.write("    Label map: 10=LUL, 11=LLL, 12=RUL, 13=RML, 14=RLL\n\n")

            f.write("="*80 + "\n")

        if verbose:
            print(f"✓ Report saved to: {report_path}")
            print(f"\n✓ Parenchymal metrics computation complete")

        return metrics

    except Exception as e:
        if verbose:
            print(f"\n⚠ Warning: Could not compute parenchymal metrics: {e}")
            import traceback
            traceback.print_exc()

        return None



def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compute parenchymal metrics')
    parser.add_argument('mhd_path', type=str, help='Path to MHD file')
    parser.add_argument('--output', type=str, default='parenchymal_output', 
                        help='Output directory')
    parser.add_argument('--fast', action='store_true', 
                        help='Use fast segmentation mode')

    args = parser.parse_args()

    metrics = integrate_parenchymal_metrics(
        args.mhd_path,
        args.output,
        fast_segmentation=args.fast,
        verbose=True
    )

    if metrics:
        print("\n" + "="*80)
        print("METRICS COMPUTED SUCCESSFULLY")
        print("="*80)
        print(f"\nMean Lung Density: {metrics['mean_lung_density_HU']:.1f} HU")
        print(f"Histogram Entropy: {metrics['histogram_entropy']:.3f}")
    else:
        print("\n❌ Failed to compute metrics")


if __name__ == "__main__":
    main()
