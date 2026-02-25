import os
import sys
import numpy as np
import datetime
from pathlib import Path
import SimpleITK as sitk
from test_robust import EnhancedCarinaDetector
from airway_refinement import EnhancedAirwayRefinementModule  
from airwais_seg import segment_airwayfull_from_mhd
from preprocessin_cleaning import SegmentationPreprocessor
from airway_graph import AirwayGraphAnalyzer
from skeleton_cleaner import integrate_skeleton_cleaning
import pandas as pd
from airway_gap_filler import integrate_gap_filling_into_pipeline
from fibrosis_scoring import integrate_fibrosis_scoring
from parenchymal_metrics import integrate_parenchymal_metrics


class CompleteAirwayPipeline:

    def __init__(self, output_root="output"):
        self.output_root = os.path.abspath(output_root)
        os.makedirs(self.output_root, exist_ok=True)
    
    def _read_image(self, path):
        """
        Read CT image from MHD file or DICOM directory.
        
        Args:
            path: Path to MHD file or DICOM directory
            
        Returns:
            SimpleITK.Image object
        """
        if os.path.isdir(path):
            # Read DICOM series with proper slice ordering
            reader = sitk.ImageSeriesReader()
            
            # Get all series IDs
            series_ids = reader.GetGDCMSeriesIDs(path)
            if len(series_ids) == 0:
                raise RuntimeError(f"No DICOM series found in {path}")
            
            # Select series with most slices if multiple
            if len(series_ids) > 1:
                best_series = None
                max_slices = 0
                for sid in series_ids:
                    files = reader.GetGDCMSeriesFileNames(path, sid)
                    if len(files) > max_slices:
                        max_slices = len(files)
                        best_series = sid
                series_id = best_series
            else:
                series_id = series_ids[0]
            
            dicom_names = reader.GetGDCMSeriesFileNames(path, series_id)
            
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
                            z_pos = dicom_names.index(fname)
                
                slice_info.append((fname, z_pos))
            
            slice_info.sort(key=lambda x: x[1])
            sorted_dicom_names = [s[0] for s in slice_info]
            
            reader.SetFileNames(sorted_dicom_names)
            reader.MetaDataDictionaryArrayUpdateOn()
            reader.LoadPrivateTagsOn()
            return reader.Execute()
        else:
            # Read MHD or other single file format
            return sitk.ReadImage(path)

    def process_single_scan(self, mhd_path, scan_name=None, 
                           fast_segmentation=False, device="gpu"):
        """
        Process a single CT scan.
        
        Args:
            mhd_path: Path to MHD file OR DICOM directory
            scan_name: Optional scan name (auto-detected if None)
            fast_segmentation: Use fast mode for TotalSegmentator
            device: "gpu" or "cpu" for TotalSegmentator (default: "gpu")
        """
        
        # Check GPU compatibility if GPU requested
        if device == "gpu":
            try:
                import torch
                if torch.cuda.is_available():
                    capability = torch.cuda.get_device_capability(0)
                    gpu_name = torch.cuda.get_device_name(0)
                    # Actually test CUDA execution instead of guessing from capability number
                    test_tensor = torch.zeros(1, device='cuda')
                    del test_tensor
                    print(f"\n✓ GPU verified: {gpu_name} (sm_{capability[0]}{capability[1]})")
                else:
                    print(f"\n  CUDA not available, switching to CPU")
                    device = "cpu"
            except Exception as e:
                print(f"\n  GPU test failed: {e}")
                print(f"   Switching to CPU mode...")
                device = "cpu"

        if scan_name is None:
            if os.path.isdir(mhd_path):
                scan_name = os.path.basename(mhd_path.rstrip('/'))
            else:
                scan_name = Path(mhd_path).stem

        print("\n" + "="*80)
        print(f" PROCESSING SCAN: {scan_name}")
        print("="*80)

        scan_output_dir = os.path.join(self.output_root, scan_name)
        os.makedirs(scan_output_dir, exist_ok=True)

        step1_dir = os.path.join(scan_output_dir, "step1_segmentation")
        step2_dir = os.path.join(scan_output_dir, "step2_trachea_removal")
        step3_dir = os.path.join(scan_output_dir, "step3_preprocessing")
        step4_dir = os.path.join(scan_output_dir, "step4_analysis")
        step5_dir = os.path.join(scan_output_dir, "step5_parenchymal_metrics")
        step6_dir = os.path.join(scan_output_dir, "step6_fibrosis_assessment")

        for d in [step1_dir, step2_dir, step3_dir, step4_dir, step5_dir, step6_dir]:
            os.makedirs(d, exist_ok=True)

        results = {
            'scan_name': scan_name,
            'input_path': mhd_path,
            'output_dir': scan_output_dir,
            'success': False,
            'error': None
        }

        try:
            print("\n" + "="*80)
            print("STEP 1: AIRWAY SEGMENTATION (TotalSegmentator + Enhanced Refinement)")
            print("="*80)

            airway_original_path = segment_airwayfull_from_mhd(
                mhd_path, 
                step1_dir, 
                fast=fast_segmentation,
                device=device
            )

            print("\n--- Applying Enhanced Refinement ---")
            print("Strategy: Refined mask for connectivity/skeleton, original for metrics")

            ct_img = self._read_image(mhd_path)
            ct_np = sitk.GetArrayFromImage(ct_img)

            airway_img = sitk.ReadImage(airway_original_path)
            airway_np = sitk.GetArrayFromImage(airway_img)
            mask_np = (airway_np > 0).astype(np.uint8)

            ARM = EnhancedAirwayRefinementModule(
                ct_np, 
                mask_np, 
                ct_img.GetSpacing(),
                verbose=True
            )

            refined_np = ARM.refine(
                enable_anti_blob=True,
                min_blob_size_voxels=10,
                min_blob_size_mm3=3,
                max_blob_distance_mm=20.0,
                enable_tubular_smoothing=False,
                enable_skeleton_reconstruction=False
            )

            refined_path = os.path.join(step1_dir, f"{scan_name}_airway_refined_enhanced.nii.gz")
            ARM.save(refined_path, airway_img)

            airway_refined_path = refined_path

            results['airway_segmentation_original'] = airway_original_path
            results['airway_segmentation_refined'] = airway_refined_path
            print(f"\n✓ Original mask (for metrics): {airway_original_path}")
            print(f"✓ Refined mask (for connectivity): {airway_refined_path}")

            print("\n" + "="*80)
            print("STEP 1.5: INTELLIGENT GAP FILLING (on refined mask)")
            print("="*80)

            gap_filled_refined_path, gap_filler = integrate_gap_filling_into_pipeline(
                mhd_path=mhd_path,
                airway_mask_path=airway_refined_path,
                output_dir=step1_dir,
                max_hole_size_mm3=100,
                max_bridge_distance_mm=10.0
            )

            gap_filled_original_path, _ = integrate_gap_filling_into_pipeline(
                mhd_path=mhd_path,
                airway_mask_path=airway_original_path,
                output_dir=step1_dir,
                max_hole_size_mm3=100,
                max_bridge_distance_mm=10.0
            )

            airway_refined_path = gap_filled_refined_path
            airway_original_path = gap_filled_original_path
            results['airway_gap_filled_refined'] = gap_filled_refined_path
            results['airway_gap_filled_original'] = gap_filled_original_path

            print("\n" + "="*80)
            print("STEP 2: ENHANCED TRACHEA REMOVAL")
            print("="*80)
            print("Using ultra-conservative trachea identification method")
            print("Applying to both refined (skeleton) and original (metrics) masks")

            from test_robust import integrate_with_pipeline

            bronchi_mask_refined, carina_coords, confidence, detector = integrate_with_pipeline(
                airway_refined_path,
                spacing=None,
                save_output=True,
                output_dir=step2_dir  
            )

            bronchi_original_path = os.path.join(step2_dir, "bronchi_enhanced_conservative.nii.gz")
            bronchi_refined_filename = f"{scan_name}_bronchi_enhanced_refined.nii.gz"
            bronchi_refined_path = os.path.join(step2_dir, bronchi_refined_filename)

            if os.path.exists(bronchi_original_path):
                if os.path.exists(bronchi_refined_path):
                    os.remove(bronchi_refined_path)
                os.rename(bronchi_original_path, bronchi_refined_path)

            bronchi_mask_original, _, _, _ = integrate_with_pipeline(
                airway_original_path,
                spacing=None,
                save_output=False,
                output_dir=step2_dir  
            )

            bronchi_original_filename = f"{scan_name}_bronchi_enhanced_original.nii.gz"
            bronchi_original_path = os.path.join(step2_dir, bronchi_original_filename)

            ref_img = sitk.ReadImage(airway_original_path)
            bronchi_original_img = sitk.GetImageFromArray(bronchi_mask_original)
            bronchi_original_img.CopyInformation(ref_img)
            sitk.WriteImage(bronchi_original_img, bronchi_original_path)

            bronchi_path = bronchi_refined_path

            carina_z, carina_y, carina_x = carina_coords

            results['trachea_info'] = {
                'detection_method': detector.detection_method,
                'confidence': confidence,
                'trachea_top_z': detector.trachea_top_z,
                'trachea_bottom_z': detector.trachea_bottom_z,
                'trachea_length_slices': detector.trachea_length,
                'trachea_length_mm': detector.trachea_length * detector.spacing[2] if detector.trachea_length else None
            }
            results['carina_coordinates'] = {'z': carina_z, 'y': carina_y, 'x': carina_x}
            results['bronchi_refined_path'] = bronchi_refined_path
            results['bronchi_original_path'] = bronchi_original_path

            print("\n" + "="*80)
            print("STEP 3: PREPROCESSING & COMPONENT RECONNECTION")
            print("="*80)
            print("Processing refined mask for skeleton topology...")

            preprocessor_refined = SegmentationPreprocessor(bronchi_refined_path)

            cleaned_mask_refined, cleaned_path_refined = preprocessor_refined.run_full_preprocessing(
                output_dir=step3_dir,
                try_reconnection=True,
                max_reconnect_distance_mm=15.0,
                min_component_size=50,
                visualize=True
            )

            print("\nProcessing original mask for accurate metrics...")
            print("CRITICAL: Using MINIMAL preprocessing to preserve peripheral airways")
            preprocessor_original = SegmentationPreprocessor(bronchi_original_path)

            cleaned_mask_original, cleaned_path_original = preprocessor_original.run_full_preprocessing(
                output_dir=step3_dir,
                try_reconnection=False,
                max_reconnect_distance_mm=0.0,
                min_component_size=5,
                visualize=False
            )
            print("✓ Original mask preserved with minimal changes (for accurate diameter/volume)")

            results['cleaned_mask_refined'] = cleaned_path_refined
            results['cleaned_mask_original'] = cleaned_path_original
            results['cleaned_mask'] = cleaned_path_refined

            cleaned_skeleton_path, _ = integrate_skeleton_cleaning(
                cleaned_path_refined,
                step4_dir,
                min_component_size=20,
                max_isolation_distance_mm=15.0,
                min_branch_length_mm=5.0
            )

            print("\n" + "="*80)
            print("STEP 4: BRONCHIAL TREE ANALYSIS WITH WEIBEL MODEL")
            print("="*80)
            print("Dual-mask strategy: refined for skeleton, original for metrics")

            analyzer = AirwayGraphAnalyzer(cleaned_path_refined)

            analysis_results = analyzer.run_full_analysis(
                output_dir=step4_dir,
                visualize=True,
                max_reconnect_distance_mm=50.0,
                min_voxels_for_reconnect=10,
                max_voxels_for_keep=200,
                original_mask_path=cleaned_path_original
            )

            results['analysis_results'] = analysis_results
            results['analyzer'] = analyzer

            print("\n" + "="*80)
            print("STEP 5: ADVANCED CLINICAL METRICS")
            print("="*80)

            try:
                advanced_metrics = analyzer.compute_advanced_metrics()
                analyzer.save_advanced_metrics(step4_dir)
                analyzer.plot_advanced_metrics(
                    save_path=os.path.join(step4_dir, "advanced_metrics_summary.png")
                )
                results['advanced_metrics'] = advanced_metrics
                print("\n✓ Advanced metrics computed and saved")

            except Exception as e:
                print(f"\n⚠ Warning: Could not compute advanced metrics: {e}")
                import traceback
                traceback.print_exc()

            print("\n" + "="*80)
            print("STEP 5.5: PARENCHYMAL METRICS")
            print("="*80)

            try:
                parenchymal_metrics = integrate_parenchymal_metrics(
                    mhd_path,
                    step5_dir,
                    fast_segmentation=fast_segmentation,
                    verbose=True,
                    device=device  # Use the working device (may have been switched to CPU)
                )

                results['parenchymal_metrics'] = parenchymal_metrics

                if parenchymal_metrics:
                    print("\n✓ Parenchymal metrics computed successfully")
                    print(f"  Mean Lung Density: {parenchymal_metrics['mean_lung_density_HU']:.1f} HU")
                    print(f"  Histogram Entropy: {parenchymal_metrics['histogram_entropy']:.3f}")
                else:
                    print("\n⚠ Parenchymal metrics computation skipped or failed")

            except Exception as e:
                print(f"\n⚠ Warning: Could not compute parenchymal metrics: {e}")
                import traceback
                traceback.print_exc()
                results['parenchymal_metrics'] = None

            print("\n" + "="*80)
            print("STEP 6: PULMONARY FIBROSIS ASSESSMENT")
            print("="*80)

            try:
                parenchymal_for_scoring = results.get('parenchymal_metrics', None)

                scorer, fibrosis_report = integrate_fibrosis_scoring(
                    analyzer,
                    output_dir=step6_dir,
                    parenchymal_metrics=parenchymal_for_scoring
                )

                results['fibrosis_scorer'] = scorer
                results['fibrosis_report'] = fibrosis_report

                print("\n✓ Fibrosis assessment complete")

            except Exception as e:
                print(f"\n⚠ Warning: Could not complete fibrosis assessment: {e}")
                import traceback
                traceback.print_exc()

            self._generate_summary_report(results, scan_output_dir, analyzer)

            results['success'] = True

        except Exception as e:
            print(f"\n❌ Error processing {scan_name}: {e}")
            import traceback
            traceback.print_exc()
            results['error'] = str(e)
            results['success'] = False

        return results

    def process_folder(self, folder_path, pattern="*.mhd", 
                      fast_segmentation=False, device="gpu"):
        """
        Batch process multiple scans.
        
        Args:
            folder_path: Directory containing MHD files OR DICOM subdirectories
            pattern: File pattern for MHD files (default: *.mhd)
            fast_segmentation: Use fast mode
            device: "gpu" or "cpu" for TotalSegmentator (default: "gpu")
        """

        print("\n" + "="*80)
        print(f" BATCH PROCESSING: {folder_path}")
        print("="*80)

        folder_path = Path(folder_path)
        
        # Try to find MHD files first
        mhd_files = list(folder_path.glob(pattern))
        
        # If no MHD files, check for DICOM subdirectories
        scan_paths = []
        if len(mhd_files) > 0:
            scan_paths = [str(f) for f in mhd_files]
            print(f"\nFound {len(scan_paths)} MHD files to process")
        else:
            # Look for subdirectories (assume they contain DICOM files)
            subdirs = [d for d in folder_path.iterdir() if d.is_dir()]
            if len(subdirs) > 0:
                scan_paths = [str(d) for d in subdirs]
                print(f"\nFound {len(scan_paths)} DICOM directories to process")

        if len(scan_paths) == 0:
            print(f"\n❌ No files matching '{pattern}' or DICOM directories found in {folder_path}")
            return []

        all_results = []

        for idx, scan_path in enumerate(scan_paths, 1):
            scan_name = Path(scan_path).stem if os.path.isfile(scan_path) else Path(scan_path).name
            
            print(f"\n{'='*80}")
            print(f" SCAN {idx}/{len(scan_paths)}: {scan_name}")
            print(f"{'='*80}")

            result = self.process_single_scan(
                scan_path,
                scan_name=scan_name,
                fast_segmentation=fast_segmentation,
                device=device
            )

            all_results.append(result)

        self._generate_batch_summary(all_results, self.output_root)

        return all_results

    def _generate_summary_report(self, results, output_dir, analyzer):

        report_path = os.path.join(output_dir, "COMPLETE_ANALYSIS_REPORT.txt")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(" "*20 + "COMPLETE AIRWAY ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Scan name: {results['scan_name']}\n")
            f.write(f"Input file: {results['input_path']}\n")
            f.write(f"Analysis date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n")

            f.write("="*80 + "\n")
            f.write("PIPELINE STEPS\n")
            f.write("="*80 + "\n\n")

            f.write("1. ✓ Segmentation (TotalSegmentator)\n")
            f.write(f"   Original (for metrics): {results.get('airway_segmentation_original', 'N/A')}\n")
            f.write(f"   Refined (for connectivity): {results.get('airway_segmentation_refined', 'N/A')}\n")
            f.write(f"   Strategy: Dual-mask approach for accurate clinical metrics\n\n")

            f.write("2. ✓ Gap Filling\n")
            f.write(f"   Refined: {results.get('airway_gap_filled_refined', 'N/A')}\n")
            f.write(f"   Original: {results.get('airway_gap_filled_original', 'N/A')}\n\n")

            f.write("3. ✓ Enhanced Trachea Removal\n")
            f.write(f"   Method: Ultra-conservative with pre-cut\n")

            if 'carina_coordinates' in results:
                carina = results['carina_coordinates']
                f.write(f"   Carina: (z={carina['z']}, y={carina['y']}, x={carina['x']})\n")

            if 'trachea_info' in results:
                trachea = results['trachea_info']
                f.write(f"   Detection method: {trachea.get('detection_method', 'N/A')}\n")
                f.write(f"   Confidence: {trachea.get('confidence', 0):.2f}/5.0\n")
                if trachea.get('trachea_length_mm'):
                    f.write(f"   Trachea: z={trachea['trachea_top_z']} to z={trachea['trachea_bottom_z']}\n")
                    f.write(f"   Length: {trachea['trachea_length_mm']:.1f} mm\n")
            f.write("\n")

            f.write("4. ✓ Preprocessing & Component Reconnection\n")
            f.write(f"   Refined (skeleton): {results.get('cleaned_mask_refined', 'N/A')}\n")
            f.write(f"   Original (metrics): {results.get('cleaned_mask_original', 'N/A')}\n\n")

            f.write("5. ✓ Bronchial Tree Analysis (Dual-Mask Strategy)\n")
            f.write(f"   Skeleton from refined mask (better connectivity)\n")
            f.write(f"   Diameters/volumes from original mask (accurate metrics)\n")
            f.write(f"   Output: {os.path.join(output_dir, 'step4_analysis')}\n\n")

            f.write("6. ✓ Advanced Clinical Metrics\n\n")

            f.write("7. ✓ Parenchymal Metrics (Lung Tissue)\n")
            f.write(f"   Output: {os.path.join(output_dir, 'step5_parenchymal_metrics')}\n\n")

            f.write("8. ✓ Pulmonary Fibrosis Assessment\n")
            f.write(f"   Output: {os.path.join(output_dir, 'step6_fibrosis_assessment')}\n\n")

            f.write("="*80 + "\n")
            f.write("BRONCHIAL TREE STATISTICS\n")
            f.write("="*80 + "\n\n")

            if hasattr(analyzer, 'branch_metrics_df') and analyzer.branch_metrics_df is not None:
                df = analyzer.branch_metrics_df

                f.write(f"Total branches: {len(df)}\n")
                f.write(f"Total tree length: {df['length_mm'].sum():.2f} mm\n")
                f.write(f"Total tree volume: {df['volume_mm3'].sum():.2f} mm³\n\n")

                f.write(f"Diameter statistics:\n")
                f.write(f"  Mean: {df['diameter_mean_mm'].mean():.2f} mm\n")
                f.write(f"  Range: {df['diameter_mean_mm'].min():.2f} - {df['diameter_mean_mm'].max():.2f} mm\n\n")

            if hasattr(analyzer, 'weibel_analysis_df') and analyzer.weibel_analysis_df is not None:
                f.write("\nWeibel Generation Analysis:\n")
                weibel_df = analyzer.weibel_analysis_df
                f.write(f"  Maximum generation: {int(weibel_df['generation'].max())}\n")
                f.write(f"  Number of generations: {len(weibel_df)}\n\n")

            if hasattr(analyzer, 'advanced_metrics') and analyzer.advanced_metrics is not None:
                f.write("="*80 + "\n")
                f.write("ADVANCED CLINICAL METRICS\n")
                f.write("="*80 + "\n\n")

                metrics = analyzer.advanced_metrics

                f.write(f"Total airway volume: {metrics['total_volume_mm3']:.2f} mm³\n\n")

                f.write("Peripheral vs Central:\n")
                f.write(f"  Central: {metrics['central_volume_mm3']:.2f} mm³ ({metrics['central_branch_count']} branches)\n")
                f.write(f"  Peripheral: {metrics['peripheral_volume_mm3']:.2f} mm³ ({metrics['peripheral_branch_count']} branches)\n")
                f.write(f"  P/C Ratio: {metrics['peripheral_to_central_ratio']:.3f}\n")

                if metrics['peripheral_to_central_ratio'] < 0.2:
                    f.write(f"    ⚠ LOW - peripheral airway loss\n")
                elif metrics['peripheral_to_central_ratio'] > 0.6:
                    f.write(f"    ✓ HIGH - well-preserved\n")
                else:
                    f.write(f"    ✓ Normal range\n")
                f.write(f"\n")

                if 'mean_tortuosity' in metrics and not pd.isna(metrics['mean_tortuosity']):
                    f.write(f"Tortuosity: {metrics['mean_tortuosity']:.3f}\n")
                    if metrics['mean_tortuosity'] > 1.5:
                        f.write(f"  ⚠ HIGH - airway distortion\n")
                    else:
                        f.write(f"  ✓ Normal range\n")
                    f.write(f"\n")

                if 'symmetry_index' in metrics and not pd.isna(metrics['symmetry_index']):
                    f.write(f"Symmetry Index: {metrics['symmetry_index']:.3f}\n")
                    if metrics['symmetry_index'] < 0.7:
                        f.write(f"  ⚠ ASYMMETRIC\n")
                    else:
                        f.write(f"  ✓ Symmetric\n")
                    f.write(f"\n")

                f.write(f"Generation Coverage: {metrics['generation_coverage']*100:.1f}%\n\n")

            if 'parenchymal_metrics' in results and results['parenchymal_metrics'] is not None:
                f.write("="*80 + "\n")
                f.write("PARENCHYMAL METRICS (LUNG TISSUE)\n")
                f.write("="*80 + "\n\n")

                paren = results['parenchymal_metrics']

                f.write(f"Mean Lung Density: {paren['mean_lung_density_HU']:.1f} HU\n")
                if paren['mean_lung_density_HU'] > -700:
                    f.write(f"  ⚠ ELEVATED - Increased tissue density (possible fibrosis)\n")
                elif paren['mean_lung_density_HU'] < -850:
                    f.write(f"  ⚠ LOW - Emphysematous changes\n")
                else:
                    f.write(f"  ✓ Normal range\n")
                f.write(f"\n")

                f.write(f"Histogram Entropy: {paren['histogram_entropy']:.3f}\n")
                if paren['histogram_entropy'] > 4.5:
                    f.write(f"  ⚠ HIGH - Heterogeneous lung tissue (possible fibrosis)\n")
                elif paren['histogram_entropy'] < 3.5:
                    f.write(f"  ✓ LOW - Homogeneous lung tissue\n")
                else:
                    f.write(f"  ✓ Normal range\n")
                f.write(f"\n")

                f.write("Interpretation:\n")
                f.write("  • Higher density = Denser tissue (fibrosis, consolidation)\n")
                f.write("  • Higher entropy = More heterogeneous patterns (fibrosis, honeycombing)\n")
                f.write("\n\n")

            if 'fibrosis_report' in results and results['fibrosis_report'] is not None:
                f.write("="*80 + "\n")
                f.write("PULMONARY FIBROSIS ASSESSMENT\n")
                f.write("="*80 + "\n\n")

                fib_rep = results['fibrosis_report']
                overall = fib_rep['overall']

                f.write(f"FIBROSIS SCORE: {overall['fibrosis_score']:.1f}/100\n")
                f.write(f"CLASSIFICATION: {overall['stage']}\n")
                f.write(f"CONFIDENCE: {overall['confidence']:.0%}\n\n")

                f.write("Component Breakdown:\n")
                f.write("-" * 80 + "\n")

                for comp_name, comp_data in fib_rep['components'].items():
                    f.write(f"\n{comp_name.replace('_', ' ').title()}:\n")

                    if 'weighted_score_airway' in comp_data:
                        weighted = comp_data['weighted_score_airway']
                    elif 'weighted_score_combined' in comp_data:
                        weighted = comp_data['weighted_score_combined']
                    elif 'weighted_score' in comp_data:
                        weighted = comp_data['weighted_score']
                    else:
                        weighted = 0.0

                    f.write(f"  Contribution: {weighted:.1f} points\n")
                    f.write(f"  Raw score: {comp_data['raw_score']:.1f}/10\n")
                    f.write(f"  Interpretation: {comp_data['interpretation']}\n")

                f.write("\n" + "="*80 + "\n")
                f.write("CLINICAL INTERPRETATION\n")
                f.write("="*80 + "\n\n")

                score = overall['fibrosis_score']

                if score < 20:
                    f.write("No significant fibrotic changes detected.\n")
                    f.write("Airways appear structurally normal.\n")
                elif score < 35:
                    f.write("Minimal fibrotic changes detected.\n")
                    f.write("Early peripheral airway involvement.\n")
                    f.write("Recommendation: Monitor for progression.\n")
                elif score < 50:
                    f.write("Mild fibrosis with measurable airway changes.\n")
                    f.write("Recommendation: Clinical correlation and follow-up.\n")
                elif score < 70:
                    f.write("Moderate fibrosis with UIP-pattern features.\n")
                    f.write("Recommendation: PFT and specialist consultation.\n")
                else:
                    f.write("Severe/advanced fibrosis detected.\n")
                    f.write("Recommendation: Urgent pulmonary evaluation.\n")

                f.write("\n" + "="*80 + "\n")
                f.write("DISCLAIMER\n")
                f.write("="*80 + "\n\n")
                f.write("This is an AUTOMATED ASSESSMENT based on airway morphology.\n")
                f.write("It does NOT replace clinical evaluation, complete CT review,\n")
                f.write("pulmonary function tests, or pathological diagnosis.\n")
                f.write("Always correlate with full clinical picture.\n\n")

            f.write("="*80 + "\n")
            f.write("OUTPUT FILES\n")
            f.write("="*80 + "\n\n")

            f.write(f"Main directory: {output_dir}\n\n")

            f.write("Key files:\n")
            f.write(f"  • step1_segmentation/          - Initial airway segmentation\n")
            f.write(f"  • step2_trachea_removal/       - Bronchi-only mask\n")
            f.write(f"  • step3_preprocessing/         - Cleaned mask\n")
            f.write(f"  • step4_analysis/              - Complete analysis\n")
            f.write(f"    - branch_metrics_complete.csv\n")
            f.write(f"    - weibel_generation_analysis.csv\n")
            f.write(f"    - Multiple visualizations\n")
            f.write(f"  • step5_parenchymal_metrics/   - Lung tissue metrics\n")
            f.write(f"    - parenchymal_metrics.json\n")
            f.write(f"    - parenchymal_report.txt\n")
            f.write(f"    - lung_mask.nii.gz\n")
            f.write(f"  • step6_fibrosis_assessment/   - Fibrosis scoring\n")
            f.write(f"    - fibrosis_assessment_report.txt\n")
            f.write(f"    - fibrosis_assessment.json\n")
            f.write(f"    - fibrosis_assessment_visualization.png\n")

            f.write("\n" + "="*80 + "\n")

        print(f"\n📄 Summary report saved: {report_path}")

    def _generate_batch_summary(self, all_results, output_dir):

        report_path = os.path.join(output_dir, "BATCH_PROCESSING_SUMMARY.txt")

        successful = [r for r in all_results if r['success']]
        failed = [r for r in all_results if not r['success']]

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(" "*25 + "BATCH PROCESSING SUMMARY\n")
            f.write("="*80 + "\n\n")

            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total scans: {len(all_results)}\n")
            f.write(f"Successful: {len(successful)}\n")
            f.write(f"Failed: {len(failed)}\n\n")

            if successful:
                f.write("="*80 + "\n")
                f.write("SUCCESSFUL SCANS\n")
                f.write("="*80 + "\n\n")

                for result in successful:
                    f.write(f"✓ {result['scan_name']}\n")
                    f.write(f"  Output: {result['output_dir']}\n")

                    if 'parenchymal_metrics' in result and result['parenchymal_metrics']:
                        paren = result['parenchymal_metrics']
                        f.write(f"  Parenchymal: Density={paren['mean_lung_density_HU']:.1f} HU, Entropy={paren['histogram_entropy']:.3f}\n")

                    if 'fibrosis_report' in result and result['fibrosis_report']:
                        fib = result['fibrosis_report']['overall']
                        f.write(f"  Fibrosis: {fib['fibrosis_score']:.1f}/100 ({fib['stage']})\n")

                    f.write("\n")

            if failed:
                f.write("="*80 + "\n")
                f.write("FAILED SCANS\n")
                f.write("="*80 + "\n\n")

                for result in failed:
                    f.write(f"❌ {result['scan_name']}\n")
                    f.write(f"  Error: {result['error']}\n\n")

            f.write("="*80 + "\n")

        print(f"\n📄 Batch summary saved: {report_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete Airway Analysis with Fibrosis Assessment')
    parser.add_argument('--input', type=str, required=True,
                        help='Input path: single MHD file or directory for batch processing')
    parser.add_argument('--output', type=str, default='output_results_with_fibrosis',
                        help='Output directory (default: output_results_with_fibrosis)')
    parser.add_argument('--fast', action='store_true',
                        help='Use fast segmentation mode')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for TotalSegmentator (requires CUDA)')
    parser.add_argument('--pattern', type=str, default='*.mhd',
                        help='File pattern for batch mode (default: *.mhd)')
    
    args = parser.parse_args()
    
    INPUT_PATH = args.input
    OUTPUT_DIR = args.output
    FAST_SEGMENTATION = args.fast
    FILE_PATTERN = args.pattern
    DEVICE = "gpu" if args.gpu else "cpu"
    
    # Determine if batch mode based on input path
    BATCH_MODE = os.path.isdir(INPUT_PATH)

    pipeline = CompleteAirwayPipeline(output_root=OUTPUT_DIR)

    print("\n" + "="*80)
    print(" "*10 + "COMPLETE AIRWAY ANALYSIS WITH FIBROSIS ASSESSMENT")
    print("="*80)
    print(f"\nInput: {INPUT_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Batch mode: {BATCH_MODE}")
    print(f"Device: {DEVICE.upper()}")
    print(f"Airway metrics: ENABLED")
    print(f"Parenchymal metrics: ENABLED")
    print(f"Fibrosis assessment: ENABLED")

    if BATCH_MODE:
        if not os.path.isdir(INPUT_PATH):
            print(f"\n❌ Error: {INPUT_PATH} is not a directory")
            sys.exit(1)

        results = pipeline.process_folder(
            INPUT_PATH,
            pattern=FILE_PATTERN,
            fast_segmentation=FAST_SEGMENTATION,
            device=DEVICE
        )

        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]

        print("\n" + "="*80)
        print(" "*20 + "BATCH PROCESSING COMPLETE")
        print("="*80)
        print(f"\nTotal scans: {len(results)}")
        print(f"✓ Successful: {len(successful)}")
        print(f"❌ Failed: {len(failed)}")

    else:
        if not os.path.exists(INPUT_PATH):
            print(f"\n❌ Error: {INPUT_PATH} does not exist")
            sys.exit(1)

        result = pipeline.process_single_scan(INPUT_PATH, fast_segmentation=FAST_SEGMENTATION, device=DEVICE)

        if result['success']:
            print("\n" + "="*80)
            print(" "*30 + "SUCCESS!")
            print("="*80)
            print(f"\n✓ Complete analysis with airway + parenchymal metrics + fibrosis assessment")
            print(f"\n📁 Results: {result['output_dir']}")

            if 'parenchymal_metrics' in result and result['parenchymal_metrics']:
                paren = result['parenchymal_metrics']
                print(f"\nParenchymal Metrics:")
                print(f"  Density: {paren['mean_lung_density_HU']:.1f} HU")
                print(f"  Entropy: {paren['histogram_entropy']:.3f}")

            if 'fibrosis_report' in result and result['fibrosis_report']:
                fib = result['fibrosis_report']['overall']
                print(f"\nFibrosis Score: {fib['fibrosis_score']:.1f}/100")
                print(f"Classification: {fib['stage']}")
        else:
            print(f"\n❌ Error: {result['error']}")
            sys.exit(1)


if __name__ == "__main__":
    main()

"""
Example usage:

# Single MHD file:
python main_pipeline.py \
    --input "/media/carolinalves11/Disk1TB/1.Dados/OSICS/test_mhd/ID00419637202311204720264.mhd" \
    --output /media/carolinalves11/Disk1TB/5.RepoFrancesca/output \
    --fast

# Single DICOM directory:
python main_pipeline.py \
    --input "/media/carolinalves11/Disk1TB/1.Dados/OSICS/test/ID00419637202311204720264" \
    --output /media/carolinalves11/Disk1TB/5.RepoFrancesca/output \
    --gpu

# Batch processing MHD files:
python main_pipeline.py \
    --input /media/carolinalves11/Disk1TB/5.RepoFrancesca/1.Inputs/OSICS_test \
    --output /media/carolinalves11/Disk1TB/5.RepoFrancesca/2.Outputs \
    --fast

# Batch processing DICOM directories:
# HSJ Seleceted (DICOM folders):
python airway_segmentation/main_pipeline.py \
    --input /media/carolinalves11/Disk1TB/1.Dados/HSJ_selected \
    --output /media/carolinalves11/Disk1TB/5.RepoFrancesca/2.Outputs/HSJ_selected \
    --gpu 

# OSICS Seleceted (DICOM folders):
python airway_segmentation/main_pipeline.py \
    --input /media/carolinalves11/Disk1TB/1.Dados/OSICS_selected \
    --output /media/carolinalves11/Disk1TB/5.RepoFrancesca/2.Outputs/OSICS_selected \
    --gpu 

"""