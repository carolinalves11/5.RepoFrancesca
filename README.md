# Airway Segmentation and Fibrosis Assessment Pipeline

Complete pipeline for airway segmentation, morphometric analysis, and pulmonary fibrosis assessment from chest CT images. 

--- 
## 🔍 Description

This automated pipeline performs a comprehensive analysis of bronchial airways from CT scans, including:

- **Automatic airway segmentation** with TotalSegmentator
- **Advanced segmentation refinement** (anti-blob, gap filling)
- **Ultra-conservative trachea removal** with carina identification
- **Preprocessing** with intelligent reconnection of disconnected components
- **3D skeletonization** and topological graph construction
- **Generational analysis** according to the Weibel model
- **Advanced morphometric metrics** (diameters, lengths, tortuosity, symmetry)
- **Parenchymal metrics** (lung density, entropy)
- **Pulmonary fibrosis scoring** (dual scoring: airway-only + combined)

---

## 💻 Requirements

### Software
- Python 3.8 or higher
- Operating System: Windows/Linux/macOS

### Python Libraries
```
numpy
scipy
SimpleITK
scikit-image
pandas
matplotlib
networkx
skan
totalsegmentator
medpy
```

### Recommended Hardware
- RAM: 16 GB minimum, 32 GB recommended
- GPU: Recommended for TotalSegmentator (optional)
- Storage: ~5 GB per single scan (with full output)

## 🔄 Pipeline Structure

The pipeline is organized into **6 main steps**:

### **STEP 1: Segmentation & Refinement**
- **Automatic segmentation** with TotalSegmentator (clean mask but less connected)
- **Enhanced Refinement** to improve connectivity:
  - Initial anti-blob (removes existing blobs)
  - Region growing from endpoints (⚠️ may create blobs)
  - Thin airway recovery 
  - Skeleton-guided expansion 
  - Morphological dilation 
- **Intelligent Gap Filling** (filling anatomical gaps and holes)

**⚠️ Note:** Refinement improves connectivity but may introduce blobs. This is why the dual-mask strategy is used.

### **STEP 2: Trachea Removal**
- Ultra-conservative carina identification
- Trachea removal preserving main bronchi
- Dual-mask strategy (refined + original)

### **STEP 3: Preprocessing**
- Connected component analysis
- Intelligent reconnection of disconnected components
- Artifact cleaning

### **STEP 4: Morphometric Analysis**
- 3D Skeletonization
- Topological graph construction
- Generational analysis (Weibel model)
- Metrics calculation: diameters, lengths, volumes, tortuosity
- Bifurcations and tapering

### **STEP 5: Parenchymal Metrics**
- Lung segmentation (TotalSegmentator)
- Mean lung density (HU)
- Histogram entropy
- Texture patterns

### **STEP 6: Fibrosis Assessment**
- **Option 1**: Airway-only scoring (airway metrics only)
- **Option 2**: Combined scoring (airway + parenchymal) - **RECOMMENDED**
- Severity classification: Mild / Moderate / Severe / Very Severe

---

## 📖 Usage

### Single Scan Processing

```python
from main_pipeline import CompleteAirwayPipeline

pipeline = CompleteAirwayPipeline(output_root="output")

results = pipeline.process_single_scan(
    mhd_path="path/to/scan.mhd",
    scan_name="patient_001",
    fast_segmentation=False  # True for fast mode
)

print(f"Success: {results['success']}")
print(f"Fibrosis Score: {results.get('fibrosis_score', 'N/A')}")
```

### Batch Processing (Multiple Patients)

```python
from main_pipeline import CompleteAirwayPipeline
from pathlib import Path

pipeline = CompleteAirwayPipeline(output_root="output_batch")

scans_dir = Path("data/scans")
mhd_files = list(scans_dir.glob("*.mhd"))

for mhd_file in mhd_files:
    try:
        results = pipeline.process_single_scan(
            mhd_path=str(mhd_file),
            scan_name=mhd_file.stem
        )
        print(f"✓ {mhd_file.stem}: Score = {results.get('fibrosis_score', 'N/A')}")
    except Exception as e:
        print(f"✗ {mhd_file.stem}: {e}")
```

### Example Scripts

```bash
# Example with single scan (fast mode)
python main_pipeline.py --input scan.mhd --output results --fast

# Example with batch processing
python main_pipeline.py --batch data/scans/ --output results_batch
```

---

## 📊 Output

For each processed scan, the pipeline generates a directory structure:

```
output/
└── patient_001/
    ├── step1_segmentation/
    │   ├── patient_001_airwayfull.nii.gz              # Original segmentation
    │   ├── patient_001_airway_refined_enhanced.nii.gz # Refined segmentation
    │   └── patient_001_airway_gap_filled.nii.gz       # With gap filling
    │
    ├── step2_trachea_removal/
    │   ├── bronchi_enhanced_refined.nii.gz            # Bronchi only (refined)
    │   ├── bronchi_enhanced_original.nii.gz           # Bronchi only (original)
    │   └── carina_info.json                           # Carina coordinates
    │
    ├── step3_preprocessing/
    │   ├── bronchi_cleaned.nii.gz                     # After preprocessing
    │   └── component_reconnection_report.txt
    │
    ├── step4_analysis/
    │   ├── skeleton.nii.gz                            # 3D Skeleton
    │   ├── branch_metrics_complete.csv                # Complete branch metrics
    │   ├── weibel_generation_analysis.csv             # Generation analysis
    │   ├── advanced_metrics.json                      # Advanced metrics
    │   ├── bifurcations.csv                           # Bifurcations
    │   └── [visualizations]                           # Graphs and visualizations
    │
    ├── step5_parenchymal_metrics/
    │   ├── lung_mask_combined.nii.gz                  # Lung mask
    │   └── parenchymal_metrics.json                   # Parenchymal metrics
    │
    ├── step6_fibrosis_assessment/
    │   ├── fibrosis_scores_airway_only.json           # Option 1 scores
    │   ├── fibrosis_scores_combined.json              # Option 2 scores (RECOMMENDED)
    │   └── fibrosis_visualization.png                 # Visualization
    │
    └── COMPLETE_ANALYSIS_REPORT.txt                   # Complete final report
```

### Final Report

The `COMPLETE_ANALYSIS_REPORT.txt` file contains:

1. **Scan information** (dimensions, spacing, quality)
2. **Bronchial tree statistics** (branches, length, volume, diameters)
3. **Weibel generational analysis** (coverage, tapering)
4. **Advanced clinical metrics** (P/C ratio, tortuosity, symmetry)
5. **Parenchymal metrics** (density, entropy)
6. **Dual Fibrosis Scoring**:
   - Airway-only score (0-100)
   - Combined score (0-100) - **RECOMMENDED**
7. **Clinical interpretation** and severity classification

---

## 🧩 Main Modules

### `main_pipeline.py`
**Main orchestrator** of the pipeline. Coordinates all steps and executes complete analysis.

**Main class:** `CompleteAirwayPipeline`

**Key methods:**
- `process_single_scan()`: Processes a single scan
- `process_batch()`: Processes multiple scans
- `_generate_complete_report()`: Generates final report

---

### `airwais_seg.py`
**Initial airway segmentation** with TotalSegmentator.

**Main functions:**
- `convert_mhd_to_nifti()`: Converts MHD → NIfTI
- `segment_airwayfull_from_mhd()`: Segments complete airways

**TotalSegmentator task:** `lung_vessels` → extracts `lung_trachea_bronchia`

---

### `airway_refinement.py`
**Advanced segmentation refinement** with anti-blob techniques.

**Class:** `EnhancedAirwayRefinementModule`

**Functionalities:**
- Region growing from endpoints (recovers peripheral airways)
- Thin airway recovery (adaptive HU thresholds central/peripheral)
- Skeleton-guided expansion
- Morphological smoothing
- **Isolated blob removal (small artifacts)**

**⚠️ IMPORTANT:** Refinement operations (region growing, recovery, dilation) can **create blobs**. Anti-blob removes blobs at the beginning, but subsequent operations may re-create them. The original TotalSegmentator segmentation is generally cleaner.

**Main method:** `refine()`

---

### `airway_gap_filler.py`
**Intelligent gap and hole filling** in airways.

**Class:** `IntelligentAirwayGapFiller`

**Strategy:**
1. Identify gaps and holes
2. Build preliminary skeleton for topology
3. Fill gaps following anatomically plausible paths
4. Validate with coherent HU intensities (air)

**Integration function:** `integrate_gap_filling_into_pipeline()`

---

### `test_robust.py`
**Ultra-conservative trachea removal** with carina identification.

**Class:** `EnhancedCarinaDetector`

**Method:**
- Identify supine vs prone
- Estimate carina position (anatomical distance from lung apex)
- Conservatively cut trachea
- Quality verification (component control)

**Output:** Bronchi-only mask + carina coordinates

**Integration function:** `integrate_with_pipeline()`

---

### `preprocessin_cleaning.py`
**Preprocessing and cleaning** of bronchial segmentation.

**Class:** `SegmentationPreprocessor`

**Functionalities:**
- Connected component analysis
- Maintains main component
- Intelligent reconnection of nearby components (path-based distances)
- Isolated artifact removal

**Main method:** `preprocess()`

---

### `skeleton_cleaner.py`
**Skeleton post-processing** for artifact removal.

**Class:** `SkeletonCleaner`

**Operations:**
- Removal of small components (< voxel threshold)
- Topology smoothing
- Connectivity validation

**Integration function:** `integrate_skeleton_cleaning()`

---

### `airway_graph.py`
**Complete topological and morphometric analysis** of the bronchial tree.

**Class:** `AirwayGraphAnalyzer`

**Main functionalities:**

1. **3D Skeletonization** (`compute_skeleton()`)
2. **NetworkX graph construction** (`build_graph()`)
3. **Carina identification by diameter** (`identify_carina()`)
4. **Weibel generation assignment** (`assign_generations_weibel()`)
5. **Metrics calculation:**
   - Branch diameters (`analyze_diameters()`)
   - Branch lengths (`calculate_branch_lengths()`)
   - Volumes and areas
   - Distances from carina
   - Inter-generational tapering
6. **Advanced clinical metrics** (`compute_advanced_metrics()`):
   - P/C Ratio (Peripheral/Central)
   - Tortuosity
   - Symmetry index
   - Generation coverage
   - Volumetric distribution
7. **3D visualizations** (skeleton, graph, generations, diameters)

**Dual-mask strategy:** Skeleton from refined, metrics from original

**Complete method:** `run_full_analysis()`

---

### `fibrosis_scoring.py`
**Scoring system** for pulmonary fibrosis assessment.

**Class:** `PulmonaryFibrosisScorer`

**Dual Scoring System:**

#### **Option 1: Airway-only** (airway metrics only)
Weights:
- Peripheral density: 35%
- Peripheral volume: 25%
- P/C ratio: 20%
- Tortuosity: 15%
- Symmetry: 5%

#### **Option 2: Combined** (airway + parenchymal) - **RECOMMENDED**
Weights:
- Parenchymal entropy: 35%
- Parenchymal density: 25%
- Peripheral density: 15%
- Peripheral volume: 15%
- Tortuosity: 5%
- Symmetry: 5%

**Output:** Score 0-100 + severity classification

**Integration function:** `integrate_fibrosis_scoring()`

---

### `parenchymal_metrics.py`
**Parenchymal metrics calculation** for IPF assessment.

**Calculated metrics:**
1. **Mean Lung Density (HU)** - Average lung tissue density
2. **Histogram Entropy** - Texture pattern heterogeneity

**Functions:**
- `segment_lungs_totalsegmentator()`: Segments lungs
- `compute_parenchymal_metrics()`: Calculates metrics
- `integrate_parenchymal_metrics()`: Pipeline integration

**Correlation with FVC%:** Entropy (r=-0.686***), Density (r=-0.648***)


## ⚙️ Advanced Configuration

### Key modifiable parameters

**In `main_pipeline.py`:**
```python
pipeline.process_single_scan(
    mhd_path="scan.mhd",
    fast_segmentation=False,  # True for TotalSegmentator fast mode
)
```

**Enhanced Refinement:**
```python
ARM.refine(
    enable_anti_blob=True,
    min_blob_size_voxels=10,      # Minimum blob size to remove
    min_blob_size_mm3=3,          # Minimum size in mm³
    max_blob_distance_mm=20.0,    # Maximum distance to consider isolated
)
```

**Gap Filling:**
```python
integrate_gap_filling_into_pipeline(
    max_hole_size_mm3=100,        # Maximum hole size to fill
    max_bridge_distance_mm=10.0   # Maximum distance for bridging
)
```

**Preprocessing:**
```python
preprocessor.preprocess(
    keep_top_n_components=1,
    reconnect_nearby=True,
    max_reconnect_distance_mm=15.0,  # Maximum distance for reconnection
    min_component_size_voxels=10
)
```

**Graph Analysis:**
```python
analyzer.run_full_analysis(
    max_reconnect_distance_mm=15.0,
    min_voxels_for_reconnect=5,
    max_voxels_for_keep=100,
    visualize=True
)
```

---

## 📝 Important Notes

### Blobs and Refinement
**IMPORTANT:** Refinement operations (region growing, thin airway recovery, dilation) can **create small isolated blobs**. This occurs because:
1. **Region growing** from endpoints may grow into areas with ambiguous HU values
2. **Thin airway recovery** may retrieve false positives
3. **Skeleton-guided expansion** amplifies even small fragments
4. **Final dilation** enlarges all voxels, including micro-artifacts

The **original TotalSegmentator segmentation** is generally cleaner. Refinement improves **connectivity** (useful for skeleton/topology) but may introduce blobs.

### Dual-Mask Strategy
The pipeline uses **two masks** for this reason:
1. **Refined mask**: For skeleton and topology (optimal connectivity, but potential blobs)
2. **Original mask**: For geometric metrics (cleaner, dimensional accuracy)

This approach balances connectivity and metric accuracy.

### Fibrosis Scoring
**Option 2 (Combined) is recommended** because:
- Includes parenchymal metrics (better correlations with FVC%)
- More robust and clinically relevant score
- Validated on OSIC dataset


### Error Handling
The pipeline includes robust error handling:
- Try-catch on each step
- Detailed logging
- Error report in final file

---
