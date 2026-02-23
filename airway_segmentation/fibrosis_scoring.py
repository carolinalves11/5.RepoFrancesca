import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os


class PulmonaryFibrosisScorer:


    def __init__(self, analyzer, verbose=True):
        self.analyzer = analyzer
        self.verbose = verbose

        self.reference_values = {
            'pc_ratio': {'mean': 0.45, 'std': 0.15, 'min': 0.25, 'max': 0.65},
            'tortuosity': {'mean': 1.25, 'std': 0.15, 'min': 1.0, 'max': 1.5},
            'generation_coverage': {'mean': 0.85, 'std': 0.10, 'min': 0.70, 'max': 1.0},
            'symmetry_index': {'mean': 0.85, 'std': 0.10, 'min': 0.70, 'max': 1.0},
            'tapering_ratio': {'mean': 0.79, 'std': 0.05, 'min': 0.70, 'max': 0.88}
        }


        self.weights_airway_only = {
            'peripheral_density': 0.35,
            'peripheral_volume': 0.25,
            'pc_ratio': 0.20,
            'tortuosity': 0.15,
            'symmetry': 0.05,
        }

        self.weights_combined = {
            'parenchymal_entropy': 0.35,
            'parenchymal_density': 0.25,

            'peripheral_density': 0.15,
            'peripheral_volume': 0.15,

            'tortuosity': 0.05,
            'symmetry': 0.05,
        }

        self.weights = self.weights_airway_only

        self.component_scores = {}

        self.fibrosis_score_airway_only = None
        self.severity_stage_airway_only = None
        self.confidence_airway_only = None

        self.fibrosis_score_combined = None
        self.severity_stage_combined = None
        self.confidence_combined = None

        self.fibrosis_score = None
        self.severity_stage = None
        self.confidence = None

        self.parenchymal_metrics = None

    def compute_pc_ratio_score(self):
        if self.verbose:
            print("\n[1/6] Computing P/C Ratio Score...")

        if not hasattr(self.analyzer, 'advanced_metrics'):
            raise ValueError("Run analyzer.compute_advanced_metrics() first")

        metrics = self.analyzer.advanced_metrics
        pc_ratio = metrics.get('peripheral_to_central_ratio', np.nan)

        if np.isnan(pc_ratio):
            if self.verbose:
                print("  ⚠ P/C ratio not available")
            return {'score': 5.0, 'value': np.nan, 'interpretation': 'Unknown'}

        ref = self.reference_values['pc_ratio']


        if pc_ratio < 0.15:
            score = 10.0
            interpretation = "Severe peripheral airway loss"
        elif pc_ratio < 0.25:
            score = 7.0 + (0.25 - pc_ratio) / 0.10 * 3.0
            interpretation = "Moderate to severe peripheral loss"
        elif pc_ratio < 0.40:
            score = 4.0 + (0.40 - pc_ratio) / 0.15 * 3.0
            interpretation = "Mild peripheral airway loss"
        elif pc_ratio <= 0.65:
            z_score = abs(pc_ratio - ref['mean']) / ref['std']
            score = min(3.0, z_score * 1.5)
            interpretation = "Within normal range"
        else:
            score = 2.0 + min(1.0, (pc_ratio - 0.65) / 0.20)
            interpretation = "Unusually high (possible compensatory changes)"

        if self.verbose:
            print(f"  P/C Ratio: {pc_ratio:.3f}")
            print(f"  Reference range: {ref['min']:.2f} - {ref['max']:.2f}")
            print(f"  Score: {score:.1f}/10")
            print(f"  {interpretation}")

        return {
            'raw_score': score,
            'value': pc_ratio,
            'interpretation': interpretation,
            'reference_range': (ref['min'], ref['max'])
        }

    def compute_tortuosity_score(self):
        if self.verbose:
            print("\n[2/6] Computing Tortuosity Score...")

        if not hasattr(self.analyzer, 'advanced_metrics'):
            raise ValueError("Run analyzer.compute_advanced_metrics() first")

        metrics = self.analyzer.advanced_metrics
        tortuosity = metrics.get('mean_tortuosity', np.nan)

        if np.isnan(tortuosity):
            if self.verbose:
                print("  ⚠ Tortuosity not available")
            return {'score': 5.0, 'value': np.nan, 'interpretation': 'Unknown'}

        ref = self.reference_values['tortuosity']


        if tortuosity < 1.0:
            score = 1.0 + (1.0 - tortuosity) * 2.0
            interpretation = "Unusually straight airways"
        elif tortuosity <= 1.5:
            z_score = abs(tortuosity - ref['mean']) / ref['std']
            score = min(3.0, z_score * 1.5)
            interpretation = "Within normal range"
        elif tortuosity <= 2.0:
            score = 4.0 + (tortuosity - 1.5) / 0.5 * 2.0
            interpretation = "Mild airway distortion"
        elif tortuosity <= 2.5:
            score = 7.0 + (tortuosity - 2.0) / 0.5 * 1.0
            interpretation = "Moderate airway distortion"
        else:
            score = 9.0 + min(1.0, (tortuosity - 2.5) / 0.5)
            interpretation = "Severe airway distortion"

        if self.verbose:
            print(f"  Mean Tortuosity: {tortuosity:.3f}")
            print(f"  Reference range: {ref['min']:.2f} - {ref['max']:.2f}")
            print(f"  Score: {score:.1f}/10")
            print(f"  {interpretation}")

        return {
            'raw_score': score,
            'value': tortuosity,
            'interpretation': interpretation,
            'reference_range': (ref['min'], ref['max'])
        }

    def compute_generation_coverage_score(self):
        if self.verbose:
            print("\n[3/6] Computing Generation Coverage Score...")

        if not hasattr(self.analyzer, 'advanced_metrics'):
            raise ValueError("Run analyzer.compute_advanced_metrics() first")

        metrics = self.analyzer.advanced_metrics
        coverage = metrics.get('generation_coverage', np.nan)
        missing_gens = metrics.get('missing_generations', [])

        if np.isnan(coverage):
            if self.verbose:
                print("  ⚠ Generation coverage not available")
            return {'score': 5.0, 'value': np.nan, 'interpretation': 'Unknown'}

        ref = self.reference_values['generation_coverage']


        if coverage > 0.90:
            score = (1.0 - coverage) * 10.0
            interpretation = "Excellent generation coverage"
        elif coverage >= 0.80:
            score = 1.0 + (0.90 - coverage) / 0.10 * 2.0
            interpretation = "Good coverage with minor gaps"
        elif coverage >= 0.70:
            score = 4.0 + (0.80 - coverage) / 0.10 * 2.0
            interpretation = "Fair coverage with moderate gaps"
        elif coverage >= 0.50:
            score = 7.0 + (0.70 - coverage) / 0.20 * 1.0
            interpretation = "Poor coverage with significant gaps"
        else:
            score = 9.0 + min(1.0, (0.50 - coverage) / 0.30)
            interpretation = "Very poor coverage - severe airway loss"

        if self.verbose:
            print(f"  Generation Coverage: {coverage:.1%}")
            print(f"  Missing generations: {len(missing_gens)}")
            if len(missing_gens) > 0 and len(missing_gens) <= 10:
                print(f"  Missing: {missing_gens}")
            print(f"  Score: {score:.1f}/10")
            print(f"  {interpretation}")

        return {
            'raw_score': score,
            'value': coverage,
            'missing_count': len(missing_gens),
            'interpretation': interpretation,
            'reference_range': (ref['min'], ref['max'])
        }

    def compute_airway_volume_score(self):
        if self.verbose:
            print("\n[3b/6] Computing Airway Volume Score...")

        if not hasattr(self.analyzer, 'advanced_metrics'):
            raise ValueError("Run analyzer.compute_advanced_metrics() first")

        metrics = self.analyzer.advanced_metrics
        volume = metrics.get('total_volume_mm3', np.nan)

        if np.isnan(volume):
            if self.verbose:
                print("  ⚠ Volume not available")
            return {'raw_score': 5.0, 'value': np.nan, 'interpretation': 'Unknown'}


        if volume > 30000:
            score = 0.0
            interpretation = "Excellent airway volume - well preserved"
        elif volume > 20000:
            score = 3.0 * (30000 - volume) / 10000
            interpretation = "Good airway volume"
        elif volume > 15000:
            score = 3.0 + 3.0 * (20000 - volume) / 5000
            interpretation = "Moderate airway volume reduction"
        elif volume > 10000:
            score = 6.0 + 2.0 * (15000 - volume) / 5000
            interpretation = "Significant airway volume loss"
        else:
            score = 8.0 + min(2.0, (10000 - volume) / 5000)
            interpretation = "Severe airway volume loss"

        score = min(10.0, max(0.0, score))

        if self.verbose:
            print(f"  Total Volume: {volume:.2f} mm³")
            print(f"  Score: {score:.1f}/10")
            print(f"  {interpretation}")

        return {
            'raw_score': score,
            'value': volume,
            'interpretation': interpretation,
            'reference_range': (15000, 30000)
        }

    def compute_symmetry_score(self):
        if self.verbose:
            print("\n[4/6] Computing Symmetry Score...")

        if not hasattr(self.analyzer, 'advanced_metrics'):
            raise ValueError("Run analyzer.compute_advanced_metrics() first")

        metrics = self.analyzer.advanced_metrics
        symmetry = metrics.get('symmetry_index', np.nan)

        if np.isnan(symmetry):
            if self.verbose:
                print("  ⚠ Symmetry index not available")
            return {'score': 5.0, 'value': np.nan, 'interpretation': 'Unknown'}

        ref = self.reference_values['symmetry_index']


        if symmetry > 0.90:
            score = (1.0 - symmetry) * 10.0
            interpretation = "Highly symmetric - bilateral disease pattern"
        elif symmetry >= 0.80:
            score = 1.0 + (0.90 - symmetry) / 0.10 * 2.0
            interpretation = "Good symmetry with minor differences"
        elif symmetry >= 0.70:
            score = 4.0 + (0.80 - symmetry) / 0.10 * 1.0
            interpretation = "Fair symmetry"
        elif symmetry >= 0.50:
            score = 6.0 + (0.70 - symmetry) / 0.20 * 1.0
            interpretation = "Moderate asymmetry - possible focal disease"
        else:
            score = 8.0 + min(2.0, (0.50 - symmetry) / 0.30 * 2.0)
            interpretation = "Severe asymmetry - unilateral predominance"

        if self.verbose:
            print(f"  Symmetry Index: {symmetry:.3f}")
            print(f"  Reference range: {ref['min']:.2f} - {ref['max']:.2f}")
            print(f"  Score: {score:.1f}/10")
            print(f"  {interpretation}")

        return {
            'raw_score': score,
            'value': symmetry,
            'interpretation': interpretation,
            'reference_range': (ref['min'], ref['max'])
        }

    def compute_volume_distribution_score(self):
        if self.verbose:
            print("\n[5/6] Computing Volume Distribution Score...")

        if not hasattr(self.analyzer, 'advanced_metrics'):
            raise ValueError("Run analyzer.compute_advanced_metrics() first")

        metrics = self.analyzer.advanced_metrics
        vol_per_gen = metrics.get('volume_per_generation', {})

        if len(vol_per_gen) < 3:
            if self.verbose:
                print("  ⚠ Insufficient generation data")
            return {'score': 5.0, 'value': 0, 'interpretation': 'Insufficient data'}

        generations = sorted(vol_per_gen.keys())
        volumes = [vol_per_gen[g] for g in generations]


        decreasing_count = sum(1 for i in range(len(volumes)-1) if volumes[i] > volumes[i+1])
        decreasing_ratio = decreasing_count / (len(volumes) - 1) if len(volumes) > 1 else 0

        abrupt_drops = 0
        for i in range(len(volumes) - 1):
            if volumes[i] > 0:
                drop_ratio = (volumes[i] - volumes[i+1]) / volumes[i]
                if drop_ratio > 0.80:
                    abrupt_drops += 1

        if len(volumes) >= 5:
            early_vols = volumes[:5]
            cv = np.std(early_vols) / np.mean(early_vols) if np.mean(early_vols) > 0 else 0
        else:
            cv = 0

        score = 0.0
        interpretation_parts = []

        if decreasing_ratio < 0.6:
            score += 3.0
            interpretation_parts.append("erratic volume pattern")
        elif decreasing_ratio < 0.8:
            score += 1.5
            interpretation_parts.append("irregular decrease")

        if abrupt_drops > 2:
            score += 4.0
            interpretation_parts.append(f"multiple abrupt volume drops ({abrupt_drops})")
        elif abrupt_drops > 0:
            score += 2.0
            interpretation_parts.append(f"abrupt volume drop detected")

        if cv > 1.5:
            score += 3.0
            interpretation_parts.append("high volume variability")
        elif cv > 1.0:
            score += 1.5
            interpretation_parts.append("moderate volume variability")

        score = min(10.0, score)

        if len(interpretation_parts) == 0:
            interpretation = "Normal volume distribution pattern"
        else:
            interpretation = "Abnormal pattern: " + ", ".join(interpretation_parts)

        if self.verbose:
            print(f"  Generations analyzed: {len(generations)}")
            print(f"  Decreasing ratio: {decreasing_ratio:.2f}")
            print(f"  Abrupt drops: {abrupt_drops}")
            print(f"  Coefficient of variation: {cv:.2f}")
            print(f"  Score: {score:.1f}/10")
            print(f"  {interpretation}")

        return {
            'raw_score': score,
            'value': decreasing_ratio,
            'abrupt_drops': abrupt_drops,
            'cv': cv,
            'interpretation': interpretation
        }

    def compute_tapering_score(self):
        if self.verbose:
            print("\n[6/6] Computing Tapering Score...")

        if not hasattr(self.analyzer, 'tapering_ratios_df') or self.analyzer.tapering_ratios_df is None:
            if self.verbose:
                print("  ⚠ Tapering ratios not available")
            return {'score': 5.0, 'value': np.nan, 'interpretation': 'Unknown'}

        if len(self.analyzer.tapering_ratios_df) == 0:
            if self.verbose:
                print("  ⚠ No tapering ratios computed")
            return {'score': 5.0, 'value': np.nan, 'interpretation': 'Unknown'}

        ratios = self.analyzer.tapering_ratios_df['diameter_ratio'].values
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)

        ref = self.reference_values['tapering_ratio']
        weibel_theoretical = 2**(-1/3)


        deviation = abs(mean_ratio - weibel_theoretical)

        if deviation < 0.05:
            score = deviation / 0.05 * 2.0
            interpretation = "Normal tapering pattern"
        elif deviation < 0.10:
            score = 2.0 + (deviation - 0.05) / 0.05 * 2.0
            interpretation = "Mild tapering abnormality"
        elif deviation < 0.15:
            score = 5.0 + (deviation - 0.10) / 0.05 * 2.0
            interpretation = "Moderate tapering abnormality"
        else:
            score = 8.0 + min(2.0, (deviation - 0.15) / 0.10 * 2.0)
            interpretation = "Severe tapering abnormality"

        if std_ratio > 0.15:
            score += 1.0
            interpretation += " with high variability"

        score = min(10.0, score)

        if self.verbose:
            print(f"  Mean tapering ratio: {mean_ratio:.3f}")
            print(f"  Weibel theoretical: {weibel_theoretical:.3f}")
            print(f"  Deviation: {deviation:.3f}")
            print(f"  Std deviation: {std_ratio:.3f}")
            print(f"  Score: {score:.1f}/10")
            print(f"  {interpretation}")

        return {
            'raw_score': score,
            'value': mean_ratio,
            'deviation': deviation,
            'std': std_ratio,
            'interpretation': interpretation,
            'reference': weibel_theoretical
        }

    def compute_peripheral_density_score(self):
        if self.verbose:
            print("\n[7/9] Computing Peripheral Branch Density Score...")

        if not hasattr(self.analyzer, 'advanced_metrics'):
            raise ValueError("Run analyzer.compute_advanced_metrics() first")

        metrics = self.analyzer.advanced_metrics
        density = metrics.get('peripheral_branch_density', np.nan)

        if np.isnan(density):
            if self.verbose:
                print("  ⚠ Peripheral density not available")
            return {'raw_score': 5.0, 'value': np.nan, 'interpretation': 'Unknown'}


        if density < 0.03:
            score = 10.0
            interpretation = "Severe peripheral airway loss"
        elif density < 0.05:
            score = 7.0 + (0.05 - density) / 0.02 * 3.0
            interpretation = "Moderate to severe peripheral loss"
        elif density < 0.08:
            score = 4.0 + (0.08 - density) / 0.03 * 3.0
            interpretation = "Mild peripheral airway loss"
        elif density <= 0.15:
            score = max(0.0, 4.0 - (density - 0.08) / 0.07 * 4.0)
            interpretation = "Within normal range"
        else:
            score = 0.0
            interpretation = "Excellent peripheral preservation"

        if self.verbose:
            print(f"  Peripheral branch density: {density:.4f} branches/mm³")
            print(f"  Score: {score:.1f}/10")
            print(f"  {interpretation}")

        return {
            'raw_score': score,
            'value': density,
            'interpretation': interpretation
        }

    def compute_peripheral_volume_score(self):
        if self.verbose:
            print("\n[8/9] Computing Peripheral Volume/Diameter Score...")

        if not hasattr(self.analyzer, 'advanced_metrics'):
            raise ValueError("Run analyzer.compute_advanced_metrics() first")

        metrics = self.analyzer.advanced_metrics

        periph_diam = metrics.get('mean_peripheral_diameter_mm', np.nan)
        periph_vol = metrics.get('mean_peripheral_branch_volume_mm3', np.nan)
        periph_pct = metrics.get('peripheral_volume_percent', np.nan)

        scores = []
        details = []

        if not np.isnan(periph_diam):
            if periph_diam < 1.5:
                diam_score = 10.0
                details.append("Severe diameter reduction")
            elif periph_diam < 2.0:
                diam_score = 6.0 + (2.0 - periph_diam) / 0.5 * 4.0
                details.append("Moderate diameter reduction")
            elif periph_diam < 2.5:
                diam_score = 3.0 + (2.5 - periph_diam) / 0.5 * 3.0
                details.append("Mild diameter reduction")
            else:
                diam_score = max(0.0, 3.0 - (periph_diam - 2.5) / 1.0 * 3.0)
                details.append("Normal diameter")
            scores.append(diam_score)

        if not np.isnan(periph_vol):
            if periph_vol < 10:
                vol_score = 10.0
                details.append("Severe volume loss")
            elif periph_vol < 20:
                vol_score = 6.0 + (20 - periph_vol) / 10 * 4.0
                details.append("Moderate volume loss")
            elif periph_vol < 30:
                vol_score = 3.0 + (30 - periph_vol) / 10 * 3.0
                details.append("Mild volume reduction")
            else:
                vol_score = max(0.0, 3.0 - (periph_vol - 30) / 30 * 3.0)
                details.append("Good volume")
            scores.append(vol_score)

        if not np.isnan(periph_pct):
            if periph_pct < 10:
                pct_score = 10.0
            elif periph_pct < 15:
                pct_score = 6.0 + (15 - periph_pct) / 5 * 4.0
            elif periph_pct < 20:
                pct_score = 3.0 + (20 - periph_pct) / 5 * 3.0
            else:
                pct_score = max(0.0, 3.0 - (periph_pct - 20) / 10 * 3.0)
            scores.append(pct_score)

        if len(scores) == 0:
            if self.verbose:
                print("  ⚠ No peripheral volume metrics available")
            return {'raw_score': 5.0, 'value': np.nan, 'interpretation': 'Unknown'}

        final_score = np.mean(scores)
        interpretation = "; ".join(details) if details else "Unknown"

        if self.verbose:
            print(f"  Mean peripheral diameter: {periph_diam:.2f} mm")
            print(f"  Mean peripheral branch volume: {periph_vol:.2f} mm³")
            print(f"  Peripheral volume %: {periph_pct:.1f}%")
            print(f"  Score: {final_score:.1f}/10")
            print(f"  {interpretation}")

        return {
            'raw_score': final_score,
            'value': {
                'diameter': periph_diam,
                'branch_volume': periph_vol,
                'volume_percent': periph_pct
            },
            'interpretation': interpretation
        }

    def load_parenchymal_metrics(self, parenchymal_data):

        self.parenchymal_metrics = parenchymal_data
        if self.verbose:
            print(f"\n✓ Parenchymal metrics loaded:")
            print(f"  Mean lung density: {parenchymal_data.get('mean_lung_density_HU', 'N/A')} HU")
            print(f"  Histogram entropy: {parenchymal_data.get('histogram_entropy', 'N/A')}")

    def compute_parenchymal_entropy_score(self):
        if self.verbose:
            print("\n[9/11] Computing Parenchymal Entropy Score...")

        if self.parenchymal_metrics is None:
            if self.verbose:
                print("  ⚠ Parenchymal metrics not loaded")
            return {'raw_score': 5.0, 'value': np.nan, 'interpretation': 'Not available'}

        entropy = self.parenchymal_metrics.get('histogram_entropy', np.nan)

        if np.isnan(entropy):
            if self.verbose:
                print("  ⚠ Histogram entropy not available")
            return {'raw_score': 5.0, 'value': np.nan, 'interpretation': 'Unknown'}


        if entropy < 4.0:
            score = 0.0
            interpretation = "Very homogeneous (unusual)"
        elif entropy < 5.0:
            score = max(0.0, (5.0 - entropy) / 1.0 * 2.0)
            interpretation = "Normal homogeneity"
        elif entropy < 5.5:
            score = 2.0 + (entropy - 5.0) / 0.5 * 2.0
            interpretation = "Mild texture heterogeneity"
        elif entropy < 6.0:
            score = 4.0 + (entropy - 5.5) / 0.5 * 2.0
            interpretation = "Moderate texture heterogeneity"
        elif entropy < 6.5:
            score = 6.0 + (entropy - 6.0) / 0.5 * 2.0
            interpretation = "Marked texture heterogeneity"
        else:
            score = 8.0 + min(2.0, (entropy - 6.5) / 0.5 * 2.0)
            interpretation = "Severe texture heterogeneity"

        if self.verbose:
            print(f"  Histogram entropy: {entropy:.3f}")
            print(f"  Score: {score:.1f}/10")
            print(f"  {interpretation}")

        return {
            'raw_score': score,
            'value': entropy,
            'interpretation': interpretation
        }

    def compute_parenchymal_density_score(self):
        if self.verbose:
            print("\n[10/11] Computing Parenchymal Density Score...")

        if self.parenchymal_metrics is None:
            if self.verbose:
                print("  ⚠ Parenchymal metrics not loaded")
            return {'raw_score': 5.0, 'value': np.nan, 'interpretation': 'Not available'}

        density = self.parenchymal_metrics.get('mean_lung_density_HU', np.nan)

        if np.isnan(density):
            if self.verbose:
                print("  ⚠ Mean lung density not available")
            return {'raw_score': 5.0, 'value': np.nan, 'interpretation': 'Unknown'}


        if density < -900:
            score = 0.0
            interpretation = "Very low density (emphysematous)"
        elif density < -850:
            score = max(0.0, (-850 - density) / 50 * 1.0)
            interpretation = "Low density"
        elif density < -750:
            score = 1.0 + (-750 - density) / 100 * 2.0
            interpretation = "Normal density"
        elif density < -700:
            score = 3.0 + (-700 - density) / 50 * 2.0
            interpretation = "Mildly increased density"
        elif density < -650:
            score = 5.0 + (-650 - density) / 50 * 2.0
            interpretation = "Moderately increased density"
        elif density < -600:
            score = 7.0 + (-600 - density) / 50 * 2.0
            interpretation = "Markedly increased density"
        else:
            score = 9.0 + min(1.0, (-600 - density) / 50)
            interpretation = "Severe parenchymal density increase"

        if self.verbose:
            print(f"  Mean lung density: {density:.1f} HU")
            print(f"  Score: {score:.1f}/10")
            print(f"  {interpretation}")

        return {
            'raw_score': score,
            'value': density,
            'interpretation': interpretation
        }

    def compute_overall_score(self, score_type='airway_only'):

        if self.verbose:
            print("\n" + "="*70)
            print("COMPUTING OVERALL FIBROSIS SCORE")
            if score_type == 'both':
                print("Computing BOTH airway_only and combined scores")
            else:
                print(f"Score type: {score_type.upper()}")
            print("="*70)

        self.component_scores = {
            'peripheral_density': self.compute_peripheral_density_score(),
            'peripheral_volume': self.compute_peripheral_volume_score(),
            'pc_ratio': self.compute_pc_ratio_score(),
            'tortuosity': self.compute_tortuosity_score(),
            'symmetry': self.compute_symmetry_score(),
            'airway_volume': self.compute_airway_volume_score(),
            'generation_coverage': self.compute_generation_coverage_score(),
            'volume_distribution': self.compute_volume_distribution_score(),
            'tapering': self.compute_tapering_score()
        }

        if self.parenchymal_metrics is not None:
            self.component_scores['parenchymal_entropy'] = self.compute_parenchymal_entropy_score()
            self.component_scores['parenchymal_density'] = self.compute_parenchymal_density_score()

        compute_airway = score_type in ['airway_only', 'both']
        compute_combined = score_type in ['combined', 'both']

        if compute_airway:
            total_score = 0.0
            available_weight = 0.0

            for component, weight in self.weights_airway_only.items():
                if weight == 0.0:
                    continue

                comp_data = self.component_scores.get(component)
                if comp_data is None:
                    continue

                raw_score = comp_data.get('raw_score', 5.0)
                value = comp_data.get('value', np.nan)

                is_valid = False
                if isinstance(value, dict):
                    is_valid = any(not np.isnan(v) for v in value.values() if isinstance(v, (int, float)))
                else:
                    is_valid = not np.isnan(value)

                if is_valid or component == 'volume_distribution':
                    total_score += raw_score * weight
                    available_weight += weight

            if available_weight > 0:
                self.fibrosis_score_airway_only = (total_score / available_weight) * 10.0
            else:
                self.fibrosis_score_airway_only = 50.0

            self.severity_stage_airway_only = self._classify_severity(self.fibrosis_score_airway_only)
            self.confidence_airway_only = available_weight

            if self.verbose:
                print(f"\n{'='*70}")
                print("AIRWAY-ONLY SCORE (Opzione 1)")
                print(f"{'='*70}")
                print(f"Fibrosis Score: {self.fibrosis_score_airway_only:.1f}/100")
                print(f"Severity Stage: {self.severity_stage_airway_only}")
                print(f"Confidence: {self.confidence_airway_only:.0%}")

        if compute_combined:
            if self.parenchymal_metrics is None:
                if self.verbose:
                    print(f"\n⚠ Cannot compute COMBINED score: parenchymal metrics not loaded")
                self.fibrosis_score_combined = None
                self.severity_stage_combined = None
                self.confidence_combined = None
            else:
                total_score = 0.0
                available_weight = 0.0

                for component, weight in self.weights_combined.items():
                    if weight == 0.0:
                        continue

                    comp_data = self.component_scores.get(component)
                    if comp_data is None:
                        continue

                    raw_score = comp_data.get('raw_score', 5.0)
                    value = comp_data.get('value', np.nan)

                    is_valid = False
                    if isinstance(value, dict):
                        is_valid = any(not np.isnan(v) for v in value.values() if isinstance(v, (int, float)))
                    else:
                        is_valid = not np.isnan(value)

                    if is_valid or component == 'volume_distribution':
                        total_score += raw_score * weight
                        available_weight += weight

                if available_weight > 0:
                    self.fibrosis_score_combined = (total_score / available_weight) * 10.0
                else:
                    self.fibrosis_score_combined = 50.0

                self.severity_stage_combined = self._classify_severity(self.fibrosis_score_combined)
                self.confidence_combined = available_weight

                if self.verbose:
                    print(f"\n{'='*70}")
                    print("COMBINED SCORE (Opzione 2) - RECOMMENDED")
                    print(f"{'='*70}")
                    print(f"Fibrosis Score: {self.fibrosis_score_combined:.1f}/100")
                    print(f"Severity Stage: {self.severity_stage_combined}")
                    print(f"Confidence: {self.confidence_combined:.0%}")

        if score_type == 'airway_only':
            self.fibrosis_score = self.fibrosis_score_airway_only
            self.severity_stage = self.severity_stage_airway_only
            self.confidence = self.confidence_airway_only
        elif score_type == 'combined' and self.fibrosis_score_combined is not None:
            self.fibrosis_score = self.fibrosis_score_combined
            self.severity_stage = self.severity_stage_combined
            self.confidence = self.confidence_combined
        elif score_type == 'both':
            if self.fibrosis_score_combined is not None:
                self.fibrosis_score = self.fibrosis_score_combined
                self.severity_stage = self.severity_stage_combined
                self.confidence = self.confidence_combined
            else:
                self.fibrosis_score = self.fibrosis_score_airway_only
                self.severity_stage = self.severity_stage_airway_only
                self.confidence = self.confidence_airway_only

        return self.fibrosis_score, self.severity_stage, self.confidence

    def _classify_severity(self, score):
        if score < 20:
            return "Minimal/No fibrosis"
        elif score < 35:
            return "Mild fibrosis"
        elif score < 50:
            return "Moderate fibrosis"
        elif score < 70:
            return "Moderate-severe fibrosis (UIP pattern)"
        else:
            return "Severe/Advanced fibrosis"

    def generate_report(self, output_dir):
        if self.fibrosis_score is None:
            raise ValueError("Run compute_overall_score() first")

        os.makedirs(output_dir, exist_ok=True)

        report_path = os.path.join(output_dir, "fibrosis_assessment_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write(" "*15 + "PULMONARY FIBROSIS ASSESSMENT REPORT\n")
            f.write("="*70 + "\n\n")

            f.write(f"SCORING METHODS\n")
            f.write("-"*70 + "\n")

            if self.fibrosis_score_airway_only is not None:
                f.write(f"\n1. AIRWAY-ONLY SCORE (Opzione 1)\n")
                f.write(f"   Pure airway morphometry\n")
                f.write(f"   Score: {self.fibrosis_score_airway_only:.1f}/100\n")
                f.write(f"   Stage: {self.severity_stage_airway_only}\n")
                f.write(f"   Confidence: {self.confidence_airway_only:.0%}\n")

            if self.fibrosis_score_combined is not None:
                f.write(f"\n2. COMBINED SCORE (Opzione 2) - RECOMMENDED\n")
                f.write(f"   Airway + Parenchymal metrics\n")
                f.write(f"   Score: {self.fibrosis_score_combined:.1f}/100\n")
                f.write(f"   Stage: {self.severity_stage_combined}\n")
                f.write(f"   Confidence: {self.confidence_combined:.0%}\n")
            else:
                f.write(f"\n2. COMBINED SCORE: Not available (no parenchymal metrics)\n")

            f.write(f"\n" + "-"*70 + "\n")
            f.write(f"DEFAULT SCORE USED FOR INTERPRETATION:\n")
            f.write(f"   Score: {self.fibrosis_score:.1f}/100\n")
            f.write(f"   Stage: {self.severity_stage}\n")
            f.write(f"   Confidence: {self.confidence:.0%}\n\n")

            f.write("="*70 + "\n")
            f.write("DETAILED COMPONENT ANALYSIS\n")
            f.write("="*70 + "\n\n")

            for component, weight in self.weights.items():
                comp_data = self.component_scores.get(component, {})
                f.write(f"{component.replace('_', ' ').upper()}\n")
                f.write("-"*70 + "\n")
                f.write(f"Weight in assessment: {weight:.0%}\n")
                
                # Format raw_score safely
                raw_score = comp_data.get('raw_score', 'N/A')
                if isinstance(raw_score, (int, float)):
                    f.write(f"Raw score: {raw_score:.1f}/10\n")
                else:
                    f.write(f"Raw score: {raw_score}/10\n")
                
                f.write(f"Measured value: {comp_data.get('value', 'N/A')}\n")

                if 'reference_range' in comp_data:
                    ref_range = comp_data['reference_range']
                    try:
                        f.write(f"Reference range: {ref_range[0]:.2f} - {ref_range[1]:.2f}\n")
                    except (TypeError, ValueError):
                        f.write(f"Reference range: {ref_range}\n")

                f.write(f"Interpretation: {comp_data.get('interpretation', 'N/A')}\n\n")

            f.write("="*70 + "\n")
            f.write("CLINICAL INTERPRETATION\n")
            f.write("="*70 + "\n\n")

            self._write_clinical_interpretation(f)

            f.write("\n" + "="*70 + "\n")
            f.write("DISCLAIMER\n")
            f.write("="*70 + "\n")
            f.write("This is an AUTOMATED ASSESSMENT based on quantitative airway morphometry.\n")
            f.write("It does NOT replace:\n")
            f.write("- Clinical evaluation by a pulmonologist\n")
            f.write("- Complete CT image review by a radiologist\n")
            f.write("- Pulmonary function tests (PFT)\n")
            f.write("- Histopathological diagnosis when indicated\n")
            f.write("- Multidisciplinary discussion (MDD)\n\n")
            f.write("Always correlate with complete clinical picture, imaging patterns,\n")
            f.write("and other diagnostic findings.\n")

        print(f"✓ Report saved: {report_path}")

        json_path = os.path.join(output_dir, "fibrosis_assessment.json")
        report_data = {
            'overall': {
                'fibrosis_score': float(self.fibrosis_score),
                'stage': self.severity_stage,
                'confidence': float(self.confidence)
            },
            'scoring_methods': {}
        }

        if self.fibrosis_score_airway_only is not None:
            report_data['scoring_methods']['airway_only'] = {
                'fibrosis_score': float(self.fibrosis_score_airway_only),
                'stage': self.severity_stage_airway_only,
                'confidence': float(self.confidence_airway_only),
                'description': 'Pure airway morphometry (Opzione 1)'
            }

        if self.fibrosis_score_combined is not None:
            report_data['scoring_methods']['combined'] = {
                'fibrosis_score': float(self.fibrosis_score_combined),
                'stage': self.severity_stage_combined,
                'confidence': float(self.confidence_combined),
                'description': 'Airway + Parenchymal (Opzione 2) - RECOMMENDED'
            }

        report_data['components'] = {}

        for component, comp_data in self.component_scores.items():
            weight_airway = self.weights_airway_only.get(component, 0.0)
            weight_combined = self.weights_combined.get(component, 0.0)

            report_data['components'][component] = {
                'raw_score': float(comp_data.get('raw_score', np.nan)),
                'weighted_score_airway': float(comp_data.get('raw_score', 5.0) * weight_airway * 10.0),
                'weighted_score_combined': float(comp_data.get('raw_score', 5.0) * weight_combined * 10.0),
                'value': float(comp_data.get('value', np.nan)) if not isinstance(comp_data.get('value'), (list, dict)) else str(comp_data.get('value')),
                'interpretation': comp_data.get('interpretation', 'N/A'),
                'weight_airway': float(weight_airway),
                'weight_combined': float(weight_combined)
            }

        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"✓ JSON saved: {json_path}")

        self.plot_assessment(save_path=os.path.join(output_dir, "fibrosis_assessment_visualization.png"))

        return report_data

    def _write_clinical_interpretation(self, file_handle):
        f = file_handle
        score = self.fibrosis_score

        if score < 20:
            f.write("No significant fibrotic changes detected based on airway morphometry.\n")
            f.write("Airways appear structurally normal with preserved peripheral airways.\n")
            f.write("\nRecommendation:\n")
            f.write("- No specific follow-up required for fibrosis\n")
            f.write("- Standard screening if risk factors present\n")

        elif score < 35:
            f.write("Minimal fibrotic changes detected.\n")
            f.write("Early peripheral airway involvement with preserved central airways.\n")

            pc_ratio = self.component_scores.get('pc_ratio', {}).get('raw_score', 0)
            if pc_ratio > 5:
                f.write("Note: Peripheral airway loss is the primary finding.\n")

            f.write("\nRecommendation:\n")
            f.write("- Monitor for progression with follow-up CT in 6-12 months\n")
            f.write("- Baseline pulmonary function tests\n")
            f.write("- Clinical correlation advised\n")

        elif score < 50:
            f.write("Mild fibrosis with measurable structural airway changes.\n")
            f.write("Bilateral peripheral airway involvement detected.\n")

            components_high = [k for k, v in self.component_scores.items() 
                             if v.get('raw_score', 0) > 6]
            if components_high:
                f.write(f"Prominent findings: {', '.join([k.replace('_', ' ') for k in components_high])}\n")

            f.write("\nRecommendation:\n")
            f.write("- Pulmonologist consultation recommended\n")
            f.write("- Complete pulmonary function tests (PFT) including DLCO\n")
            f.write("- Follow-up CT in 3-6 months\n")
            f.write("- Consider autoimmune workup if clinically indicated\n")

        elif score < 70:
            f.write("Moderate fibrosis with features suggesting UIP pattern.\n")
            f.write("Significant peripheral airway loss and structural remodeling.\n")

            pc_ratio = self.component_scores.get('pc_ratio', {}).get('value', 0.5)
            if pc_ratio < 0.25:
                f.write("ALERT: Severe peripheral airway loss detected (P/C < 0.25)\n")

            coverage = self.component_scores.get('generation_coverage', {}).get('value', 1.0)
            if coverage < 0.70:
                f.write("ALERT: Significant airway generation loss (>30% missing)\n")

            f.write("\nRecommendation:\n")
            f.write("- URGENT pulmonologist referral\n")
            f.write("- Complete diagnostic workup:\n")
            f.write("  * High-resolution CT with complete evaluation\n")
            f.write("  * Comprehensive PFTs with 6-minute walk test\n")
            f.write("  * Consider bronchoscopy ± biopsy\n")
            f.write("- Multidisciplinary discussion recommended\n")
            f.write("- Consider antifibrotic therapy evaluation\n")

        else:
            f.write("Severe/advanced fibrosis detected.\n")
            f.write("Extensive airway destruction with marked peripheral loss.\n")
            f.write("Pattern consistent with advanced fibrotic lung disease.\n")

            f.write("\nCRITICAL FINDINGS:\n")

            for component, comp_data in self.component_scores.items():
                if comp_data.get('raw_score', 0) >= 8:
                    f.write(f"- {component.replace('_', ' ').title()}: {comp_data.get('interpretation')}\n")

            f.write("\nRecommendation:\n")
            f.write("- URGENT pulmonary specialist evaluation\n")
            f.write("- Comprehensive assessment:\n")
            f.write("  * Complete PFT battery\n")
            f.write("  * Echocardiography (evaluate pulmonary hypertension)\n")
            f.write("  * 6-minute walk test with oxygen saturation\n")
            f.write("  * Consider right heart catheterization\n")
            f.write("- Multidisciplinary team discussion MANDATORY\n")
            f.write("- Antifibrotic therapy consideration\n")
            f.write("- Lung transplant evaluation if appropriate\n")
            f.write("- Palliative care consultation if end-stage\n")

    def plot_assessment(self, save_path=None):
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, :])
        self._plot_score_gauge(ax1)

        ax2 = fig.add_subplot(gs[1, 0], projection='polar')
        self._plot_radar_chart(ax2)

        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_component_bars(ax3)

        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_reference_comparison(ax4)

        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_severity_distribution(ax5)

        ax6 = fig.add_subplot(gs[2, 1:])
        self._plot_metrics_table(ax6)

        plt.suptitle('Pulmonary Fibrosis Assessment - Airway Morphometry Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved: {save_path}")

        plt.show()

    def _plot_score_gauge(self, ax):
        ax.clear()
        ax.axis('off')

        score = self.fibrosis_score

        theta = np.linspace(np.pi, 0, 100)

        colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b']
        boundaries = [0, 20, 35, 50, 70, 100]

        for i in range(len(boundaries)-1):
            start_angle = np.pi * (1 - boundaries[i]/100)
            end_angle = np.pi * (1 - boundaries[i+1]/100)
            theta_segment = np.linspace(start_angle, end_angle, 20)

            x = np.concatenate([[0], np.cos(theta_segment), [0]])
            y = np.concatenate([[0], np.sin(theta_segment), [0]])

            ax.fill(x, y, color=colors[i], alpha=0.6, edgecolor='black', linewidth=2)

        needle_angle = np.pi * (1 - score/100)
        needle_x = [0, 0.9 * np.cos(needle_angle)]
        needle_y = [0, 0.9 * np.sin(needle_angle)]
        ax.plot(needle_x, needle_y, 'k-', linewidth=4)
        ax.plot(0, 0, 'ko', markersize=15)

        ax.text(0, -0.3, f'{score:.1f}', ha='center', va='top', 
               fontsize=36, fontweight='bold')
        ax.text(0, -0.45, f'{self.severity_stage}', ha='center', va='top',
               fontsize=14, style='italic')
        ax.text(0, -0.6, f'Confidence: {self.confidence:.0%}', ha='center', va='top',
               fontsize=11)

        ax.text(-0.95, 0, 'Minimal', ha='right', va='center', fontsize=9)
        ax.text(0.95, 0, 'Severe', ha='left', va='center', fontsize=9)

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.7, 1.2)
        ax.set_aspect('equal')

    def _plot_radar_chart(self, ax):
        categories = [k.replace('_', ' ').title() for k in self.weights.keys()]
        scores = [self.component_scores.get(k, {}).get('raw_score', 5.0) for k in self.weights.keys()]

        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        scores += scores[:1]
        angles += angles[:1]

        ax.plot(angles, scores, 'o-', linewidth=2, color='#3498db', label='Patient')
        ax.fill(angles, scores, alpha=0.25, color='#3498db')

        reference_scores = [2.0] * (N + 1)
        ax.plot(angles, reference_scores, '--', linewidth=1.5, color='#2ecc71',
               label='Normal range', alpha=0.7)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=9)
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(['2', '4', '6', '8', '10'], size=8)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        ax.set_title('Component Scores\n(0-10 scale)', fontsize=11, pad=20)

    def _plot_component_bars(self, ax):
        components = list(self.weights.keys())
        contributions = [self.component_scores.get(k, {}).get('raw_score', 5.0) * 
                        self.weights[k] * 10 for k in components]

        colors_map = {
            'pc_ratio': '#e74c3c',
            'tortuosity': '#e67e22',
            'generation_coverage': '#f39c12',
            'symmetry': '#3498db',
            'volume_distribution': '#9b59b6',
            'tapering': '#1abc9c'
        }

        colors = [colors_map.get(k, '#95a5a6') for k in components]
        labels = [k.replace('_', ' ').title() for k in components]

        bars = ax.barh(labels, contributions, color=colors, edgecolor='black', linewidth=1)

        for i, (bar, val) in enumerate(zip(bars, contributions)):
            ax.text(val + 0.5, i, f'{val:.1f}', va='center', fontsize=9)

        ax.set_xlabel('Contribution to Total Score', fontsize=10)
        ax.set_title('Component Contributions', fontsize=11, fontweight='bold')
        ax.set_xlim(0, max(contributions) * 1.15)
        ax.grid(axis='x', alpha=0.3)

    def _plot_reference_comparison(self, ax):
        components_with_refs = {k: v for k, v in self.component_scores.items() 
                               if 'reference_range' in v}

        if not components_with_refs:
            ax.text(0.5, 0.5, 'No reference\ndata available', 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            return

        y_pos = np.arange(len(components_with_refs))
        labels = [k.replace('_', ' ').title() for k in components_with_refs.keys()]

        for i, (component, comp_data) in enumerate(components_with_refs.items()):
            ref_range = comp_data['reference_range']
            value = comp_data.get('value', np.nan)

            if np.isnan(value):
                continue

            ax.plot(ref_range, [i, i], 'k-', linewidth=8, alpha=0.2, solid_capstyle='round')

            color = '#2ecc71' if ref_range[0] <= value <= ref_range[1] else '#e74c3c'
            ax.plot(value, i, 'o', markersize=12, color=color, markeredgecolor='black', 
                   markeredgewidth=1.5, zorder=3)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Value', fontsize=10)
        ax.set_title('Measured vs Reference Range', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()

    def _plot_severity_distribution(self, ax):
        stages = ['Minimal/No', 'Mild', 'Moderate', 'Moderate-severe', 'Severe/Advanced']
        boundaries = [0, 20, 35, 50, 70, 100]
        colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b']

        for i in range(len(stages)):
            ax.barh(0, boundaries[i+1] - boundaries[i], left=boundaries[i],
                   color=colors[i], edgecolor='black', linewidth=1, alpha=0.7)

            mid = (boundaries[i] + boundaries[i+1]) / 2
            ax.text(mid, 0, stages[i], ha='center', va='center', 
                   fontsize=8, fontweight='bold')

        ax.plot(self.fibrosis_score, 0, 'v', markersize=15, color='blue',
               markeredgecolor='black', markeredgewidth=2, zorder=3)
        ax.text(self.fibrosis_score, -0.5, f'Patient\n{self.fibrosis_score:.1f}',
               ha='center', va='top', fontsize=9, fontweight='bold')

        ax.set_xlim(0, 100)
        ax.set_ylim(-0.7, 0.5)
        ax.set_xlabel('Fibrosis Score', fontsize=10)
        ax.set_title('Severity Classification', fontsize=11, fontweight='bold')
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    def _plot_metrics_table(self, ax):
        ax.axis('off')

        table_data = []
        table_data.append(['Overall Score', f'{self.fibrosis_score:.1f}/100', ''])
        table_data.append(['Severity Stage', self.severity_stage, ''])
        table_data.append(['Confidence', f'{self.confidence:.0%}', ''])
        table_data.append(['', '', ''])

        for component, comp_data in self.component_scores.items():
            name = component.replace('_', ' ').title()
            value = comp_data.get('value', 'N/A')

            if isinstance(value, float):
                if not np.isnan(value):
                    value_str = f'{value:.3f}'
                else:
                    value_str = 'N/A'
            else:
                value_str = str(value)

            score = comp_data.get('raw_score', 'N/A')
            if isinstance(score, float):
                score_str = f'{score:.1f}/10'
            else:
                score_str = str(score)

            table_data.append([name, value_str, score_str])

        table = ax.table(cellText=table_data,
                        colLabels=['Metric', 'Value', 'Score'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.4, 0.3, 0.3])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        for i in range(3):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        for i in range(1, len(table_data) + 1):
            if i == 4:
                for j in range(3):
                    table[(i, j)].set_facecolor('#ecf0f1')
            elif i < 4:
                for j in range(3):
                    table[(i, j)].set_facecolor('#e8f8f5')
                    table[(i, j)].set_text_props(weight='bold')
            else:
                for j in range(3):
                    table[(i, j)].set_facecolor('#ffffff' if i % 2 == 0 else '#f8f9fa')

        ax.set_title('Key Metrics Summary', fontsize=11, fontweight='bold', pad=10)


def integrate_fibrosis_scoring(analyzer, output_dir, parenchymal_metrics=None):


    print("\n" + "="*70)
    print(" "*15 + "PULMONARY FIBROSIS SCORING SYSTEM")
    print("="*70)

    if not hasattr(analyzer, 'advanced_metrics'):
        print("\nComputing advanced metrics...")
        try:
            analyzer.compute_advanced_metrics()
        except Exception as e:
            print(f"⚠ Warning: Could not compute advanced metrics: {e}")
            return None, None

    scorer = PulmonaryFibrosisScorer(analyzer, verbose=True)

    if parenchymal_metrics is not None:
        print(f"\n✓ Parenchymal metrics loaded")
        scorer.parenchymal_metrics = parenchymal_metrics
        score_type = 'both'
    else:
        print(f"\n⚠ No parenchymal metrics - computing airway_only score")
        score_type = 'airway_only'

    fibrosis_score, severity_stage, confidence = scorer.compute_overall_score(score_type=score_type)

    report = scorer.generate_report(output_dir)

    print(f"\n{'='*70}")
    print("FIBROSIS ASSESSMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Score: {fibrosis_score:.1f}/100")
    print(f"Stage: {severity_stage}")
    print(f"Confidence: {confidence:.0%}")
    print(f"\nResults saved to: {output_dir}")

    return scorer, report