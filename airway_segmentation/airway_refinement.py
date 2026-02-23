import numpy as np
import SimpleITK as sitk
from scipy.ndimage import (distance_transform_edt, binary_dilation, binary_erosion,
                           label, generate_binary_structure, binary_closing, gaussian_filter)
from skimage.morphology import skeletonize, ball, remove_small_holes, binary_opening
from skimage.filters import threshold_otsu, threshold_multiotsu
from collections import deque
import os


class EnhancedAirwayRefinementModule:


    def __init__(self, intensity_img, mask, spacing, verbose=True):
        self.img = intensity_img.astype(np.int16)
        self.mask = mask.astype(np.uint8)
        self.spacing = spacing
        self.verbose = verbose
        self.refined = None

        self.hu_thresholds = self._estimate_adaptive_thresholds()
        self.distance_transform = None

    def _estimate_adaptive_thresholds(self):
        if self.verbose:
            print("\n[Enhanced Refinement] Estimating adaptive HU thresholds...")

        flat = self.img[self.mask > 0].flatten()
        flat = flat[(flat > -1200) & (flat < 200)]

        if len(flat) < 100:
            return {'central': -850, 'intermediate': -750, 'peripheral': -650}

        try:
            thresholds = threshold_multiotsu(flat, classes=3)
            if len(thresholds) >= 2:
                t0, t1 = thresholds[0], thresholds[1]
                central_threshold = min(t0 + 100, -700)
                intermediate_threshold = min(t1 + 100, -550)
                peripheral_threshold = min(t1 + 120, -400)
            elif len(thresholds) == 1:
                t0 = thresholds[0]
                central_threshold = min(t0 + 100, -700)
                intermediate_threshold = -750
                peripheral_threshold = -650
            else:
                raise ValueError("No valid thresholds returned")
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Otsu failed ({e}), using relaxed defaults")
            central_threshold = -900
            intermediate_threshold = -700
            peripheral_threshold = -500

        thresholds_dict = {
            'central': central_threshold,
            'intermediate': intermediate_threshold,
            'peripheral': peripheral_threshold
        }

        if self.verbose:
            print(f"  Central airways: HU < {central_threshold:.1f}")
            print(f"  Intermediate: HU < {intermediate_threshold:.1f}")
            print(f"  Peripheral: HU < {peripheral_threshold:.1f}")

        return thresholds_dict

    def _compute_distance_transform(self):
        if self.distance_transform is None:
            if self.verbose:
                print("\n[Enhanced Refinement] Computing distance transform...")
            self.distance_transform = distance_transform_edt(
                self.mask == 0, 
                sampling=self.spacing
            )
        return self.distance_transform


    def remove_small_blobs(self, min_size_voxels=50, min_size_mm3=10):

        if self.verbose:
            print("\n[Anti-Blob] Removing small isolated blobs...")

        binary_mask = (self.mask > 0).astype(np.uint8)

        structure = generate_binary_structure(3, 3)
        labeled, num_components = label(binary_mask, structure=structure)

        voxel_volume = self.spacing[0] * self.spacing[1] * self.spacing[2]
        min_voxels = max(min_size_voxels, int(min_size_mm3 / voxel_volume))

        component_sizes = [(i, np.sum(labeled == i)) for i in range(1, num_components + 1)]
        component_sizes.sort(key=lambda x: x[1], reverse=True)

        if len(component_sizes) == 0:
            return self.mask

        main_id = component_sizes[0][0]
        main_size = component_sizes[0][1]

        cleaned = np.zeros_like(binary_mask)
        cleaned[labeled == main_id] = 1

        removed_count = 0
        removed_voxels = 0
        kept_small = 0

        for comp_id, size in component_sizes[1:]:
            if size >= min_voxels:
                cleaned[labeled == comp_id] = 1
                kept_small += 1
            else:
                removed_count += 1
                removed_voxels += size

        if self.verbose:
            print(f"  Main component: {main_size:,} voxels")
            print(f"  Small components kept: {kept_small}")
            print(f"  Removed blobs: {removed_count} ({removed_voxels:,} voxels)")

        self.mask = cleaned.astype(np.uint8)
        return self.mask

    def detect_and_remove_spurious_blobs(self, max_blob_distance_mm=15.0,
                                         max_elongation_ratio=3.0):

        if self.verbose:
            print(f"\n[Anti-Blob] Detecting spurious blobs (geometric analysis)...")

        binary_mask = (self.mask > 0).astype(np.uint8)
        structure = generate_binary_structure(3, 3)
        labeled, num_components = label(binary_mask, structure=structure)

        if num_components <= 1:
            if self.verbose:
                print("  Only one component, skipping")
            return self.mask

        component_sizes = [(i, np.sum(labeled == i)) for i in range(1, num_components + 1)]
        component_sizes.sort(key=lambda x: x[1], reverse=True)
        main_id = component_sizes[0][0]

        main_component = (labeled == main_id).astype(np.uint8)
        dist_from_main = distance_transform_edt(
            ~main_component.astype(bool),
            sampling=self.spacing
        )

        cleaned = main_component.copy()
        removed_count = 0
        removed_voxels = 0

        for comp_id, size in component_sizes[1:]:
            comp_mask = (labeled == comp_id)
            comp_coords = np.argwhere(comp_mask)

            z_extent = (np.max(comp_coords[:, 0]) - np.min(comp_coords[:, 0]) + 1) * self.spacing[2]
            y_extent = (np.max(comp_coords[:, 1]) - np.min(comp_coords[:, 1]) + 1) * self.spacing[1]
            x_extent = (np.max(comp_coords[:, 2]) - np.min(comp_coords[:, 2]) + 1) * self.spacing[0]

            extents = sorted([z_extent, y_extent, x_extent])
            elongation = extents[2] / extents[0] if extents[0] > 0 else 1.0

            min_distance = np.min(dist_from_main[comp_mask])

            hu_values = self.img[comp_mask]
            mean_hu = np.mean(hu_values)

            is_blob = (
                elongation < 1.5 and
                min_distance > max_blob_distance_mm and
                mean_hu > -650 and
                size < 15
            )

            if is_blob:
                removed_count += 1
                removed_voxels += size
                if self.verbose and removed_count <= 10:
                    print(f"    Removed blob {comp_id}: {size} voxels, "
                          f"dist={min_distance:.1f}mm, elongation={elongation:.2f}, HU={mean_hu:.1f}")
            else:
                cleaned[comp_mask] = 1

        if self.verbose:
            print(f"  Total spurious blobs removed: {removed_count} ({removed_voxels:,} voxels)")

        self.mask = cleaned.astype(np.uint8)
        return self.mask

    def morphological_tubular_smoothing(self):

        if self.verbose:
            print("\n[Anti-Blob] Applying tubular morphological smoothing...")

        binary_mask = (self.mask > 0).astype(np.uint8)

        opened = binary_opening(binary_mask, ball(1))

        smoothed = binary_closing(opened, ball(2))

        smoothed = binary_erosion(smoothed, ball(1))
        smoothed = binary_dilation(smoothed, ball(1))

        voxels_before = np.sum(binary_mask)
        voxels_after = np.sum(smoothed)

        if self.verbose:
            print(f"  Voxels before: {voxels_before:,}")
            print(f"  Voxels after: {voxels_after:,}")
            print(f"  Change: {voxels_after - voxels_before:+,} voxels")

        self.mask = smoothed.astype(np.uint8)
        return self.mask

    def skeleton_guided_reconstruction(self, max_reconstruction_mm=3.0):

        if self.verbose:
            print("\n[Anti-Blob] Skeleton-guided tubular reconstruction...")

        binary_mask = (self.mask > 0).astype(np.uint8)

        if self.verbose:
            print("  Computing skeleton...")
        skeleton = skeletonize(binary_mask)

        if np.sum(skeleton) < 10:
            if self.verbose:
                print("  Skeleton too small, skipping reconstruction")
            return self.mask

        dt = distance_transform_edt(binary_mask, sampling=self.spacing)

        skeleton_coords = np.argwhere(skeleton)
        reconstructed = np.zeros_like(binary_mask)

        max_radius_voxels = int(max_reconstruction_mm / np.mean(self.spacing))

        for coord in skeleton_coords:
            z, y, x = coord

            if (0 <= z < dt.shape[0] and
                0 <= y < dt.shape[1] and
                0 <= x < dt.shape[2]):
                local_radius = dt[z, y, x]
            else:
                local_radius = 1.0

            local_radius = min(local_radius, max_radius_voxels)
            radius_voxels = max(1, int(local_radius))

            for dz in range(-radius_voxels, radius_voxels + 1):
                for dy in range(-radius_voxels, radius_voxels + 1):
                    for dx in range(-radius_voxels, radius_voxels + 1):
                        if dz*dz + dy*dy + dx*dx <= radius_voxels*radius_voxels:
                            nz, ny, nx = z + dz, y + dy, x + dx
                            if (0 <= nz < reconstructed.shape[0] and
                                0 <= ny < reconstructed.shape[1] and
                                0 <= nx < reconstructed.shape[2]):
                                reconstructed[nz, ny, nx] = 1

        voxels_before = np.sum(binary_mask)
        voxels_after = np.sum(reconstructed)

        if self.verbose:
            print(f"  Original: {voxels_before:,} voxels")
            print(f"  Reconstructed: {voxels_after:,} voxels")

        self.mask = reconstructed.astype(np.uint8)
        return self.mask


    def _get_adaptive_threshold(self, position):
        dt = self._compute_distance_transform()
        z, y, x = position
        dist_from_edge = dt[z, y, x]

        if dist_from_edge < 5:
            return self.hu_thresholds['peripheral']
        elif dist_from_edge < 15:
            return self.hu_thresholds['intermediate']
        else:
            return self.hu_thresholds['central']

    def _priority_region_grow(self, seeds, max_dist_mm=4.0, max_voxels=2000):
        if len(seeds) == 0:
            return []

        from heapq import heappush, heappop

        visited = set()
        grown = []
        queue = []

        for seed in seeds:
            seed_tuple = tuple(seed)
            if seed_tuple not in visited:
                heappush(queue, (0.0, seed_tuple))
                visited.add(seed_tuple)

        max_dist_vox = int(max_dist_mm / min(self.spacing))
        voxel_count = 0

        while queue and voxel_count < max_voxels:
            priority, current_pos = heappop(queue)
            z, y, x = current_pos

            grown.append(current_pos)
            voxel_count += 1

            threshold = self._get_adaptive_threshold(current_pos)

            for dz in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dz == 0 and dy == 0 and dx == 0:
                            continue

                        nz, ny, nx = z + dz, y + dy, x + dx
                        neighbor_pos = (nz, ny, nx)

                        if not (0 <= nz < self.img.shape[0] and
                                0 <= ny < self.img.shape[1] and
                                0 <= nx < self.img.shape[2]):
                            continue

                        if neighbor_pos in visited:
                            continue

                        hu_value = self.img[nz, ny, nx]
                        if hu_value >= threshold:
                            continue

                        dist_from_seeds = min([
                            np.linalg.norm(np.array(neighbor_pos) - np.array(seed)) 
                            for seed in seeds
                        ])

                        if dist_from_seeds > max_dist_vox:
                            continue

                        hu_score = max(0, (threshold - hu_value) / 100.0)
                        connectivity_bonus = 1.0 if self.mask[nz, ny, nx] > 0 else 0.0
                        priority_score = dist_from_seeds - hu_score - connectivity_bonus

                        heappush(queue, (priority_score, neighbor_pos))
                        visited.add(neighbor_pos)

        return grown

    def _detect_endpoints_and_tips(self):
        if self.verbose:
            print("\n[Enhanced Refinement] Detecting airway endpoints...")

        binary_mask = (self.mask > 0).astype(np.uint8)
        skeleton = skeletonize(binary_mask)

        from scipy.ndimage import convolve
        kernel = np.ones((3, 3, 3))
        kernel[1, 1, 1] = 0
        neighbor_count = convolve(skeleton.astype(int), kernel, mode='constant')
        neighbor_count = neighbor_count * skeleton

        endpoints = (neighbor_count == 1) & (skeleton > 0)
        endpoint_coords = np.argwhere(endpoints)

        if self.verbose:
            print(f"  Found {len(endpoint_coords)} endpoints")

        dt = self._compute_distance_transform()
        valid_endpoints = []

        for ep in endpoint_coords:
            z, y, x = ep
            dist_to_edge = dt[z, y, x]
            if dist_to_edge < 3.0:
                valid_endpoints.append(ep)

        if self.verbose:
            print(f"  Valid endpoints near edge: {len(valid_endpoints)}")

        return valid_endpoints

    def _recover_thin_airways(self):
        if self.verbose:
            print("\n[Enhanced Refinement] Recovering thin airways...")

        binary_mask = (self.mask > 0).astype(np.uint8)
        closed = binary_closing(binary_mask, ball(2))
        gap_candidates = closed & (~binary_mask.astype(bool))
        gap_coords = np.argwhere(gap_candidates)

        if len(gap_coords) == 0:
            if self.verbose:
                print("  No gaps to fill")
            return np.zeros_like(self.mask)

        recovered = np.zeros_like(self.mask)
        recovered_count = 0

        for coord in gap_coords:
            z, y, x = coord
            hu_value = self.img[z, y, x]
            threshold = self._get_adaptive_threshold(coord)

            if hu_value < threshold:
                recovered[z, y, x] = 1
                recovered_count += 1

        if self.verbose:
            print(f"  Recovered {recovered_count} thin airway voxels")

        return recovered

    def _expand_near_skeleton(self):
        if self.verbose:
            print("\n[Enhanced Refinement] Expanding along skeleton...")

        binary_mask = (self.mask > 0).astype(np.uint8)
        skeleton = skeletonize(binary_mask)
        dt = self._compute_distance_transform()

        skeleton_coords = np.argwhere(skeleton)
        expanded = np.zeros_like(self.mask)
        expanded_count = 0

        for skel_pos in skeleton_coords:
            z, y, x = skel_pos

            for dz in range(-2, 3):
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        nz, ny, nx = z + dz, y + dy, x + dx

                        if not (0 <= nz < self.img.shape[0] and
                                0 <= ny < self.img.shape[1] and
                                0 <= nx < self.img.shape[2]):
                            continue

                        if self.mask[nz, ny, nx] > 0:
                            continue

                        hu_value = self.img[nz, ny, nx]
                        threshold = self._get_adaptive_threshold((nz, ny, nx))

                        if hu_value < threshold:
                            dist = np.sqrt(dz**2 + dy**2 + dx**2) * np.mean(self.spacing)
                            if dist < 2.0:
                                expanded[nz, ny, nx] = 1
                                expanded_count += 1

        if self.verbose:
            print(f"  Expanded {expanded_count} voxels near skeleton")

        return expanded


    def refine(self, enable_anti_blob=True, 
              min_blob_size_voxels=50,
              min_blob_size_mm3=10,
              max_blob_distance_mm=15.0,
              enable_tubular_smoothing=True,
              enable_skeleton_reconstruction=False):

        if self.verbose:
            print("\n" + "="*70)
            print("ENHANCED AIRWAY REFINEMENT WITH ANTI-BLOB MODULE")
            print("="*70)

        refined = self.mask.copy()
        initial_voxels = np.sum(refined > 0)

        if enable_anti_blob:
            if self.verbose:
                print("\n" + "="*70)
                print("PHASE 1: ANTI-BLOB PROCESSING")
                print("="*70)

            self.remove_small_blobs(
                min_size_voxels=min_blob_size_voxels,
                min_size_mm3=min_blob_size_mm3
            )

            self.detect_and_remove_spurious_blobs(
                max_blob_distance_mm=max_blob_distance_mm
            )

            if enable_tubular_smoothing:
                self.morphological_tubular_smoothing()

        endpoints = self._detect_endpoints_and_tips()
        if len(endpoints) > 0:
            if self.verbose:
                print(f"\n[Phase 2] Region growing from {len(endpoints)} endpoints...")

            grown_total = 0
            for ep in endpoints:
                grown = self._priority_region_grow([ep], max_dist_mm=4.0, max_voxels=500)
                for gz, gy, gx in grown:
                    if self.mask[gz, gy, gx] == 0:
                        self.mask[gz, gy, gx] = 1
                        grown_total += 1

            if self.verbose:
                print(f"  Added {grown_total} voxels from endpoint growing")

        thin_airways = self._recover_thin_airways()
        self.mask = np.logical_or(self.mask, thin_airways).astype(np.uint8)
        thin_count = np.sum(thin_airways)

        expanded = self._expand_near_skeleton()
        self.mask = np.logical_or(self.mask, expanded).astype(np.uint8)
        expanded_count = np.sum(expanded)

        if enable_skeleton_reconstruction:
            if self.verbose:
                print("\n[Phase 5] Skeleton-guided reconstruction (slow)...")
            self.skeleton_guided_reconstruction(max_reconstruction_mm=3.0)

        if self.verbose:
            print("\n[Phase 6] Final morphological smoothing...")

        self.mask = binary_dilation(self.mask, ball(1))
        self.mask = binary_erosion(self.mask, ball(1))

        self.refined = self.mask.astype(np.uint8)
        final_voxels = np.sum(self.refined > 0)
        added_voxels = final_voxels - initial_voxels

        if self.verbose:
            print(f"\n" + "="*70)
            print("REFINEMENT SUMMARY")
            print("="*70)
            print(f"Initial voxels: {initial_voxels:,}")
            print(f"Final voxels: {final_voxels:,}")
            print(f"Net change: {added_voxels:+,} ({added_voxels/initial_voxels*100:+.1f}%)")
            if enable_anti_blob:
                print(f"\nAnti-blob processing: ENABLED")
                print(f"  Small blobs removed")
                print(f"  Spurious blobs removed")
                if enable_tubular_smoothing:
                    print(f"  Tubular smoothing applied")
            if enable_skeleton_reconstruction:
                print(f"Skeleton reconstruction: ENABLED")

        return self.refined

    def save(self, path, ref_img):
        if self.refined is None:
            raise ValueError("Run refine() first")

        out = sitk.GetImageFromArray(self.refined)
        out.CopyInformation(ref_img)
        sitk.WriteImage(out, path)

        if self.verbose:
            print(f"\n✓ Saved refined mask: {path}")

        return path
