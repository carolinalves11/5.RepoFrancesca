import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label, binary_dilation, binary_erosion
from scipy.ndimage import distance_transform_edt, generate_binary_structure
from skimage.morphology import skeletonize, remove_small_objects
import networkx as nx


class SkeletonCleaner:

    def __init__(self, skeleton, original_mask, spacing):
        self.skeleton = skeleton.copy()
        self.mask = original_mask
        self.spacing = spacing
        self.cleaned_skeleton = None

    def remove_small_components(self, min_size_voxels=10):
        print(f"\n[Skeleton Cleaner] Removing components < {min_size_voxels} voxels")

        structure = generate_binary_structure(3, 3)
        labeled, num_components = label(self.skeleton, structure=structure)

        print(f"  Found {num_components} components before cleaning")

        component_sizes = []
        for i in range(1, num_components + 1):
            size = np.sum(labeled == i)
            component_sizes.append((i, size))

        component_sizes.sort(key=lambda x: x[1], reverse=True)

        cleaned = np.zeros_like(self.skeleton)
        kept_count = 0
        removed_voxels = 0

        for comp_id, size in component_sizes:
            if size >= min_size_voxels:
                cleaned[labeled == comp_id] = 1
                kept_count += 1
            else:
                removed_voxels += size

        print(f"  Kept {kept_count} components, removed {num_components - kept_count}")
        print(f"  Removed {removed_voxels:,} voxels from small components")

        self.skeleton = cleaned
        return self.skeleton

    def remove_isolated_branches(self, max_distance_mm=15.0):
        print(f"\n[Skeleton Cleaner] Removing isolated branches (dist > {max_distance_mm}mm)")

        structure = generate_binary_structure(3, 3)
        labeled, num_components = label(self.skeleton, structure=structure)

        if num_components <= 1:
            print("  Only one component, nothing to remove")
            return self.skeleton

        sizes = [(i, np.sum(labeled == i)) for i in range(1, num_components + 1)]
        sizes.sort(key=lambda x: x[1], reverse=True)
        main_id = sizes[0][0]

        main_component = (labeled == main_id)
        main_coords = np.argwhere(main_component)

        print(f"  Main component: {sizes[0][1]:,} voxels")
        print(f"  Checking {num_components - 1} other components...")

        kept_components = [main_id]
        removed_count = 0
        removed_voxels = 0

        for comp_id, size in sizes[1:]:
            comp_coords = np.argwhere(labeled == comp_id)

            comp_sample = comp_coords[::max(1, len(comp_coords)//10)]
            main_sample = main_coords[::max(1, len(main_coords)//50)]

            min_dist = float('inf')
            for cp in comp_sample:
                for mp in main_sample:
                    dist_vect = (cp - mp) * np.array([self.spacing[2], 
                                                      self.spacing[1], 
                                                      self.spacing[0]])
                    dist = np.linalg.norm(dist_vect)
                    if dist < min_dist:
                        min_dist = dist

            if min_dist <= max_distance_mm:
                kept_components.append(comp_id)
            else:
                removed_count += 1
                removed_voxels += size
                print(f"    Component {comp_id} ({size} voxels): dist={min_dist:.1f}mm > threshold, REMOVED")

        cleaned = np.zeros_like(self.skeleton)
        for comp_id in kept_components:
            cleaned[labeled == comp_id] = 1

        print(f"  ✓ Removed {removed_count} isolated components ({removed_voxels:,} voxels)")

        self.skeleton = cleaned
        return self.skeleton

    def prune_short_branches(self, min_branch_length_mm=3.0):
        print(f"\n[Skeleton Cleaner] Pruning branches < {min_branch_length_mm}mm")

        from scipy.ndimage import convolve

        kernel = np.ones((3, 3, 3))
        kernel[1, 1, 1] = 0

        neighbor_count = convolve(self.skeleton.astype(int), kernel, mode='constant')
        neighbor_count = neighbor_count * self.skeleton

        endpoints = (neighbor_count == 1) & (self.skeleton > 0)
        endpoint_coords = np.argwhere(endpoints)

        print(f"  Found {len(endpoint_coords)} endpoints")

        pruned = self.skeleton.copy()
        pruned_count = 0
        total_pruned_voxels = 0

        for ep in endpoint_coords:
            path = self._trace_from_endpoint(ep, neighbor_count, max_length_mm=min_branch_length_mm)

            if path is not None and len(path) > 0:
                path_length_mm = (len(path) - 1) * np.mean(self.spacing)

                if path_length_mm < min_branch_length_mm:
                    for pz, py, px in path:
                        pruned[pz, py, px] = 0
                    pruned_count += 1
                    total_pruned_voxels += len(path)

        print(f"  ✓ Pruned {pruned_count} short branches ({total_pruned_voxels} voxels)")

        self.skeleton = pruned
        return self.skeleton

    def _trace_from_endpoint(self, start, neighbor_count, max_length_mm=5.0):
        max_voxels = int(max_length_mm / np.mean(self.spacing)) + 1
        path = [tuple(start)]
        current = start
        visited = {tuple(start)}

        for _ in range(max_voxels):
            z, y, x = current
            neighbors = []

            for dz in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dz == 0 and dy == 0 and dx == 0:
                            continue

                        nz, ny, nx = z + dz, y + dy, x + dx

                        if (0 <= nz < self.skeleton.shape[0] and
                            0 <= ny < self.skeleton.shape[1] and
                            0 <= nx < self.skeleton.shape[2]):

                            if self.skeleton[nz, ny, nx] > 0 and (nz, ny, nx) not in visited:
                                neighbors.append((nz, ny, nx))

            if len(neighbors) == 0:
                break
            elif len(neighbors) > 1:
                break
            else:
                current = np.array(neighbors[0])
                visited.add(tuple(neighbors[0]))
                path.append(tuple(neighbors[0]))

                nc = neighbor_count[current[0], current[1], current[2]]
                if nc > 2:
                    break

        return path

    def remove_spurious_spurs(self, max_spur_length_mm=2.0):
        print(f"\n[Skeleton Cleaner] Removing spurious spurs < {max_spur_length_mm}mm")

        initial_voxels = np.sum(self.skeleton > 0)

        for iteration in range(3):
            self.prune_short_branches(min_branch_length_mm=max_spur_length_mm)
            current_voxels = np.sum(self.skeleton > 0)

            if current_voxels == initial_voxels:
                break

            initial_voxels = current_voxels

        return self.skeleton

    def smooth_skeleton(self):
        print("\n[Skeleton Cleaner] Applying morphological smoothing")

        from skimage.morphology import ball

        dilated = binary_dilation(self.skeleton, structure=ball(1))

        try:
            from skimage.morphology import skeletonize_3d
            smoothed = skeletonize_3d(dilated)
        except:
            from skimage.morphology import skeletonize
            smoothed = skeletonize(dilated)

        voxels_before = np.sum(self.skeleton > 0)
        voxels_after = np.sum(smoothed > 0)

        print(f"  Voxels before: {voxels_before:,}")
        print(f"  Voxels after: {voxels_after:,}")

        self.skeleton = smoothed.astype(np.uint8)
        return self.skeleton

    def full_cleaning_pipeline(self, 
                               min_component_size=10,
                               max_isolation_distance_mm=15.0,
                               min_branch_length_mm=3.0,
                               enable_smoothing=False):
        print("\n" + "="*70)
        print("SKELETON CLEANING PIPELINE")
        print("="*70)

        initial_voxels = np.sum(self.skeleton > 0)
        print(f"\nInitial skeleton: {initial_voxels:,} voxels")

        self.remove_small_components(min_size_voxels=min_component_size)

        self.remove_isolated_branches(max_distance_mm=max_isolation_distance_mm)

        self.prune_short_branches(min_branch_length_mm=min_branch_length_mm)

        self.remove_spurious_spurs(max_spur_length_mm=min_branch_length_mm)

        if enable_smoothing:
            self.smooth_skeleton()

        self.remove_small_components(min_size_voxels=min_component_size)

        final_voxels = np.sum(self.skeleton > 0)
        removed = initial_voxels - final_voxels

        print(f"\n" + "="*70)
        print("CLEANING RESULTS")
        print("="*70)
        print(f"Initial voxels: {initial_voxels:,}")
        print(f"Final voxels: {final_voxels:,}")
        print(f"Removed: {removed:,} ({removed/initial_voxels*100:.1f}%)")

        self.cleaned_skeleton = self.skeleton
        return self.cleaned_skeleton

    def save_cleaned_skeleton(self, output_path, reference_image):
        if self.cleaned_skeleton is None:
            skeleton_to_save = self.skeleton
        else:
            skeleton_to_save = self.cleaned_skeleton

        skeleton_sitk = sitk.GetImageFromArray(skeleton_to_save.astype(np.uint8))
        skeleton_sitk.CopyInformation(reference_image)
        sitk.WriteImage(skeleton_sitk, output_path)

        print(f"\n✓ Cleaned skeleton saved: {output_path}")
        return output_path


def integrate_skeleton_cleaning(mask_path, output_dir,
                                min_component_size=10,
                                max_isolation_distance_mm=15.0,
                                min_branch_length_mm=3.0,
                                enable_smoothing=False):


    import os
    from preprocessin_cleaning import SegmentationPreprocessor

    os.makedirs(output_dir, exist_ok=True)

    sitk_img = sitk.ReadImage(mask_path)
    mask = sitk.GetArrayFromImage(sitk_img)
    spacing = sitk_img.GetSpacing()

    print("\n[Integration] Computing initial skeleton...")
    binary_mask = (mask > 0).astype(np.uint8)
    skeleton = skeletonize(binary_mask)

    cleaner = SkeletonCleaner(skeleton, mask, spacing)
    cleaned_skeleton = cleaner.full_cleaning_pipeline(
        min_component_size=min_component_size,
        max_isolation_distance_mm=max_isolation_distance_mm,
        min_branch_length_mm=min_branch_length_mm,
        enable_smoothing=enable_smoothing
    )

    scan_name = os.path.splitext(os.path.basename(mask_path))[0]
    output_path = os.path.join(output_dir, f"{scan_name}_skeleton_cleaned.nii.gz")
    cleaner.save_cleaned_skeleton(output_path, sitk_img)

    return output_path, cleaned_skeleton