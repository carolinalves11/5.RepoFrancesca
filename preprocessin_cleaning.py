import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import label, binary_dilation, binary_erosion, distance_transform_edt
from scipy.ndimage import generate_binary_structure
import pandas as pd
from skimage.morphology import ball, skeletonize
from skan import Skeleton, summarize
import networkx as nx
import os


class SegmentationPreprocessor:

    def __init__(self, mask_path, spacing=None):
        self.mask_path = mask_path

        print(f"Loading segmentation from: {mask_path}")
        self.sitk_image = sitk.ReadImage(mask_path)
        self.original_mask = sitk.GetArrayFromImage(self.sitk_image)

        self.spacing = spacing if spacing else self.sitk_image.GetSpacing()
        print(f"Spacing (x,y,z): {self.spacing} mm")
        print(f"Shape (z,y,x): {self.original_mask.shape}")
        print(f"Positive voxels: {np.sum(self.original_mask > 0)}")

        self.cleaned_mask = None
        self.connected_components = None
        self.component_stats = None
        self.reconnection_info = []

        self.preliminary_skeleton = None
        self.preliminary_graph = None

    def analyze_components(self):
        print("\n" + "="*60)
        print("CONNECTED COMPONENTS ANALYSIS (PRE-SKELETONIZATION)")
        print("="*60)

        binary_mask = (self.original_mask > 0).astype(np.uint8)

        structure = generate_binary_structure(3, 3)
        labeled_array, num_features = label(binary_mask, structure=structure)

        self.connected_components = labeled_array

        print(f"\nFound {num_features} connected components")

        if num_features == 0:
            print("WARNING: No components found in mask!")
            return None

        component_stats = []
        for i in range(1, num_features + 1):
            component_mask = (labeled_array == i)
            voxel_count = np.sum(component_mask)

            voxel_volume = self.spacing[0] * self.spacing[1] * self.spacing[2]
            volume_mm3 = voxel_count * voxel_volume

            coords = np.argwhere(component_mask)

            if len(coords) > 0:
                centroid = np.mean(coords, axis=0)

                z_min, y_min, x_min = np.min(coords, axis=0)
                z_max, y_max, x_max = np.max(coords, axis=0)

                z_extent = (z_max - z_min) * self.spacing[2]
                y_extent = (y_max - y_min) * self.spacing[1]
                x_extent = (x_max - x_min) * self.spacing[0]

                component_stats.append({
                    'component_id': i,
                    'voxel_count': voxel_count,
                    'volume_mm3': volume_mm3,
                    'centroid_z': centroid[0],
                    'centroid_y': centroid[1],
                    'centroid_x': centroid[2],
                    'bbox_min_z': z_min,
                    'bbox_min_y': y_min,
                    'bbox_min_x': x_min,
                    'bbox_max_z': z_max,
                    'bbox_max_y': y_max,
                    'bbox_max_x': x_max,
                    'extent_z_mm': z_extent,
                    'extent_y_mm': y_extent,
                    'extent_x_mm': x_extent
                })

        self.component_stats = pd.DataFrame(component_stats)
        self.component_stats = self.component_stats.sort_values('voxel_count', ascending=False)

        print(f"\n{'='*60}")
        print("COMPONENT STATISTICS")
        print(f"{'='*60}")
        print(f"Largest component: {self.component_stats['voxel_count'].iloc[0]:,} voxels "
              f"({self.component_stats['volume_mm3'].iloc[0]:.2f} mm³)")

        if num_features > 1:
            print(f"2nd largest: {self.component_stats['voxel_count'].iloc[1]:,} voxels "
                  f"({self.component_stats['volume_mm3'].iloc[1]:.2f} mm³)")
            print(f"Smallest: {self.component_stats['voxel_count'].iloc[-1]} voxels "
                  f"({self.component_stats['volume_mm3'].iloc[-1]:.2f} mm³)")

            total_voxels = np.sum(self.original_mask > 0)
            main_percentage = (self.component_stats['voxel_count'].iloc[0] / total_voxels) * 100

            print(f"\nMain component represents {main_percentage:.1f}% of total segmentation")
            print(f"Disconnected components: {num_features - 1}")

            print(f"\n{'Component':<12} {'Voxels':<12} {'Volume (mm³)':<15} {'Extent (mm)'}")
            print("-" * 60)
            for idx, row in self.component_stats.head(10).iterrows():
                extent_str = f"{row['extent_x_mm']:.1f}×{row['extent_y_mm']:.1f}×{row['extent_z_mm']:.1f}"
                print(f"{row['component_id']:<12} {row['voxel_count']:<12,} {row['volume_mm3']:<15.2f} {extent_str}")

        return self.component_stats

    def keep_largest_component(self):
        print(f"\n{'='*60}")
        print("KEEPING LARGEST COMPONENT ONLY")
        print(f"{'='*60}")

        if self.connected_components is None:
            raise ValueError("Run analyze_components() first")

        largest_id = self.component_stats.iloc[0]['component_id']

        self.cleaned_mask = (self.connected_components == largest_id).astype(np.uint8)

        original_voxels = np.sum(self.original_mask > 0)
        cleaned_voxels = np.sum(self.cleaned_mask > 0)
        removed_voxels = original_voxels - cleaned_voxels

        print(f"Original voxels: {original_voxels:,}")
        print(f"Kept voxels: {cleaned_voxels:,}")
        print(f"Removed voxels: {removed_voxels:,} ({removed_voxels/original_voxels*100:.2f}%)")

        return self.cleaned_mask

    def compute_preliminary_skeleton_and_graph(self):
        print(f"\n{'='*60}")
        print("COMPUTING PRELIMINARY SKELETON FOR ANALYSIS")
        print(f"{'='*60}")

        if self.cleaned_mask is None:
            self.keep_largest_component()

        binary_mask = (self.cleaned_mask > 0).astype(np.uint8)
        print("Computing 3D skeleton (this may take a few minutes)...")
        self.preliminary_skeleton = skeletonize(binary_mask)
        print(f"Skeleton computed: {np.sum(self.preliminary_skeleton > 0)} voxels")

        print("Computing distance transform...")
        spacing_zyx = (self.spacing[2], self.spacing[1], self.spacing[0])
        self.distance_transform = distance_transform_edt(binary_mask, sampling=spacing_zyx)

        print("Building preliminary graph with skan...")
        skeleton_obj = Skeleton(self.preliminary_skeleton, spacing=spacing_zyx)
        branch_data = summarize(skeleton_obj)

        print(f"Identified {len(branch_data)} branches in preliminary skeleton")

        self.preliminary_graph = nx.Graph()

        coordinates = skeleton_obj.coordinates
        for idx in range(len(coordinates)):
            pos = coordinates[idx]
            z, y, x = int(pos[0]), int(pos[1]), int(pos[2])

            if (0 <= z < self.distance_transform.shape[0] and
                0 <= y < self.distance_transform.shape[1] and
                0 <= x < self.distance_transform.shape[2]):
                diameter = self.distance_transform[z, y, x] * 2
            else:
                diameter = 0

            self.preliminary_graph.add_node(idx, pos=pos, diameter=diameter)

        for _, row in branch_data.iterrows():
            node1 = int(row['node-id-src'])
            node2 = int(row['node-id-dst'])
            length = row['branch-distance']

            try:
                coords = skeleton_obj.path_coordinates(row.name)
                diameters = []
                for coord in coords:
                    z, y, x = coord
                    if (0 <= z < self.distance_transform.shape[0] and
                        0 <= y < self.distance_transform.shape[1] and
                        0 <= x < self.distance_transform.shape[2]):
                        diameters.append(self.distance_transform[z, y, x] * 2)
                avg_diameter = np.mean(diameters) if diameters else 0
            except:
                avg_diameter = 0

            self.preliminary_graph.add_edge(node1, node2, 
                                           length=length, 
                                           diameter=avg_diameter,
                                           branch_type=row['branch-type'])

        print(f"Graph created: {len(self.preliminary_graph.nodes())} nodes, "
              f"{len(self.preliminary_graph.edges())} edges")

        return self.preliminary_skeleton, self.preliminary_graph

    def calculate_path_distance(self, component_coords, main_component_coords):

        if self.preliminary_graph is None:
            return self._calculate_euclidean_distance(component_coords, main_component_coords)

        comp_sample = component_coords[::max(1, len(component_coords)//20)]
        main_sample = main_component_coords[::max(1, len(main_component_coords)//50)]

        skeleton_coords = np.array([self.preliminary_graph.nodes[n]['pos'] 
                                    for n in self.preliminary_graph.nodes()])

        min_distance = float('inf')
        best_comp_point = None
        best_main_point = None

        for comp_point in comp_sample:
            dists_to_skel = np.linalg.norm(skeleton_coords - comp_point, axis=1)
            comp_node = np.argmin(dists_to_skel)

            for main_point in main_sample:
                dists_to_main = np.linalg.norm(skeleton_coords - main_point, axis=1)
                main_node = np.argmin(dists_to_main)

                try:
                    path_length = nx.shortest_path_length(
                        self.preliminary_graph, 
                        comp_node, 
                        main_node, 
                        weight='length'
                    )

                    total_distance = (path_length + 
                                    dists_to_skel[comp_node] * np.mean(self.spacing) +
                                    dists_to_main[main_node] * np.mean(self.spacing))

                    if total_distance < min_distance:
                        min_distance = total_distance
                        best_comp_point = comp_point
                        best_main_point = main_point

                except nx.NetworkXNoPath:
                    continue

        if best_comp_point is None:
            return self._calculate_euclidean_distance(component_coords, main_component_coords)

        return min_distance, best_comp_point, best_main_point

    def _calculate_euclidean_distance(self, component_coords, main_component_coords):
        comp_sample = component_coords[::max(1, len(component_coords)//20)]
        main_sample = main_component_coords[::max(1, len(main_component_coords)//50)]

        min_distance = float('inf')
        best_comp_point = None
        best_main_point = None

        for comp_point in comp_sample:
            for main_point in main_sample:
                dist_vector = (comp_point - main_point) * np.array([self.spacing[2], 
                                                                     self.spacing[1], 
                                                                     self.spacing[0]])
                distance = np.linalg.norm(dist_vector)

                if distance < min_distance:
                    min_distance = distance
                    best_comp_point = comp_point
                    best_main_point = main_point

        return min_distance, best_comp_point, best_main_point

    def reconnect_nearby_components(self, max_distance_mm=10.0, min_component_size=50,
                                   max_components_to_reconnect=10, use_path_distance=True):

        print(f"\n{'='*60}")
        print("ATTEMPTING COMPONENT RECONNECTION (PATH-BASED)")
        print(f"{'='*60}")
        print(f"Max reconnection distance: {max_distance_mm} mm")
        print(f"Min component size: {min_component_size} voxels")
        print(f"Distance method: {'Path-based (along skeleton)' if use_path_distance else 'Euclidean'}")

        if self.connected_components is None:
            raise ValueError("Run analyze_components() first")

        if len(self.component_stats) <= 1:
            print("Only one component found - no reconnection needed")
            self.cleaned_mask = (self.original_mask > 0).astype(np.uint8)
            return self.cleaned_mask

        if use_path_distance and self.preliminary_graph is None:
            print("\nComputing preliminary skeleton for path-based distances...")
            self.compute_preliminary_skeleton_and_graph()

        main_id = self.component_stats.iloc[0]['component_id']
        main_mask = (self.connected_components == main_id).astype(np.uint8)
        main_component_coords = np.argwhere(main_mask > 0)

        print(f"\nMain component: {len(main_component_coords):,} voxels")

        reconnected_count = 0
        kept_isolated_count = 0

        for idx in range(1, min(len(self.component_stats), max_components_to_reconnect + 1)):
            comp_row = self.component_stats.iloc[idx]
            comp_id = comp_row['component_id']
            comp_size = comp_row['voxel_count']

            if comp_size < min_component_size:
                print(f"\nComponent {comp_id}: {comp_size} voxels - too small, skipping")
                continue

            print(f"\nComponent {comp_id}: {comp_size} voxels ({comp_row['volume_mm3']:.2f} mm³)")

            comp_mask = (self.connected_components == comp_id).astype(np.uint8)
            comp_coords = np.argwhere(comp_mask > 0)

            if use_path_distance:
                min_distance, best_comp_point, best_main_point = self.calculate_path_distance(
                    comp_coords, main_component_coords
                )
                distance_type = "path"
            else:
                min_distance, best_comp_point, best_main_point = self._calculate_euclidean_distance(
                    comp_coords, main_component_coords
                )
                distance_type = "Euclidean"

            print(f"  Minimum {distance_type} distance: {min_distance:.2f} mm")

            if min_distance <= max_distance_mm:
                print(f"  ✓ Reconnecting (distance {min_distance:.2f} mm ≤ {max_distance_mm} mm)")

                bridge_mask = self._create_bridge(best_comp_point, best_main_point, comp_mask.shape)

                main_mask = main_mask | comp_mask | bridge_mask

                main_component_coords = np.argwhere(main_mask > 0)

                self.reconnection_info.append({
                    'component_id': comp_id,
                    'size_voxels': comp_size,
                    'distance_mm': min_distance,
                    'distance_type': distance_type,
                    'reconnected': True
                })

                reconnected_count += 1
            else:
                print(f"  ✗ Keeping isolated (distance {min_distance:.2f} mm > {max_distance_mm} mm)")
                self.reconnection_info.append({
                    'component_id': comp_id,
                    'size_voxels': comp_size,
                    'distance_mm': min_distance,
                    'distance_type': distance_type,
                    'reconnected': False
                })
                kept_isolated_count += 1

        self.cleaned_mask = main_mask

        print(f"\n{'='*60}")
        print("RECONNECTION SUMMARY")
        print(f"{'='*60}")
        print(f"Components reconnected: {reconnected_count}")
        print(f"Components kept isolated: {kept_isolated_count}")
        print(f"Final mask voxels: {np.sum(self.cleaned_mask):,}")

        return self.cleaned_mask

    def _create_bridge(self, point1, point2, shape):
        p1 = np.array(point1, dtype=int)
        p2 = np.array(point2, dtype=int)

        distance_voxels = np.linalg.norm(p2 - p1)
        num_points = max(3, int(distance_voxels * 2))

        bridge = np.zeros(shape, dtype=np.uint8)

        for t in np.linspace(0, 1, num_points):
            center = p1 + t * (p2 - p1)
            z, y, x = center.astype(int)

            for dz in range(-2, 3):
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        if dz*dz + dy*dy + dx*dx <= 4:
                            nz, ny, nx = z + dz, y + dy, x + dx
                            if (0 <= nz < shape[0] and 
                                0 <= ny < shape[1] and 
                                0 <= nx < shape[2]):
                                bridge[nz, ny, nx] = 1

        return bridge.astype(bool)

    def morphological_cleanup(self, remove_small_holes=True, hole_size_mm3=50,
                            smooth_surface=False):

        print(f"\n{'='*60}")
        print("MORPHOLOGICAL CLEANUP")
        print(f"{'='*60}")

        if self.cleaned_mask is None:
            self.cleaned_mask = (self.original_mask > 0).astype(np.uint8)

        mask = self.cleaned_mask.copy()

        if remove_small_holes:
            print(f"Filling holes smaller than {hole_size_mm3} mm³...")

            inverted = ~mask.astype(bool)

            labeled_holes, num_holes = label(inverted)

            voxel_volume = self.spacing[0] * self.spacing[1] * self.spacing[2]
            max_hole_voxels = int(hole_size_mm3 / voxel_volume)

            filled_count = 0
            for hole_id in range(1, num_holes + 1):
                hole_size = np.sum(labeled_holes == hole_id)
                if hole_size < max_hole_voxels:
                    mask[labeled_holes == hole_id] = 1
                    filled_count += 1

            print(f"  Filled {filled_count} small holes")

        if smooth_surface:
            print("Smoothing surface (erosion + dilation)...")
            mask = binary_erosion(mask, structure=ball(1))
            mask = binary_dilation(mask, structure=ball(1))
            print("  Surface smoothed")

        self.cleaned_mask = mask.astype(np.uint8)

        return self.cleaned_mask

    def visualize_components_3d(self, save_path=None, max_components=10):
        if self.connected_components is None:
            raise ValueError("Run analyze_components() first")

        print(f"\nVisualizing top {max_components} components...")

        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')

        colors = plt.cm.tab20(np.linspace(0, 1, max_components))

        for i in range(min(max_components, len(self.component_stats))):
            comp_id = self.component_stats.iloc[i]['component_id']
            comp_size = self.component_stats.iloc[i]['voxel_count']

            coords = np.argwhere(self.connected_components == comp_id)

            if len(coords) > 5000:
                subsample = len(coords) // 5000
                coords = coords[::subsample]

            ax.scatter(coords[:, 2], coords[:, 1], coords[:, 0],
                      c=[colors[i]], s=10, alpha=0.6,
                      label=f'Comp {comp_id} ({comp_size:,} vox)')

        ax.set_xlabel('X (voxel)')
        ax.set_ylabel('Y (voxel)')
        ax.set_zlabel('Z (voxel)')
        title = f'Connected Components Analysis\n{len(self.component_stats)} total components'
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.show()

    def visualize_before_after(self, save_path=None):
        if self.cleaned_mask is None:
            raise ValueError("Run preprocessing steps first")

        fig = plt.figure(figsize=(18, 6))

        ax1 = fig.add_subplot(131, projection='3d')
        original_coords = np.argwhere(self.original_mask > 0)
        subsample = max(1, len(original_coords) // 5000)
        original_coords = original_coords[::subsample]
        ax1.scatter(original_coords[:, 2], original_coords[:, 1], original_coords[:, 0],
                   c='blue', s=1, alpha=0.5)
        ax1.set_title(f'Original\n({np.sum(self.original_mask > 0):,} voxels)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        ax2 = fig.add_subplot(132, projection='3d')
        cleaned_coords = np.argwhere(self.cleaned_mask > 0)
        subsample = max(1, len(cleaned_coords) // 5000)
        cleaned_coords = cleaned_coords[::subsample]
        ax2.scatter(cleaned_coords[:, 2], cleaned_coords[:, 1], cleaned_coords[:, 0],
                   c='green', s=1, alpha=0.5)

        title = f'Cleaned\n({np.sum(self.cleaned_mask > 0):,} voxels)'
        ax2.set_title(title)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        ax3 = fig.add_subplot(133, projection='3d')
        removed = (self.original_mask > 0) & (self.cleaned_mask == 0)
        removed_coords = np.argwhere(removed)
        if len(removed_coords) > 0:
            subsample = max(1, len(removed_coords) // 5000)
            removed_coords = removed_coords[::subsample]
            ax3.scatter(removed_coords[:, 2], removed_coords[:, 1], removed_coords[:, 0],
                       c='red', s=1, alpha=0.5)
        ax3.set_title(f'Removed\n({np.sum(removed):,} voxels)')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.show()

    def save_cleaned_mask(self, output_path):
        if self.cleaned_mask is None:
            raise ValueError("No cleaned mask available. Run preprocessing first.")

        cleaned_sitk = sitk.GetImageFromArray(self.cleaned_mask)
        cleaned_sitk.CopyInformation(self.sitk_image)

        sitk.WriteImage(cleaned_sitk, output_path)
        print(f"\nCleaned mask saved: {output_path}")

        return output_path

    def save_report(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        report_path = os.path.join(output_dir, "preprocessing_report.txt")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("AIRWAY SEGMENTATION PREPROCESSING REPORT\n")
            f.write("="*60 + "\n\n")

            f.write(f"Input file: {self.mask_path}\n")
            f.write(f"Spacing (x,y,z): {self.spacing} mm\n")
            f.write(f"Shape (z,y,x): {self.original_mask.shape}\n\n")

            f.write(f"Original voxels: {np.sum(self.original_mask > 0):,}\n")
            if self.cleaned_mask is not None:
                f.write(f"Cleaned voxels: {np.sum(self.cleaned_mask > 0):,}\n")
                removed = np.sum(self.original_mask > 0) - np.sum(self.cleaned_mask > 0)
                f.write(f"Removed voxels: {removed:,} ({removed/np.sum(self.original_mask > 0)*100:.2f}%)\n\n")

            if self.component_stats is not None:
                f.write("\nCONNECTED COMPONENTS:\n")
                f.write(f"Total components found: {len(self.component_stats)}\n")
                f.write(f"Largest component: {self.component_stats['voxel_count'].iloc[0]:,} voxels\n")
                if len(self.component_stats) > 1:
                    f.write(f"2nd largest: {self.component_stats['voxel_count'].iloc[1]:,} voxels\n")
                f.write("\n")

            if self.reconnection_info:
                f.write("\nRECONNECTION ATTEMPTS:\n")
                for info in self.reconnection_info:
                    status = "✓ Reconnected" if info['reconnected'] else "✗ Kept isolated"
                    distance_type = info.get('distance_type', 'Euclidean')
                    f.write(f"Component {info['component_id']}: {info['size_voxels']} voxels, "
                           f"{distance_type} distance {info['distance_mm']:.2f} mm - {status}\n")

        print(f"Report saved: {report_path}")

        if self.component_stats is not None:
            csv_path = os.path.join(output_dir, "component_statistics.csv")
            self.component_stats.to_csv(csv_path, index=False)
            print(f"Component stats saved: {csv_path}")

    def run_full_preprocessing(self, output_dir, 
                              try_reconnection=True,
                              max_reconnect_distance_mm=10.0,
                              min_component_size=50,
                              visualize=True):

        print("\n" + "="*60)
        print("AIRWAY SEGMENTATION PREPROCESSING PIPELINE")
        print("="*60)
        print("NOTE: Trachea removal has been disabled")

        os.makedirs(output_dir, exist_ok=True)

        self.analyze_components()

        self.keep_largest_component()

        if try_reconnection:
            self.compute_preliminary_skeleton_and_graph()

        if try_reconnection and len(self.component_stats) > 1:
            self.reconnect_nearby_components(
                max_distance_mm=max_reconnect_distance_mm,
                min_component_size=min_component_size,
                use_path_distance=True
            )

        self.morphological_cleanup(
            remove_small_holes=True,
            hole_size_mm3=50,
            smooth_surface=False
        )

        cleaned_path = os.path.join(output_dir, "cleaned_airway_mask_complete.nii.gz")
        self.save_cleaned_mask(cleaned_path)
        self.save_report(output_dir)

        if visualize:
            self.visualize_components_3d(
                save_path=os.path.join(output_dir, "components_3d.png")
            )
            self.visualize_before_after(
                save_path=os.path.join(output_dir, "before_after.png")
            )

        print("\n" + "="*60)
        print("PREPROCESSING COMPLETED!")
        print("="*60)
        print(f"\nCleaned mask ready: {cleaned_path}")
        print("✓ Complete airway tree preserved (trachea + bronchi)")
        print("You can now use this for detailed airway tree analysis")

        return self.cleaned_mask, cleaned_path