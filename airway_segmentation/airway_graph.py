import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.morphology import skeletonize, ball, binary_dilation
from scipy.ndimage import distance_transform_edt, label
from skan import Skeleton, summarize
import networkx as nx
import pandas as pd
import json
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from scipy import ndimage
from collections import deque

class AirwayGraphAnalyzer:
    """
    Analyzes the 3D structure of airways with Weibel generation analysis:
    - Generates 3D skeleton
    - Builds topological graph
    - Calculates diameters and lengths along branches
    - Identifies and analyzes bifurcations
    - Classifies branches by generation (Weibel model)
    - Analyzes diameter tapering across generations
    - Analyzes and manages connected components of the skeleton
    """

    def __init__(self, airway_mask_path, spacing=None, carina_coords=None, original_mask_path=None):
        """
        Args:
            airway_mask_path: Path to .nii.gz airway mask file (bronchi only, no trachea)
                             Used for skeleton/topology
            spacing: Tuple (x,y,z) of spacing in mm. If None, reads from image
            carina_coords: Optional initial carina coordinates
            original_mask_path: Optional path to original mask for accurate diameter/volume metrics
                               (dual-mask strategy: refined for skeleton, original for metrics)
        """
        self.mask_path = airway_mask_path
        self.original_mask_path = original_mask_path

        print(f"Loading mask from: {airway_mask_path}")
        self.sitk_image = sitk.ReadImage(airway_mask_path)
        self.mask = sitk.GetArrayFromImage(self.sitk_image)

        self.spacing = spacing if spacing else self.sitk_image.GetSpacing()
        print(f"Spacing (x,y,z): {self.spacing} mm")
        print(f"Shape (z,y,x): {self.mask.shape}")
        print(f"Positive voxels: {np.sum(self.mask > 0)}")

        self.original_mask = None
        self.original_distance_transform = None
        if original_mask_path:
            print(f"\nLoading ORIGINAL mask for accurate metrics: {original_mask_path}")
            original_sitk = sitk.ReadImage(original_mask_path)
            self.original_mask = sitk.GetArrayFromImage(original_sitk)
            print(f"Original mask positive voxels: {np.sum(self.original_mask > 0)}")
            print("DUAL-MASK STRATEGY: Skeleton from refined, metrics from original")

        self.skeleton = None
        self.skeleton_obj = None
        self.graph = None
        self.branch_data = None
        self.distance_transform = None
        self.connected_components = None
        self.component_stats = None

        self.carina_node = None
        self.generation_assignments = None
        self.weibel_analysis_df = None
        self.initial_carina_coords = tuple(carina_coords) if carina_coords is not None else None

    def compute_skeleton(self):
        """Computes the 3D skeleton of the mask"""
        print("\n=== 3D Skeletonization ===")

        binary_mask = (self.mask > 0).astype(np.uint8)

        print("Computing 3D skeleton (may take a few minutes)...")
        self.skeleton = skeletonize(binary_mask)

        self.skeleton_voxels = np.sum(self.skeleton > 0)
        print(f"Skeleton computed: {self.skeleton_voxels} voxels")

        print("Computing distance transform for diameters...")
        self.distance_transform = distance_transform_edt(binary_mask, sampling=self.spacing)

        if self.original_mask is not None:
            print("Computing distance transform on ORIGINAL mask for accurate diameters...")
            original_binary = (self.original_mask > 0).astype(np.uint8)
            self.original_distance_transform = distance_transform_edt(original_binary, sampling=self.spacing)
            print("✓ DUAL-MASK: Skeleton from refined, diameters from original")
        else:
            self.original_distance_transform = self.distance_transform

        return self.skeleton

    def analyze_connected_components(self):
        """Analyzes connected components of the skeleton"""
        print("\n=== SKELETON CONNECTED COMPONENTS ANALYSIS ===")

        if self.skeleton is None:
            raise ValueError("Compute skeleton first with compute_skeleton()")

        labeled_array, num_features = label(self.skeleton, structure=np.ones((3,3,3)))

        print(f"Total number of connected components in skeleton: {num_features}")

        component_stats = []
        for i in range(1, num_features + 1):
            component_mask = (labeled_array == i)
            voxel_count = np.sum(component_mask)

            volume_mm3 = voxel_count * self.spacing[0] * self.spacing[1] * self.spacing[2]

            coordinates = np.argwhere(component_mask)

            if len(coordinates) > 0:
                centroid = np.mean(coordinates, axis=0)

                z_min, y_min, x_min = np.min(coordinates, axis=0)
                z_max, y_max, x_max = np.max(coordinates, axis=0)

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
                    'bbox_max_x': x_max
                })

        self.connected_components = labeled_array
        self.component_stats = pd.DataFrame(component_stats)

        self.component_stats = self.component_stats.sort_values('voxel_count', ascending=False)

        print(f"\nConnected components statistics:")
        print(f"  Largest component: {self.component_stats['voxel_count'].iloc[0]} voxels "
              f"({self.component_stats['volume_mm3'].iloc[0]:.2f} mm³)")
        print(f"  Smallest component: {self.component_stats['voxel_count'].iloc[-1]} voxels "
              f"({self.component_stats['volume_mm3'].iloc[-1]:.2f} mm³)")
        print(f"  Average size: {self.component_stats['voxel_count'].mean():.1f} voxels "
              f"({self.component_stats['volume_mm3'].mean():.2f} mm³)")

        large_components = self.component_stats[self.component_stats['voxel_count'] >= 10]
        small_components = self.component_stats[self.component_stats['voxel_count'] < 10]

        print(f"\nSize distribution:")
        print(f"  Components with ≥10 voxels: {len(large_components)}")
        print(f"  Components with <10 voxels: {len(small_components)}")

        if hasattr(self, 'skeleton_voxels') and self.skeleton_voxels > 0:
            print(f"  Percentage of voxels in main component: "
                  f"{self.component_stats['voxel_count'].iloc[0] / self.skeleton_voxels * 100:.1f}%")
        else:
            skeleton_voxels_current = np.sum(self.skeleton > 0)
            print(f"  Percentage of voxels in main component: "
                  f"{self.component_stats['voxel_count'].iloc[0] / skeleton_voxels_current * 100:.1f}%")

        return self.connected_components, self.component_stats

    def smart_component_management(self, max_reconnect_distance_mm=15.0, min_voxels_for_reconnect=5, 
                                 max_voxels_for_keep=100, remove_tiny_components=True):
        """
        Intelligent component management:
        - Attempts to reconnect nearby significant components
        - Removes components not worth reconnecting
        """
        print(f"\n=== INTELLIGENT COMPONENT MANAGEMENT ===")
        print(f"Max reconnection distance: {max_reconnect_distance_mm} mm")
        print(f"Min voxels for reconnection: {min_voxels_for_reconnect}")
        print(f"Max voxels to keep isolated: {max_voxels_for_keep}")

        if self.connected_components is None:
            raise ValueError("Analyze connected components first with analyze_connected_components()")

        if len(self.component_stats) <= 1:
            print("Only one component found, nothing to manage")
            return self.skeleton

        main_component_id = self.component_stats.iloc[0]['component_id']
        main_component_mask = (self.connected_components == main_component_id)
        main_component_coords = np.argwhere(main_component_mask)

        if len(main_component_coords) == 0:
            print("Main component is empty, cannot proceed")
            return self.skeleton

        components_reconnected = 0
        components_kept_isolated = 0
        components_removed = 0
        voxels_reconnected = 0
        voxels_kept = 0
        voxels_removed = 0

        print(f"\nAnalyzing isolated components:")

        new_skeleton = self.skeleton.copy()

        for i in range(1, len(self.component_stats)):
            comp_row = self.component_stats.iloc[i]
            comp_id = comp_row['component_id']
            voxel_count = comp_row['voxel_count']
            centroid = np.array([comp_row['centroid_z'], comp_row['centroid_y'], comp_row['centroid_x']])

            comp_mask = (self.connected_components == comp_id)
            comp_coords = np.argwhere(comp_mask)

            if len(comp_coords) == 0:
                continue

            min_distance = float('inf')
            best_comp_point = None
            best_main_point = None

            comp_sample = comp_coords[::max(1, len(comp_coords)//5)]
            main_sample = main_component_coords[::max(1, len(main_component_coords)//50)]

            for comp_point in comp_sample:
                for main_point in main_sample:
                    dist_vector = (comp_point - main_point) * np.array([self.spacing[2], self.spacing[1], self.spacing[0]])
                    distance = np.linalg.norm(dist_vector)

                    if distance < min_distance:
                        min_distance = distance
                        best_comp_point = comp_point
                        best_main_point = main_point

            print(f"Component {comp_id}: {voxel_count} voxels, minimum distance: {min_distance:.2f} mm")

            if (min_distance <= max_reconnect_distance_mm and 
                voxel_count >= min_voxels_for_reconnect):
                print(f"  → Reconnected! (distance: {min_distance:.2f} mm)")
                self._create_bridge_to_skeleton(best_comp_point, best_main_point, new_skeleton)
                components_reconnected += 1
                voxels_reconnected += voxel_count

            elif voxel_count > max_voxels_for_keep:
                print(f"  → Kept isolated (too large: {voxel_count} voxels)")
                components_kept_isolated += 1
                voxels_kept += voxel_count

            elif remove_tiny_components and voxel_count < min_voxels_for_reconnect:
                print(f"  → Removed (too small: {voxel_count} voxels)")
                new_skeleton[comp_mask] = 0
                components_removed += 1
                voxels_removed += voxel_count

            else:
                print(f"  → Kept isolated (distance: {min_distance:.2f} mm)")
                components_kept_isolated += 1
                voxels_kept += voxel_count

        self.skeleton = new_skeleton.astype(bool)

        print(f"\n=== COMPONENT MANAGEMENT RESULTS ===")
        print(f"Reconnected components: {components_reconnected}")
        print(f"Components kept isolated: {components_kept_isolated}")
        print(f"Removed components: {components_removed}")
        print(f"Reconnected voxels: {voxels_reconnected}")
        print(f"Voxels kept isolated: {voxels_kept}")
        print(f"Removed voxels: {voxels_removed}")

        if components_reconnected > 0:
            print(f"✓ Reconnected {components_reconnected} components to main structure")
        if components_kept_isolated > 0:
            print(f"✓ Kept {components_kept_isolated} components isolated (significant)")
        if components_removed > 0:
            print(f"✓ Removed {components_removed} components (noise)")

        self._force_connect_top_components(new_skeleton)

        return self.skeleton

    def _force_connect_top_components(self, skeleton_array):
        """
        CRITICAL FIX for asymmetry issue:
        Forces connection between the 2 largest components (left & right lung)
        even if they are far apart in skeleton space.
        """
        print(f"\n=== FORCE CONNECTING TOP COMPONENTS (ASYMMETRY FIX) ===")

        labeled_array, num_features = label(skeleton_array, structure=np.ones((3,3,3)))

        if num_features < 2:
            print("Only 1 component found - no need to force connection")
            return

        component_sizes = []
        for i in range(1, num_features + 1):
            size = np.sum(labeled_array == i)
            component_sizes.append((i, size))

        component_sizes.sort(key=lambda x: x[1], reverse=True)

        if len(component_sizes) >= 2:
            comp1_id, comp1_size = component_sizes[0]
            comp2_id, comp2_size = component_sizes[1]

            if comp2_size > 0.20 * comp1_size:
                print(f"Found 2 large components:")
                print(f"  Component 1: {comp1_size:,} voxels")
                print(f"  Component 2: {comp2_size:,} voxels")
                print(f"  → Force connecting them (likely left + right lung)")

                comp1_coords = np.argwhere(labeled_array == comp1_id)
                comp2_coords = np.argwhere(labeled_array == comp2_id)

                comp1_sample = comp1_coords[::max(1, len(comp1_coords)//100)]
                comp2_sample = comp2_coords[::max(1, len(comp2_coords)//100)]

                min_distance = float('inf')
                best_p1, best_p2 = None, None

                for p1 in comp1_sample:
                    for p2 in comp2_sample:
                        dist_vector = (p1 - p2) * np.array([self.spacing[2], self.spacing[1], self.spacing[0]])
                        distance = np.linalg.norm(dist_vector)
                        if distance < min_distance:
                            min_distance = distance
                            best_p1, best_p2 = p1, p2

                print(f"  Minimum distance between components: {min_distance:.1f} mm")
                print(f"  Creating bridge...")

                self._create_bridge_to_skeleton(best_p1, best_p2, skeleton_array)

                print(f"  ✓ Bridge created - both lungs now connected!")
            else:
                print(f"2nd component too small ({comp2_size}/{comp1_size} = {comp2_size/comp1_size:.1%}) - likely noise")

        self.skeleton = skeleton_array.astype(bool)

    def _create_bridge_to_skeleton(self, point1, point2, skeleton_array):
        """Creates a bridge between two points in 3D space and adds it to skeleton"""
        p1 = np.array(point1, dtype=int)
        p2 = np.array(point2, dtype=int)

        distance_pixels = np.linalg.norm(p2 - p1)
        distance_mm = distance_pixels * np.mean(self.spacing)
        num_points = max(3, int(distance_mm / 2))

        for t in np.linspace(0, 1, num_points):
            intermediate_point = p1 + t * (p2 - p1)
            z, y, x = intermediate_point.astype(int)

            if (0 <= z < skeleton_array.shape[0] and 
                0 <= y < skeleton_array.shape[1] and 
                0 <= x < skeleton_array.shape[2]):
                skeleton_array[z, y, x] = 1

        for point in [p1, p2]:
            z, y, x = point[0], point[1], point[2]
            for dz in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        nz, ny, nx = z + dz, y + dy, x + dx
                        if (0 <= nz < skeleton_array.shape[0] and 
                            0 <= ny < skeleton_array.shape[1] and 
                            0 <= nx < skeleton_array.shape[2]):
                            skeleton_array[nz, ny, nx] = 1

    def build_graph(self):
        """Builds graph from skeleton using skan"""
        print("\n=== Graph Construction ===")

        if self.skeleton is None:
            raise ValueError("Compute skeleton first with compute_skeleton()")

        spacing_zyx = (self.spacing[2], self.spacing[1], self.spacing[0])
        self.skeleton_obj = Skeleton(self.skeleton, spacing=spacing_zyx)

        self.branch_data = summarize(self.skeleton_obj)

        print(f"Number of identified branches: {len(self.branch_data)}")
        print(f"Number of junctions: {self.skeleton_obj.n_paths}")

        self.graph = self._create_networkx_graph()

        return self.graph

    def _create_networkx_graph(self):
        """Creates a NetworkX graph from skeleton"""
        G = nx.Graph()

        coordinates = self.skeleton_obj.coordinates
        for idx in range(len(coordinates)):
            pos = coordinates[idx]
            G.add_node(idx, pos=pos)

        for _, row in self.branch_data.iterrows():
            node1 = int(row['node-id-src'])
            node2 = int(row['node-id-dst'])
            length = row['branch-distance']
            G.add_edge(node1, node2, length=length, branch_type=row['branch-type'])

        num_components = nx.number_connected_components(G)
        if num_components > 1:
            print(f"\n🚨 CRITICAL: Graph has {num_components} disconnected components!")
            components = list(nx.connected_components(G))
            for i, comp in enumerate(sorted(components, key=len, reverse=True)):
                print(f"   Component {i+1}: {len(comp)} nodes")
        else:
            print(f"✓ Graph is fully connected ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")

        return G

    def set_initial_carina_coords(self, coords):
        """
        Set initial carina coordinates (voxel coordinates: z, y, x).
        These coordinates typically come from an external detector (e.g. test_robust).
        """
        if coords is None:
            self.initial_carina_coords = None
        else:
            self.initial_carina_coords = tuple(coords)

    def load_carina_from_json(self, json_path='carina_coordinates.json'):
        """
        Loads carina coordinates from a JSON file. Supported formats:
        - {'voxel_coordinates': {'z':..,'y':..,'x':..}}
        - [z, y, x] or {'carina': [z,y,x]} or simple list/tuple
        Returns the coordinates (z,y,x) or None if not found.
        """
        if not os.path.exists(json_path):
            return None

        try:
            with open(json_path, 'r') as jf:
                data = json.load(jf)

            if isinstance(data, dict) and 'voxel_coordinates' in data:
                v = data['voxel_coordinates']
                z = int(v.get('z', 0))
                y = int(v.get('y', 0))
                x = int(v.get('x', 0))
                self.initial_carina_coords = (z, y, x)
                return self.initial_carina_coords

            if isinstance(data, (list, tuple)) and len(data) >= 3:
                self.initial_carina_coords = (int(data[0]), int(data[1]), int(data[2]))
                return self.initial_carina_coords

            for key in ('carina', 'position', 'coordinates'):
                if key in data and isinstance(data[key], (list, tuple)) and len(data[key]) >= 3:
                    self.initial_carina_coords = (int(data[key][0]), int(data[key][1]), int(data[key][2]))
                    return self.initial_carina_coords

        except Exception as e:
            print(f"Warning: could not load carina JSON '{json_path}': {e}")

        return None

    def identify_carina(self):
        """
        Identifica la carina come il nodo con il diametro medio più grande tra i punti di biforcazione
        Restituisce anche le coordinate della carina
        """
        print("\n=== CARINA IDENTIFICATION BY DIAMETER ===")

        if self.graph is None:
            raise ValueError("Build graph first with build_graph()")

        if getattr(self, 'initial_carina_coords', None) is not None:
            target = np.array(self.initial_carina_coords, dtype=float)
            min_dist = float('inf')
            best_node = None
            for node in self.graph.nodes():
                pos = np.array(self.graph.nodes[node]['pos'], dtype=float)
                d = np.linalg.norm(pos - target)
                if d < min_dist:
                    min_dist = d
                    best_node = node

            if best_node is None:
                print("Warning: could not match provided carina coords to graph nodes")
            else:
                self.carina_node = best_node
                carina_position = self.graph.nodes[self.carina_node]['pos']
                z_i, y_i, x_i = int(round(carina_position[0])), int(round(carina_position[1])), int(round(carina_position[2]))
                if hasattr(self, 'distance_transform') and self.distance_transform is not None and \
                   0 <= z_i < self.distance_transform.shape[0] and \
                   0 <= y_i < self.distance_transform.shape[1] and \
                   0 <= x_i < self.distance_transform.shape[2]:
                    diameter_at_node = self.distance_transform[z_i, y_i, x_i] * 2
                else:
                    diameter_at_node = 0.0

                neighbor_diams = []
                for nb in self.graph.neighbors(self.carina_node):
                    edge = tuple(sorted([self.carina_node, nb]))
                    if 'diameter' in self.graph.edges[edge]:
                        neighbor_diams.append(self.graph.edges[edge]['diameter'])

                avg_branch_diameter = np.mean(neighbor_diams) if neighbor_diams else diameter_at_node

                node_info = [{
                    'node': self.carina_node,
                    'degree': self.graph.degree(self.carina_node),
                    'diameter_at_node': diameter_at_node,
                    'avg_branch_diameter': avg_branch_diameter,
                    'z': z_i,
                    'y': y_i,
                    'x': x_i,
                    'position': carina_position
                }]

                print(f"Using external carina coords mapped to node {self.carina_node} (dist {min_dist:.2f} voxels)")

                self.carina_info = {
                    'node_id': self.carina_node,
                    'position': carina_position,
                    'avg_branch_diameter': avg_branch_diameter,
                    'diameter_at_node': diameter_at_node,
                    'degree': node_info[0]['degree'],
                    'coordinates_voxel': (node_info[0]['z'], node_info[0]['y'], node_info[0]['x']),
                    'coordinates_world': (carina_position[2], carina_position[1], carina_position[0])
                }

                return self.carina_node, avg_branch_diameter, carina_position

        candidate_nodes = [node for node in self.graph.nodes() 
                        if self.graph.degree(node) >= 2]

        if len(candidate_nodes) == 0:
            print("WARNING: No bifurcation nodes found!")
            degrees = dict(self.graph.degree())
            self.carina_node = max(degrees, key=degrees.get)
            carina_diameter = 0
            carina_position = self.graph.nodes[self.carina_node]['pos']
        else:
            node_info = []
            for node in candidate_nodes:
                pos = self.graph.nodes[node]['pos']
                z, y, x = int(pos[0]), int(pos[1]), int(pos[2])

                if (0 <= z < self.distance_transform.shape[0] and
                    0 <= y < self.distance_transform.shape[1] and
                    0 <= x < self.distance_transform.shape[2]):
                    diameter = self.distance_transform[z, y, x] * 2
                else:
                    diameter = 0

                connected_branches = []
                for neighbor in self.graph.neighbors(node):
                    edge = tuple(sorted([node, neighbor]))
                    if 'diameter' in self.graph.edges[edge]:
                        connected_branches.append(self.graph.edges[edge]['diameter'])

                avg_branch_diameter = np.mean(connected_branches) if connected_branches else diameter

                node_info.append({
                    'node': node,
                    'degree': self.graph.degree(node),
                    'diameter_at_node': diameter,
                    'avg_branch_diameter': avg_branch_diameter,
                    'z': z,
                    'y': y,
                    'x': x,
                    'position': pos
                })

            node_info.sort(key=lambda x: (-x['avg_branch_diameter'], -x['z']))

            self.carina_node = node_info[0]['node']
            carina_diameter = node_info[0]['avg_branch_diameter']
            carina_position = node_info[0]['position']

            print(f"Carina identified: Node {self.carina_node}")
            print(f"  Degree: {node_info[0]['degree']}")
            print(f"  Average branch diameter: {node_info[0]['avg_branch_diameter']:.2f} mm")
            print(f"  Diameter at node: {node_info[0]['diameter_at_node']:.2f} mm")
            print(f"  Position (z,y,x): ({node_info[0]['z']}, {node_info[0]['y']:.1f}, {node_info[0]['x']:.1f})")
            print(f"  Position (world coordinates): ({carina_position[2]:.1f}, {carina_position[1]:.1f}, {carina_position[0]:.1f})")

        self.carina_info = {
            'node_id': self.carina_node,
            'position': carina_position,
            'avg_branch_diameter': carina_diameter,
            'diameter_at_node': node_info[0]['diameter_at_node'] if len(candidate_nodes) > 0 else 0,
            'degree': node_info[0]['degree'] if len(candidate_nodes) > 0 else self.graph.degree(self.carina_node),
            'coordinates_voxel': (node_info[0]['z'], node_info[0]['y'], node_info[0]['x']) if len(candidate_nodes) > 0 else (int(carina_position[0]), int(carina_position[1]), int(carina_position[2])),
            'coordinates_world': (carina_position[2], carina_position[1], carina_position[0])
        }

        return self.carina_node, carina_diameter, carina_position

    def visualize_with_carina(self, save_path=None, figsize=(16, 12)):
        """
        Visualizzazione del grafo con la carina evidenziata con un PUNTINO ROSSO
        """
        print("\n=== GRAPH VISUALIZATION WITH CARINA ===")

        if self.graph is None:
            raise ValueError("Build graph first with build_graph()")

        if not hasattr(self, 'carina_node'):
            self.identify_carina()

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        all_nodes = list(self.graph.nodes())
        node_positions = np.array([self.graph.nodes[node]['pos'] for node in all_nodes])

        node_degrees = [self.graph.degree(node) for node in all_nodes]
        node_colors = plt.cm.viridis(np.array(node_degrees) / max(node_degrees))

        ax.scatter(node_positions[:, 2], node_positions[:, 1], node_positions[:, 0],
                c=node_colors, s=50, alpha=0.7, edgecolors='black', linewidths=0.5)

        for edge in self.graph.edges():
            pos1 = self.graph.nodes[edge[0]]['pos']
            pos2 = self.graph.nodes[edge[1]]['pos']
            ax.plot([pos1[2], pos2[2]], [pos1[1], pos2[1]], [pos1[0], pos2[0]],
                color='gray', linewidth=1, alpha=0.5)

        carina_pos = self.graph.nodes[self.carina_node]['pos']
        ax.scatter(carina_pos[2], carina_pos[1], carina_pos[0],
                c='red', s=200, marker='o', edgecolors='black', linewidths=2,
                label=f'Carina (Ø={self.carina_info["avg_branch_diameter"]:.1f}mm)')

        coord_text = f'CARINA\nX:{carina_pos[2]:.1f}\nY:{carina_pos[1]:.1f}\nZ:{carina_pos[0]:.1f}'
        ax.text(carina_pos[2], carina_pos[1], carina_pos[0] + 10, 
            coord_text, fontsize=10, fontweight='bold', color='red', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        ax.set_xlabel('X (voxel)')
        ax.set_ylabel('Y (voxel)')
        ax.set_zlabel('Z (voxel)')
        ax.set_title('Bronchial Tree Graph with Carina Highlighted\n(Red dot = Carina, largest diameter bifurcation)')
        ax.legend()

        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                norm=plt.Normalize(vmin=min(node_degrees), vmax=max(node_degrees)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Node Degree', rotation=270, labelpad=20)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Graph with carina saved: {save_path}")

        plt.show()
        return fig, ax

    def assign_generations_weibel(self):
        """
        Assigns Weibel generation numbers to each branch using breadth-first search
        from the carina (generation 0).

        Weibel model:
        - Generation 0: Main bronchi (starting from carina)
        - Generation 1+: Each bifurcation increases generation by 1
        - Typical human lungs have ~23 generations
        """
        print("\n=== WEIBEL GENERATION ASSIGNMENT ===")

        if self.carina_node is None:
            self.identify_carina()

        if self.graph is None:
            raise ValueError("Build graph first")

        node_generations = {self.carina_node: -1}
        branch_generations = {}

        queue = deque([(self.carina_node, -1)])
        visited = {self.carina_node}

        while queue:
            current_node, current_gen = queue.popleft()

            for neighbor in self.graph.neighbors(current_node):
                if neighbor not in visited:
                    if self.graph.degree(current_node) >= 3:
                        next_gen = current_gen + 1
                    else:
                        next_gen = current_gen

                    node_generations[neighbor] = next_gen
                    visited.add(neighbor)
                    queue.append((neighbor, next_gen))

                    edge = tuple(sorted([current_node, neighbor]))
                    branch_generations[edge] = next_gen

        unvisited_nodes = set(self.graph.nodes()) - visited
        if len(unvisited_nodes) > 0:
            print(f"\n⚠️  WARNING: Found {len(unvisited_nodes)} nodes disconnected from carina!")
            print(f"   These likely belong to the other lung (disconnected in skeleton)")
            print(f"   Assigning them to generation 0 and processing separately...")

            disconnected_components = []
            remaining = unvisited_nodes.copy()

            while remaining:
                start_node = remaining.pop()
                component = {start_node}
                comp_queue = deque([start_node])

                while comp_queue:
                    node = comp_queue.popleft()
                    for neighbor in self.graph.neighbors(node):
                        if neighbor in remaining:
                            remaining.remove(neighbor)
                            component.add(neighbor)
                            comp_queue.append(neighbor)

                disconnected_components.append(component)

            print(f"   Found {len(disconnected_components)} disconnected component(s)")

            for comp_idx, component in enumerate(disconnected_components):
                print(f"   Component {comp_idx+1}: {len(component)} nodes")

                comp_carina = max(component, key=lambda n: self.graph.degree(n))

                node_generations[comp_carina] = -1
                comp_queue = deque([(comp_carina, -1)])
                comp_visited = {comp_carina}

                while comp_queue:
                    current_node, current_gen = comp_queue.popleft()

                    for neighbor in self.graph.neighbors(current_node):
                        if neighbor in component and neighbor not in comp_visited:
                            if self.graph.degree(current_node) >= 3:
                                next_gen = current_gen + 1
                            else:
                                next_gen = current_gen

                            node_generations[neighbor] = next_gen
                            comp_visited.add(neighbor)
                            comp_queue.append((neighbor, next_gen))

                            edge = tuple(sorted([current_node, neighbor]))
                            branch_generations[edge] = next_gen

        self.generation_assignments = {
            'nodes': node_generations,
            'branches': branch_generations
        }

        max_generation = max(node_generations.values())
        generation_counts = {}
        for gen in node_generations.values():
            generation_counts[gen] = generation_counts.get(gen, 0) + 1

        print(f"Generation assignment complete:")
        print(f"  Maximum generation: {max_generation}")
        print(f"  Number of nodes per generation:")
        for gen in sorted(generation_counts.keys()):
            print(f"    Gen {gen}: {generation_counts[gen]} nodes")

        return self.generation_assignments

    def calculate_branch_lengths(self):
        """
        Calculates the actual length of each branch by measuring the path
        through the skeleton coordinates
        """
        print("\n=== Branch Length Calculation ===")

        if self.branch_data is None:
            raise ValueError("Build graph first with build_graph()")

        branch_lengths = []

        for idx, row in self.branch_data.iterrows():
            branch_idx = idx

            try:
                coords_indices = self.skeleton_obj.path_coordinates(branch_idx)

                if len(coords_indices) < 2:
                    length_mm = 0.0
                else:
                    length_mm = 0.0
                    for i in range(len(coords_indices) - 1):
                        p1 = coords_indices[i]
                        p2 = coords_indices[i + 1]

                        dz = (p2[0] - p1[0]) * self.spacing[2]
                        dy = (p2[1] - p1[1]) * self.spacing[1]
                        dx = (p2[2] - p1[2]) * self.spacing[0]

                        segment_length = np.sqrt(dx**2 + dy**2 + dz**2)
                        length_mm += segment_length

                branch_lengths.append({
                    'branch_id': branch_idx,
                    'length_mm': length_mm,
                    'num_points': len(coords_indices),
                    'skan_distance_mm': row['branch-distance']
                })

            except Exception as e:
                print(f"Warning: Error calculating length for branch {branch_idx}: {e}")
                branch_lengths.append({
                    'branch_id': branch_idx,
                    'length_mm': 0.0,
                    'num_points': 0,
                    'skan_distance_mm': row['branch-distance']
                })

        self.branch_lengths_df = pd.DataFrame(branch_lengths)

        print(f"\nBranch Length Statistics (mm):")
        print(f"  Total branches: {len(self.branch_lengths_df)}")
        print(f"  Mean length: {self.branch_lengths_df['length_mm'].mean():.2f}")
        print(f"  Median length: {self.branch_lengths_df['length_mm'].median():.2f}")
        print(f"  Min length: {self.branch_lengths_df['length_mm'].min():.2f}")
        print(f"  Max length: {self.branch_lengths_df['length_mm'].max():.2f}")
        print(f"  Total airway length: {self.branch_lengths_df['length_mm'].sum():.2f}")

        return self.branch_lengths_df

    def analyze_diameters(self, use_robust_calculation=True, exclude_terminal_percent=10):
        """Calculates diameters along each branch using distance transform

        Args:
            use_robust_calculation: If True, uses 75th percentile instead of mean to avoid
                                   overestimation from terminal artifacts (blobs)
            exclude_terminal_percent: Percentage of terminal points to exclude from calculation
                                     (default 10% to avoid terminal blob artifacts)
        """
        print("\n=== Diameter Analysis ===")

        if self.branch_data is None:
            raise ValueError("Build graph first with build_graph()")

        distance_transform_for_metrics = self.original_distance_transform
        if distance_transform_for_metrics is not None and distance_transform_for_metrics is not self.distance_transform:
            print("  Using ORIGINAL mask distance transform for accurate diameter measurements")

        if use_robust_calculation:
            print(f"  Using ROBUST calculation (75th percentile, excluding {exclude_terminal_percent}% terminal points)")

        diameters_list = []

        for idx, row in self.branch_data.iterrows():
            branch_idx = idx

            coords_indices = self.skeleton_obj.path_coordinates(branch_idx)

            distances = []
            for coord in coords_indices:
                z, y, x = coord
                if 0 <= z < distance_transform_for_metrics.shape[0] and \
                   0 <= y < distance_transform_for_metrics.shape[1] and \
                   0 <= x < distance_transform_for_metrics.shape[2]:
                    dist = distance_transform_for_metrics[z, y, x]
                    distances.append(dist * 2)

            if distances:
                if use_robust_calculation and len(distances) > 10:
                    n_exclude = max(1, int(len(distances) * exclude_terminal_percent / 100))
                    core_distances = distances[n_exclude:-n_exclude] if len(distances) > 2*n_exclude else distances

                    if len(core_distances) > 0:
                        diameter_mean = np.percentile(core_distances, 75)
                        diameter_std = np.std(core_distances)
                        diameter_min = np.min(core_distances)
                        diameter_max = np.percentile(core_distances, 95)
                    else:
                        diameter_mean = np.percentile(distances, 75)
                        diameter_std = np.std(distances)
                        diameter_min = np.min(distances)
                        diameter_max = np.percentile(distances, 95)
                else:
                    diameter_mean = np.mean(distances)
                    diameter_std = np.std(distances)
                    diameter_min = np.min(distances)
                    diameter_max = np.max(distances)
            else:
                diameter_mean = diameter_std = diameter_min = diameter_max = 0

            diameters_list.append({
                'branch_id': branch_idx,
                'diameter_mean_mm': diameter_mean,
                'diameter_std_mm': diameter_std,
                'diameter_min_mm': diameter_min,
                'diameter_max_mm': diameter_max
            })

        self.diameters_df = pd.DataFrame(diameters_list)

        print(f"\nDiameter Statistics (mm):")
        print(f"  Mean: {self.diameters_df['diameter_mean_mm'].mean():.2f}")
        print(f"  Min: {self.diameters_df['diameter_mean_mm'].min():.2f}")
        print(f"  Max: {self.diameters_df['diameter_mean_mm'].max():.2f}")

        return self.diameters_df

    def merge_branch_metrics(self):
        """
        Merges diameters, lengths, and generation info into comprehensive branch dataframe
        """
        print("\n=== Merging Branch Metrics ===")

        if not hasattr(self, 'diameters_df'):
            raise ValueError("Calculate diameters first with analyze_diameters()")

        if not hasattr(self, 'branch_lengths_df'):
            raise ValueError("Calculate lengths first with calculate_branch_lengths()")

        self.branch_metrics_df = pd.merge(
            self.diameters_df,
            self.branch_lengths_df[['branch_id', 'length_mm', 'num_points']],
            on='branch_id',
            how='inner'
        )

        if self.generation_assignments is not None:
            generations = []
            for idx, row in self.branch_data.iterrows():
                node_src = int(row['node-id-src'])
                node_dst = int(row['node-id-dst'])
                edge = tuple(sorted([node_src, node_dst]))

                gen = self.generation_assignments['branches'].get(edge, np.nan)
                generations.append(gen)

            self.branch_metrics_df['generation'] = generations

        self.branch_metrics_df['volume_mm3'] = (
            np.pi * (self.branch_metrics_df['diameter_mean_mm'] / 2) ** 2 * 
            self.branch_metrics_df['length_mm']
        )

        self.branch_metrics_df['surface_area_mm2'] = (
            np.pi * self.branch_metrics_df['diameter_mean_mm'] * 
            self.branch_metrics_df['length_mm']
        )

        print(f"\nMerged metrics for {len(self.branch_metrics_df)} branches")
        print(f"\nColumns: {list(self.branch_metrics_df.columns)}")

        return self.branch_metrics_df

    def analyze_weibel_tapering(self):
        """
        Analyzes diameter tapering across generations according to Weibel model.
        Computes statistics for each generation and analyzes tapering ratios.
        """
        print("\n" + "="*60)
        print("WEIBEL GENERATION TAPERING ANALYSIS")
        print("="*60)

        if not hasattr(self, 'branch_metrics_df'):
            raise ValueError("Merge branch metrics first")

        if 'generation' not in self.branch_metrics_df.columns:
            raise ValueError("Assign generations first with assign_generations_weibel()")

        generation_stats = []

        for gen in sorted(self.branch_metrics_df['generation'].dropna().unique()):
            gen_data = self.branch_metrics_df[self.branch_metrics_df['generation'] == gen]

            if len(gen_data) == 0:
                continue

            stats = {
                'generation': int(gen),
                'n_branches': len(gen_data),
                'diameter_mean_mm': gen_data['diameter_mean_mm'].mean(),
                'diameter_std_mm': gen_data['diameter_mean_mm'].std(),
                'diameter_median_mm': gen_data['diameter_mean_mm'].median(),
                'diameter_min_mm': gen_data['diameter_mean_mm'].min(),
                'diameter_max_mm': gen_data['diameter_mean_mm'].max(),
                'length_mean_mm': gen_data['length_mm'].mean(),
                'length_std_mm': gen_data['length_mm'].std(),
                'length_median_mm': gen_data['length_mm'].median(),
                'total_volume_mm3': gen_data['volume_mm3'].sum(),
                'total_surface_area_mm2': gen_data['surface_area_mm2'].sum()
            }

            generation_stats.append(stats)

        self.weibel_analysis_df = pd.DataFrame(generation_stats)

        tapering_ratios = []
        for i in range(len(self.weibel_analysis_df) - 1):
            current_gen = self.weibel_analysis_df.iloc[i]
            next_gen = self.weibel_analysis_df.iloc[i + 1]
            ratio = next_gen['diameter_mean_mm'] / current_gen['diameter_mean_mm']
            tapering_ratios.append({
                'from_generation': int(current_gen['generation']),
                'to_generation': int(next_gen['generation']),
                'diameter_ratio': ratio
            })

        self.tapering_ratios_df = pd.DataFrame(tapering_ratios)

        print(f"\nGeneration statistics:")
        print(self.weibel_analysis_df.to_string(index=False))

        print(f"\n{'='*60}")
        print("TAPERING RATIOS (Weibel Model)")
        print(f"{'='*60}")
        print("\nDiameter reduction ratios between consecutive generations:")
        print(self.tapering_ratios_df.to_string(index=False))

        if len(self.tapering_ratios_df) > 0:
            mean_ratio = self.tapering_ratios_df['diameter_ratio'].mean()
            print(f"\nMean tapering ratio: {mean_ratio:.3f}")
            print(f"(Weibel's theoretical model predicts ~0.793 for symmetric branching)")

            weibel_expected = 2**(-1/3)
            print(f"\nWeibel's theoretical ratio: {weibel_expected:.3f}")
            print(f"Observed mean ratio: {mean_ratio:.3f}")
            print(f"Difference: {abs(mean_ratio - weibel_expected):.3f}")

        return self.weibel_analysis_df, self.tapering_ratios_df

    def compute_advanced_metrics(self):
        """
        Calcola metriche avanzate per correlazione con FVC/percent:
        1. Distribuzione volume per generazione (centrale vs periferico)
        2. Airway-to-vessel ratio (se disponibile segmentazione vascolare)
        3. Peripheral vs central airway ratio
        4. Tortuosità media delle vie aeree
        5. Wall thickness estimation
        """
        print("\n" + "="*70)
        print("ADVANCED CLINICAL METRICS CALCULATION")
        print("="*70)

        if not hasattr(self, 'branch_metrics_df'):
            raise ValueError("Run merge_branch_metrics() first")

        if 'generation' not in self.branch_metrics_df.columns:
            raise ValueError("Run assign_generations_weibel() first")

        metrics = {}

        print("\n[Metric 1] Volume Distribution by Generation...")

        volume_per_generation = self.branch_metrics_df.groupby('generation')['volume_mm3'].sum()

        metrics['volume_per_generation'] = volume_per_generation.to_dict()
        metrics['total_volume_mm3'] = volume_per_generation.sum()

        all_generations = set(range(int(self.branch_metrics_df['generation'].min()), 
                                    int(self.branch_metrics_df['generation'].max()) + 1))
        present_generations = set(volume_per_generation.index.astype(int))
        missing_generations = sorted(all_generations - present_generations)

        metrics['missing_generations'] = missing_generations
        metrics['generation_coverage'] = len(present_generations) / len(all_generations) if all_generations else 0

        print(f"  Total airway volume: {metrics['total_volume_mm3']:.2f} mm³")
        print(f"  Generations covered: {len(present_generations)}/{len(all_generations)} "
            f"({metrics['generation_coverage']*100:.1f}%)")

        if missing_generations:
            print(f"  ⚠ Missing generations: {missing_generations[:10]}{'...' if len(missing_generations) > 10 else ''}")

        print("\n[Metric 2] Peripheral vs Central Airway Analysis...")


        max_generation = self.branch_metrics_df['generation'].max()
        peripheral_threshold = 10

        peripheral_branches = self.branch_metrics_df[self.branch_metrics_df['generation'] > 10]

        if len(peripheral_branches) == 0 or len(peripheral_branches) < 5:
            if max_generation > 8:
                peripheral_threshold = 8
                print(f"  ⚠ Using adaptive threshold: gen > {peripheral_threshold} (max gen = {max_generation:.0f})")
            elif max_generation > 7:
                peripheral_threshold = 7
                print(f"  ⚠ Using adaptive threshold: gen > {peripheral_threshold} (max gen = {max_generation:.0f})")
            else:
                peripheral_threshold = 7
                print(f"  ⚠ Very limited airway tree: gen > {peripheral_threshold} (max gen = {max_generation:.0f})")

        central_branches = self.branch_metrics_df[self.branch_metrics_df['generation'] <= 7]
        intermediate_branches = self.branch_metrics_df[
            (self.branch_metrics_df['generation'] > 7) & 
            (self.branch_metrics_df['generation'] <= peripheral_threshold)
        ]
        peripheral_branches = self.branch_metrics_df[self.branch_metrics_df['generation'] > peripheral_threshold]

        central_volume = central_branches['volume_mm3'].sum()
        intermediate_volume = intermediate_branches['volume_mm3'].sum()
        peripheral_volume = peripheral_branches['volume_mm3'].sum()

        p_c_ratio = peripheral_volume / central_volume if central_volume > 0 else 0

        metrics['central_volume_mm3'] = central_volume
        metrics['intermediate_volume_mm3'] = intermediate_volume
        metrics['peripheral_volume_mm3'] = peripheral_volume
        metrics['peripheral_to_central_ratio'] = p_c_ratio

        metrics['central_branch_count'] = len(central_branches)
        metrics['intermediate_branch_count'] = len(intermediate_branches)
        metrics['peripheral_branch_count'] = len(peripheral_branches)
        metrics['peripheral_to_central_branch_ratio'] = (
            len(peripheral_branches) / len(central_branches) if len(central_branches) > 0 else 0
        )

        print(f"  Central airways (gen 0-7):")
        print(f"    Volume: {central_volume:.2f} mm³")
        print(f"    Branches: {len(central_branches)}")
        print(f"  Intermediate airways (gen 8-{peripheral_threshold}):")
        print(f"    Volume: {intermediate_volume:.2f} mm³")
        print(f"    Branches: {len(intermediate_branches)}")
        print(f"  Peripheral airways (gen >{peripheral_threshold}):")
        print(f"    Volume: {peripheral_volume:.2f} mm³")
        print(f"    Branches: {len(peripheral_branches)}")
        print(f"  Peripheral/Central volume ratio: {p_c_ratio:.3f}")
        print(f"  Peripheral/Central branch ratio: {metrics['peripheral_to_central_branch_ratio']:.3f}")

        if p_c_ratio < 0.15:
            print(f"  ⚠ VERY LOW P/C ratio - severe peripheral airway loss (fibrosis)")
        elif p_c_ratio < 0.3:
            print(f"  ⚠ LOW P/C ratio - moderate peripheral airway loss")
        elif p_c_ratio < 0.5:
            print(f"  → Borderline - mild peripheral involvement")
        elif p_c_ratio > 1.0:
            print(f"  ✓ EXCELLENT P/C ratio - well-preserved peripheral airways")
        else:
            print(f"  ✓ Normal P/C ratio range")

        print("\n[Metric 2b] Enhanced Peripheral Metrics...")

        metrics['peripheral_volume_percent'] = (peripheral_volume / metrics['total_volume_mm3'] * 100) if metrics['total_volume_mm3'] > 0 else 0

        if len(peripheral_branches) > 0 and 'diameter_mean_mm' in peripheral_branches.columns:
            metrics['mean_peripheral_diameter_mm'] = peripheral_branches['diameter_mean_mm'].mean()
            metrics['median_peripheral_diameter_mm'] = peripheral_branches['diameter_mean_mm'].median()
            metrics['std_peripheral_diameter_mm'] = peripheral_branches['diameter_mean_mm'].std()
        else:
            metrics['mean_peripheral_diameter_mm'] = np.nan
            metrics['median_peripheral_diameter_mm'] = np.nan
            metrics['std_peripheral_diameter_mm'] = np.nan

        if len(peripheral_branches) > 0:
            metrics['mean_peripheral_branch_volume_mm3'] = peripheral_branches['volume_mm3'].mean()
            metrics['median_peripheral_branch_volume_mm3'] = peripheral_branches['volume_mm3'].median()
        else:
            metrics['mean_peripheral_branch_volume_mm3'] = np.nan
            metrics['median_peripheral_branch_volume_mm3'] = np.nan

        metrics['peripheral_branch_density'] = len(peripheral_branches) / peripheral_volume if peripheral_volume > 0 else 0

        if len(central_branches) > 0 and 'diameter_mean_mm' in central_branches.columns and not np.isnan(metrics['mean_peripheral_diameter_mm']):
            mean_central_diameter = central_branches['diameter_mean_mm'].mean()
            metrics['mean_central_diameter_mm'] = mean_central_diameter
            metrics['central_to_peripheral_diameter_ratio'] = mean_central_diameter / metrics['mean_peripheral_diameter_mm'] if metrics['mean_peripheral_diameter_mm'] > 0 else np.nan
        else:
            metrics['mean_central_diameter_mm'] = np.nan
            metrics['central_to_peripheral_diameter_ratio'] = np.nan

        if 'diameter_mean_mm' in self.branch_metrics_df.columns:
            all_diameters = self.branch_metrics_df['diameter_mean_mm'].dropna()
            if len(all_diameters) > 0:
                metrics['diameter_cv'] = all_diameters.std() / all_diameters.mean() if all_diameters.mean() > 0 else np.nan
                metrics['diameter_mean_all'] = all_diameters.mean()
                metrics['diameter_std_all'] = all_diameters.std()
            else:
                metrics['diameter_cv'] = np.nan
                metrics['diameter_mean_all'] = np.nan
                metrics['diameter_std_all'] = np.nan

        print(f"  Peripheral volume %: {metrics['peripheral_volume_percent']:.2f}%")
        print(f"  Mean peripheral diameter: {metrics['mean_peripheral_diameter_mm']:.3f} mm")
        print(f"  Mean peripheral branch volume: {metrics['mean_peripheral_branch_volume_mm3']:.2f} mm³")
        print(f"  Peripheral branch density: {metrics['peripheral_branch_density']:.4f} branches/mm³")
        print(f"  Central/Peripheral diameter ratio: {metrics['central_to_peripheral_diameter_ratio']:.3f}")
        print(f"  Diameter CV (all airways): {metrics['diameter_cv']:.3f}")

        print("\n[Metric 3] Airway Tortuosity Analysis...")


        tortuosity_values = []

        for idx, row in self.branch_metrics_df.iterrows():
            if pd.isna(row.get('distance_from_carina_proximal_mm')) or pd.isna(row.get('distance_from_carina_distal_mm')):
                continue

            branch_length = row['length_mm']

            try:
                proximal_node = int(row['proximal_node'])
                distal_node = int(row['distal_node'])

                pos_prox = np.array(self.graph.nodes[proximal_node]['pos'])
                pos_dist = np.array(self.graph.nodes[distal_node]['pos'])

                euclidean_dist = np.linalg.norm(
                    (pos_dist - pos_prox) * np.array([self.spacing[2], self.spacing[1], self.spacing[0]])
                )

                if euclidean_dist > 0.5:
                    tortuosity = branch_length / euclidean_dist
                    if 0.8 < tortuosity < 5.0:
                        tortuosity_values.append(tortuosity)

            except Exception:
                continue

        if tortuosity_values:
            metrics['mean_tortuosity'] = np.mean(tortuosity_values)
            metrics['median_tortuosity'] = np.median(tortuosity_values)
            metrics['std_tortuosity'] = np.std(tortuosity_values)

            print(f"  Mean tortuosity: {metrics['mean_tortuosity']:.3f}")
            print(f"  Median tortuosity: {metrics['median_tortuosity']:.3f}")

            if metrics['mean_tortuosity'] > 1.5:
                print(f"  ⚠ High tortuosity suggests airway distortion (fibrosis)")
            else:
                print(f"  ✓ Normal tortuosity range")
        else:
            metrics['mean_tortuosity'] = np.nan
            print("  Could not calculate tortuosity (insufficient data)")

        print("\n[Metric 4] Airway Wall Thickness Estimation...")


        if hasattr(self, 'distance_transform') and self.distance_transform is not None:

            self.branch_metrics_df['diameter_to_length_ratio'] = (
                self.branch_metrics_df['diameter_mean_mm'] / 
                self.branch_metrics_df['length_mm']
            )

            metrics['mean_diameter_to_length_ratio'] = self.branch_metrics_df['diameter_to_length_ratio'].mean()

            print(f"  Mean diameter/length ratio: {metrics['mean_diameter_to_length_ratio']:.3f}")
            print(f"  Note: True wall thickness requires additional segmentation")
        else:
            metrics['mean_diameter_to_length_ratio'] = np.nan
            print("  Wall thickness estimation not available (distance transform missing)")

        print("\n[Metric 5] Generational Symmetry Analysis...")


        if hasattr(self, 'carina_node') and self.carina_node is not None:
            carina_neighbors = list(self.graph.neighbors(self.carina_node))

            if len(carina_neighbors) >= 2:
                neighbor_positions = [self.graph.nodes[n]['pos'] for n in carina_neighbors]


                z_coords = [pos[0] for pos in neighbor_positions]

                if len(set(z_coords)) > 1:
                    sorted_neighbors = sorted(zip(carina_neighbors, neighbor_positions), 
                                            key=lambda x: x[1][0])

                    mid_point = len(sorted_neighbors) // 2
                    left_side_nodes = [n[0] for n in sorted_neighbors[:mid_point]]
                    right_side_nodes = [n[0] for n in sorted_neighbors[mid_point:]]

                    from collections import deque

                    def count_descendants(start_nodes):
                        visited = set(start_nodes)
                        queue = deque(start_nodes)
                        count = 0

                        while queue:
                            node = queue.popleft()
                            count += 1
                            for neighbor in self.graph.neighbors(node):
                                if neighbor not in visited and neighbor != self.carina_node:
                                    visited.add(neighbor)
                                    queue.append(neighbor)
                        return count

                    left_count = count_descendants(left_side_nodes)
                    right_count = count_descendants(right_side_nodes)

                    symmetry_index = min(left_count, right_count) / max(left_count, right_count) if max(left_count, right_count) > 0 else 0

                    metrics['left_side_branch_count'] = left_count
                    metrics['right_side_branch_count'] = right_count
                    metrics['symmetry_index'] = symmetry_index

                    print(f"  Left side branches: {left_count}")
                    print(f"  Right side branches: {right_count}")
                    print(f"  Symmetry index: {symmetry_index:.3f}")

                    if symmetry_index < 0.5:
                        print(f"  ⚠ SEVERE asymmetry - strong unilateral disease")
                    elif symmetry_index < 0.7:
                        print(f"  ⚠ Moderate asymmetry - possible unilateral predominance")
                    elif symmetry_index < 0.85:
                        print(f"  → Mild asymmetry - within acceptable range")
                    else:
                        print(f"  ✓ Good symmetry")

                    if left_count < right_count * 0.7:
                        print(f"  → Left lung more affected ({left_count} vs {right_count} branches)")
                    elif right_count < left_count * 0.7:
                        print(f"  → Right lung more affected ({right_count} vs {left_count} branches)")
                else:
                    metrics['symmetry_index'] = np.nan
                    print("  Could not distinguish left/right (coordinates issue)")
            else:
                metrics['symmetry_index'] = np.nan
                print("  Insufficient branches from carina for symmetry analysis")
        else:
            metrics['symmetry_index'] = np.nan
            print("  Carina not identified, cannot compute symmetry")

        self.advanced_metrics = metrics

        print("\n" + "="*70)
        print("ADVANCED METRICS SUMMARY")
        print("="*70)
        print(f"Total Volume: {metrics['total_volume_mm3']:.2f} mm³")
        print(f"Peripheral/Central Ratio: {metrics['peripheral_to_central_ratio']:.3f}")
        print(f"Mean Tortuosity: {metrics.get('mean_tortuosity', 'N/A')}")
        print(f"Symmetry Index: {metrics.get('symmetry_index', 'N/A')}")
        print(f"Generation Coverage: {metrics['generation_coverage']*100:.1f}%")

        return metrics


    def save_advanced_metrics(self, output_dir):
        """
        Salva metriche avanzate in CSV e JSON
        """
        import os
        import json

        if not hasattr(self, 'advanced_metrics'):
            raise ValueError("Run compute_advanced_metrics() first")

        os.makedirs(output_dir, exist_ok=True)

        volume_df = pd.DataFrame([
            {'generation': gen, 'volume_mm3': vol}
            for gen, vol in self.advanced_metrics['volume_per_generation'].items()
        ])
        volume_path = os.path.join(output_dir, "volume_per_generation.csv")
        volume_df.to_csv(volume_path, index=False)
        print(f"✓ Saved: {volume_path}")

        metrics_json = {k: v for k, v in self.advanced_metrics.items() 
                        if k != 'volume_per_generation'}

        json_path = os.path.join(output_dir, "advanced_metrics.json")
        with open(json_path, 'w') as f:
            json.dump(metrics_json, f, indent=2, default=str)
        print(f"✓ Saved: {json_path}")

        pc_breakdown = pd.DataFrame([
            {
                'region': 'Central (Gen 0-10)',
                'volume_mm3': self.advanced_metrics['central_volume_mm3'],
                'branch_count': self.advanced_metrics['central_branch_count']
            },
            {
                'region': 'Intermediate (Gen 11-15)',
                'volume_mm3': self.advanced_metrics['intermediate_volume_mm3'],
                'branch_count': self.advanced_metrics['intermediate_branch_count']
            },
            {
                'region': 'Peripheral (Gen >15)',
                'volume_mm3': self.advanced_metrics['peripheral_volume_mm3'],
                'branch_count': self.advanced_metrics['peripheral_branch_count']
            }
        ])

        pc_path = os.path.join(output_dir, "peripheral_central_breakdown.csv")
        pc_breakdown.to_csv(pc_path, index=False)
        print(f"✓ Saved: {pc_path}")

        return metrics_json


    def plot_advanced_metrics(self, save_path=None):
        """
        Visualizzazione delle metriche avanzate
        """
        if not hasattr(self, 'advanced_metrics'):
            raise ValueError("Run compute_advanced_metrics() first")

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        ax1 = axes[0, 0]
        volume_data = self.advanced_metrics['volume_per_generation']
        generations = sorted(volume_data.keys())
        volumes = [volume_data[g] for g in generations]

        ax1.bar(generations, volumes, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Volume (mm³)')
        ax1.set_title('Volume Distribution by Generation')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3, axis='y')

        ax2 = axes[0, 1]
        regions = ['Central\n(Gen 0-10)', 'Intermediate\n(Gen 11-15)', 'Peripheral\n(Gen >15)']
        volumes_pc = [
            self.advanced_metrics['central_volume_mm3'],
            self.advanced_metrics['intermediate_volume_mm3'],
            self.advanced_metrics['peripheral_volume_mm3']
        ]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        ax2.bar(regions, volumes_pc, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Volume (mm³)')
        ax2.set_title('Peripheral vs Central Airway Volume')
        ax2.grid(True, alpha=0.3, axis='y')

        p_c_ratio = self.advanced_metrics['peripheral_to_central_ratio']
        ax2.text(0.5, 0.95, f'P/C Ratio: {p_c_ratio:.3f}', 
                transform=ax2.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=11, fontweight='bold')

        ax3 = axes[1, 0]
        branch_counts = [
            self.advanced_metrics['central_branch_count'],
            self.advanced_metrics['intermediate_branch_count'],
            self.advanced_metrics['peripheral_branch_count']
        ]

        ax3.bar(regions, branch_counts, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Number of Branches')
        ax3.set_title('Branch Count by Region')
        ax3.grid(True, alpha=0.3, axis='y')

        ax4 = axes[1, 1]
        ax4.axis('off')

        summary_text = f"""
    ADVANCED METRICS SUMMARY

    Total Airway Volume: {self.advanced_metrics['total_volume_mm3']:.2f} mm³

    Peripheral/Central Ratio: {p_c_ratio:.3f}
    {'  ⚠ LOW - Peripheral loss' if p_c_ratio < 0.2 else '  ✓ Normal range' if p_c_ratio < 0.6 else '  ✓ Well-preserved'}

    Mean Tortuosity: {self.advanced_metrics.get('mean_tortuosity', 'N/A')}
    {'  ⚠ HIGH - Airway distortion' if isinstance(self.advanced_metrics.get('mean_tortuosity'), float) and self.advanced_metrics.get('mean_tortuosity') > 1.5 else '  ✓ Normal' if isinstance(self.advanced_metrics.get('mean_tortuosity'), float) else ''}

    Symmetry Index: {self.advanced_metrics.get('symmetry_index', 'N/A')}
    {'  ⚠ ASYMMETRIC - Unilateral disease?' if isinstance(self.advanced_metrics.get('symmetry_index'), float) and self.advanced_metrics.get('symmetry_index') < 0.7 else '  ✓ Symmetric' if isinstance(self.advanced_metrics.get('symmetry_index'), float) else ''}

    Generation Coverage: {self.advanced_metrics['generation_coverage']*100:.1f}%

    Missing Generations: {len(self.advanced_metrics['missing_generations'])}
    """

        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved advanced metrics plot: {save_path}")

        plt.show()

        return fig
    def calculate_distances_from_carina(self):
        """
        Calculates cumulative PATH distances from the carina to each endpoint.
        Uses shortest path along the graph (not Euclidean distance).
        """
        print("\n=== DISTANCE FROM CARINA CALCULATION (PATH-BASED) ===")

        if self.graph is None:
            raise ValueError("Build graph first with build_graph()")

        if not hasattr(self, 'branch_metrics_df'):
            raise ValueError("Calculate branch metrics first with merge_branch_metrics()")

        if self.carina_node is None:
            self.identify_carina()

        print(f"Calculating path distances from carina (Node {self.carina_node})...")

        try:
            distances_from_carina = nx.single_source_dijkstra_path_length(
                self.graph, 
                self.carina_node, 
                weight='length'
            )
        except nx.NetworkXError as e:
            print(f"Error calculating distances: {e}")
            print("Using connected component containing carina.")
            connected_nodes = nx.node_connected_component(self.graph, self.carina_node)
            subgraph = self.graph.subgraph(connected_nodes)
            distances_from_carina = nx.single_source_dijkstra_path_length(
                subgraph, 
                self.carina_node, 
                weight='length'
            )

        print(f"Calculated path distances for {len(distances_from_carina)} nodes")

        branch_distances = []

        for idx, row in self.branch_data.iterrows():
            branch_id = idx
            node_src = int(row['node-id-src'])
            node_dst = int(row['node-id-dst'])
            branch_length = row['branch-distance']

            dist_src = distances_from_carina.get(node_src, np.nan)
            dist_dst = distances_from_carina.get(node_dst, np.nan)

            if not np.isnan(dist_src) and not np.isnan(dist_dst):
                proximal_distance = min(dist_src, dist_dst)
                distal_distance = max(dist_src, dist_dst)
                proximal_node = node_src if dist_src < dist_dst else node_dst
                distal_node = node_dst if dist_src < dist_dst else node_src
            elif not np.isnan(dist_src):
                proximal_distance = dist_src
                distal_distance = dist_src + branch_length
                proximal_node = node_src
                distal_node = node_dst
            elif not np.isnan(dist_dst):
                proximal_distance = dist_dst
                distal_distance = dist_dst + branch_length
                proximal_node = node_dst
                distal_node = node_src
            else:
                proximal_distance = np.nan
                distal_distance = np.nan
                proximal_node = node_src
                distal_node = node_dst

            branch_distances.append({
                'branch_id': branch_id,
                'proximal_node': proximal_node,
                'distal_node': distal_node,
                'distance_from_carina_proximal_mm': proximal_distance,
                'distance_from_carina_distal_mm': distal_distance,
                'branch_length_mm': branch_length
            })

        self.branch_distances_from_carina_df = pd.DataFrame(branch_distances)

        self.branch_metrics_df = pd.merge(
            self.branch_metrics_df,
            self.branch_distances_from_carina_df[['branch_id', 
                                                'proximal_node', 
                                                'distal_node',
                                                'distance_from_carina_proximal_mm',
                                                'distance_from_carina_distal_mm']],
            on='branch_id',
            how='left'
        )

        print("\nPath distance from carina statistics:")
        valid_distances = self.branch_metrics_df['distance_from_carina_distal_mm'].dropna()
        if len(valid_distances) > 0:
            print(f"  Min distance (proximal branches): {valid_distances.min():.2f} mm")
            print(f"  Max distance (distal branches): {valid_distances.max():.2f} mm")
            print(f"  Mean distance: {valid_distances.mean():.2f} mm")
            print(f"  Median distance: {valid_distances.median():.2f} mm")

        endpoint_nodes = [node for node in self.graph.nodes() if self.graph.degree(node) == 1]
        endpoint_distances = [distances_from_carina.get(node, np.nan) for node in endpoint_nodes]
        endpoint_distances = [d for d in endpoint_distances if not np.isnan(d)]

        if endpoint_distances:
            print(f"\nEndpoint statistics ({len(endpoint_distances)} endpoints):")
            print(f"  Closest endpoint to carina: {min(endpoint_distances):.2f} mm")
            print(f"  Farthest endpoint from carina: {max(endpoint_distances):.2f} mm")
            print(f"  Mean endpoint distance: {np.mean(endpoint_distances):.2f} mm")

        return self.branch_metrics_df

    def identify_bifurcations(self):
        """Identifies bifurcations (nodes with degree >= 3)"""
        print("\n=== Bifurcation Identification ===")

        if self.graph is None:
            raise ValueError("Build graph first with build_graph()")

        bifurcations = []

        for node in self.graph.nodes():
            degree = self.graph.degree(node)

            if degree >= 3:
                pos = self.graph.nodes[node]['pos']

                generation = self.generation_assignments['nodes'].get(node, np.nan) if self.generation_assignments else np.nan

                connected_branches = list(self.graph.edges(node))

                bifurcations.append({
                    'node_id': node,
                    'position': pos,
                    'degree': degree,
                    'num_branches': len(connected_branches),
                    'generation': generation,
                    'z': pos[0],
                    'y': pos[1],
                    'x': pos[2]
                })

        self.bifurcations_df = pd.DataFrame(bifurcations)

        print(f"Identified bifurcations: {len(self.bifurcations_df)}")
        if len(self.bifurcations_df) > 0:
            print(f"  Average degree: {self.bifurcations_df['degree'].mean():.2f}")
            print(f"  Max degree: {self.bifurcations_df['degree'].max()}")

            if 'generation' in self.bifurcations_df.columns:
                gen_counts = self.bifurcations_df['generation'].value_counts().sort_index()
                print(f"\nBifurcations per generation:")
                for gen, count in gen_counts.items():
                    if not np.isnan(gen):
                        print(f"  Generation {int(gen)}: {count} bifurcations")

        return self.bifurcations_df

    def visualize_graph_with_generations(self, save_path=None, max_generation=None, 
                                       show_node_labels=True, show_branch_labels=True,
                                       node_size=50, font_size=8, figsize=(16, 12)):
        """
        Visualizza il grafo 3D con etichette delle generazioni

        Args:
            save_path: Percorso per salvare l'immagine
            max_generation: Generazione massima da visualizzare
            show_node_labels: Mostra le etichette sui nodi
            show_branch_labels: Mostra le etichette sui rami
            node_size: Dimensione dei nodi
            font_size: Dimensione del font
        """
        print("\n=== 3D GRAPH WITH GENERATION LABELS ===")

        if self.graph is None:
            raise ValueError("Build graph first with build_graph()")

        if self.generation_assignments is None:
            raise ValueError("Assign generations first with assign_generations_weibel()")

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        if max_generation is not None:
            nodes_to_plot = [node for node in self.graph.nodes() 
                           if self.generation_assignments['nodes'].get(node, np.nan) <= max_generation]
        else:
            nodes_to_plot = list(self.graph.nodes())

        generations = [self.generation_assignments['nodes'].get(node, -1) for node in nodes_to_plot]
        unique_gens = sorted(set(generations))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_gens)))
        gen_to_color = {gen: colors[i] for i, gen in enumerate(unique_gens)}

        for node in nodes_to_plot:
            pos = self.graph.nodes[node]['pos']
            gen = self.generation_assignments['nodes'].get(node, -1)
            color = gen_to_color[gen]

            degree = self.graph.degree(node)
            size = node_size * (1 + degree * 0.2)

            ax.scatter(pos[2], pos[1], pos[0], 
                      c=[color], s=size, alpha=0.8, edgecolors='black', linewidths=0.5)

            if show_node_labels:
                ax.text(pos[2], pos[1], pos[0], f'{gen}', 
                       fontsize=font_size, ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))

        for edge in self.graph.edges():
            node1, node2 = edge
            if node1 in nodes_to_plot and node2 in nodes_to_plot:
                pos1 = self.graph.nodes[node1]['pos']
                pos2 = self.graph.nodes[node2]['pos']

                gen1 = self.generation_assignments['nodes'].get(node1, -1)
                gen2 = self.generation_assignments['nodes'].get(node2, -1)
                avg_gen = (gen1 + gen2) / 2

                closest_gen = min(unique_gens, key=lambda x: abs(x - avg_gen))
                color = gen_to_color[closest_gen]

                ax.plot([pos1[2], pos2[2]], [pos1[1], pos2[1]], [pos1[0], pos2[0]],
                       color=color, linewidth=2, alpha=0.7)

                if show_branch_labels:
                    mid_point = [(pos1[0] + pos2[0])/2, 
                                (pos1[1] + pos2[1])/2, 
                                (pos1[2] + pos2[2])/2]
                    branch_gen = self.generation_assignments['branches'].get(tuple(sorted(edge)), np.nan)
                    if not np.isnan(branch_gen):
                        ax.text(mid_point[2], mid_point[1], mid_point[0], f'G{branch_gen}',
                               fontsize=font_size-1, ha='center', va='center',
                               bbox=dict(boxstyle="round,pad=0.1", facecolor='yellow', alpha=0.7))

        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=gen_to_color[gen], markersize=8,
                                     label=f'Gen {gen}')
                          for gen in unique_gens]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.15, 1), 
                 loc='upper left', fontsize=9, title="Generations")

        ax.set_xlabel('X (voxel)')
        ax.set_ylabel('Y (voxel)')
        ax.set_zlabel('Z (voxel)')

        title = f'Bronchial Tree - Graph with Generation Labels\n{len(nodes_to_plot)} nodes, {len(unique_gens)} generations'
        if max_generation is not None:
            title += f' (up to Gen {max_generation})'
        ax.set_title(title)

        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Graph with generations saved: {save_path}")

        plt.show()

        return fig, ax

    def visualize_weibel_generations_enhanced(self, save_path=None, max_generation=None, 
                                            show_labels=True, label_frequency=0.3):
        """
        Visualizzazione migliorata delle generazioni di Weibel con etichette

        Args:
            save_path: Percorso per salvare l'immagine
            max_generation: Generazione massima da visualizzare
            show_labels: Mostra le etichette delle generazioni
            label_frequency: Frequenza delle etichette (0-1, più alto = più etichette)
        """
        print("\n=== ENHANCED WEIBEL GENERATION VISUALIZATION ===")

        if 'generation' not in self.branch_metrics_df.columns:
            raise ValueError("Assign generations first with assign_generations_weibel()")

        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')

        if max_generation is not None:
            plot_data = self.branch_metrics_df[self.branch_metrics_df['generation'] <= max_generation]
        else:
            plot_data = self.branch_metrics_df

        max_gen = int(plot_data['generation'].max())
        colors = plt.cm.tab20(np.linspace(0, 1, max_gen + 1))

        plotted_count = 0
        labeled_branches = set()

        for idx, row in plot_data.iterrows():
            if pd.isna(row['generation']):
                continue

            branch_id = int(row['branch_id'])
            generation = int(row['generation'])

            try:
                coords = self.skeleton_obj.path_coordinates(branch_id)

                if len(coords) > 1:
                    z_coords = coords[:, 0]
                    y_coords = coords[:, 1]
                    x_coords = coords[:, 2]

                    color = colors[generation % len(colors)]

                    line = ax.plot(x_coords, y_coords, z_coords, 
                                  color=color, linewidth=2, alpha=0.8,
                                  label=f'Gen {generation}' if generation not in labeled_branches else "")

                    if generation not in labeled_branches:
                        labeled_branches.add(generation)

                    plotted_count += 1

                    if show_labels and np.random.random() < label_frequency:
                        mid_idx = len(coords) // 2
                        if mid_idx < len(coords):
                            mid_point = coords[mid_idx]
                            ax.text(mid_point[2], mid_point[1], mid_point[0], 
                                   f'G{generation}', fontsize=8, ha='center', va='center',
                                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

            except Exception as e:
                continue

        print(f"Branches plotted: {plotted_count}")

        legend_elements = []
        for gen in sorted(labeled_branches):
            color = colors[gen % len(colors)]
            legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=2, 
                                            label=f'Generation {gen}'))

        ax.legend(handles=legend_elements, bbox_to_anchor=(1.15, 1), 
                 loc='upper left', fontsize=9, title="Weibel Generations")

        ax.set_xlabel('X (voxel)')
        ax.set_ylabel('Y (voxel)')
        ax.set_zlabel('Z (voxel)')

        title = f'Bronchial Tree - Weibel Generations\n{plotted_count} branches, {len(labeled_branches)} generations'
        if max_generation is not None:
            title += f' (up to Gen {max_generation})'
        ax.set_title(title)

        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Enhanced Weibel visualization saved: {save_path}")

        plt.show()

        return fig, ax

    def visualize_weibel_generations(self, save_path=None, max_generation=None):
        """
        Visualizes branches colored by Weibel generation
        """
        print("\n=== WEIBEL GENERATION VISUALIZATION ===")

        if 'generation' not in self.branch_metrics_df.columns:
            raise ValueError("Assign generations first with assign_generations_weibel()")

        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')

        if max_generation is not None:
            plot_data = self.branch_metrics_df[self.branch_metrics_df['generation'] <= max_generation]
        else:
            plot_data = self.branch_metrics_df

        max_gen = int(plot_data['generation'].max())
        colors = plt.cm.tab20(np.linspace(0, 1, max_gen + 1))

        plotted_count = 0
        for idx, row in plot_data.iterrows():
            if pd.isna(row['generation']):
                continue

            branch_id = int(row['branch_id'])
            generation = int(row['generation'])

            try:
                coords = self.skeleton_obj.path_coordinates(branch_id)

                if len(coords) > 1:
                    z_coords = coords[:, 0]
                    y_coords = coords[:, 1]
                    x_coords = coords[:, 2]

                    color = colors[generation % len(colors)]

                    ax.plot(x_coords, y_coords, z_coords, 
                           color=color, linewidth=2, alpha=0.8)
                    plotted_count += 1

            except Exception as e:
                continue

        print(f"Branches plotted: {plotted_count}")

        legend_elements = [plt.Line2D([0], [0], color=colors[i % len(colors)], 
                                     linewidth=2, label=f'Gen {i}')
                          for i in range(max_gen + 1)]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.15, 1), 
                 loc='upper left', fontsize=9)

        ax.set_xlabel('X (voxel)')
        ax.set_ylabel('Y (voxel)')
        ax.set_zlabel('Z (voxel)')
        ax.set_title(f'Bronchial Tree - Weibel Generations\n{plotted_count} branches, {max_gen+1} generations')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved: {save_path}")

        plt.show()

    def visualize_weibel_generations_with_carina(self, save_path=None, max_generation=None):
        """
        Visualizza le generazioni di Weibel con la carina evidenziata con PUNTINO
        """
        print("\n=== WEIBEL GENERATIONS WITH CARINA ===")

        if 'generation' not in self.branch_metrics_df.columns:
            raise ValueError("Assign generations first with assign_generations_weibel()")

        if not hasattr(self, 'carina_node'):
            self.identify_carina()

        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')

        if max_generation is not None:
            plot_data = self.branch_metrics_df[self.branch_metrics_df['generation'] <= max_generation]
        else:
            plot_data = self.branch_metrics_df

        max_gen = int(plot_data['generation'].max())
        colors = plt.cm.tab20(np.linspace(0, 1, max_gen + 1))

        plotted_count = 0
        for idx, row in plot_data.iterrows():
            if pd.isna(row['generation']):
                continue

            branch_id = int(row['branch_id'])
            generation = int(row['generation'])

            try:
                coords = self.skeleton_obj.path_coordinates(branch_id)

                if len(coords) > 1:
                    z_coords = coords[:, 0]
                    y_coords = coords[:, 1]
                    x_coords = coords[:, 2]

                    color = colors[generation % len(colors)]

                    ax.plot(x_coords, y_coords, z_coords, 
                        color=color, linewidth=2, alpha=0.8)
                    plotted_count += 1

            except Exception as e:
                continue

        carina_pos = self.graph.nodes[self.carina_node]['pos']
        ax.scatter(carina_pos[2], carina_pos[1], carina_pos[0],
                c='red', s=300, marker='o', edgecolors='black', linewidths=3,
                label=f'Carina (Ø={self.carina_info["avg_branch_diameter"]:.1f}mm)')

        coord_text = f'CARINA\nX:{carina_pos[2]:.1f}\nY:{carina_pos[1]:.1f}\nZ:{carina_pos[0]:.1f}'
        ax.text(carina_pos[2], carina_pos[1], carina_pos[0] + 15, 
            coord_text, fontsize=11, fontweight='bold', color='red', ha='center',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.9))

        print(f"Branches plotted: {plotted_count}")

        legend_elements = [plt.Line2D([0], [0], color=colors[i % len(colors)], 
                                    linewidth=2, label=f'Gen {i}')
                        for i in range(max_gen + 1)]
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='red', 
                                        markersize=10, label='Carina', linewidth=0))

        ax.legend(handles=legend_elements, bbox_to_anchor=(1.15, 1), 
                loc='upper left', fontsize=10)

        ax.set_xlabel('X (voxel)')
        ax.set_ylabel('Y (voxel)')
        ax.set_zlabel('Z (voxel)')
        ax.set_title(f'Weibel Generations with Carina\n{plotted_count} branches, Carina = Largest diameter bifurcation')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved: {save_path}")

        plt.show()
        return fig, ax

    def get_carina_coordinates(self):
        """
        Restituisce le coordinate della carina in diversi sistemi di riferimento
        """
        if not hasattr(self, 'carina_info'):
            self.identify_carina()

        carina_info = self.carina_info

        coordinates = {
            'node_id': carina_info['node_id'],
            'voxel_coordinates': {
                'z': carina_info['coordinates_voxel'][0],
                'y': carina_info['coordinates_voxel'][1], 
                'x': carina_info['coordinates_voxel'][2]
            },
            'world_coordinates': {
                'x': carina_info['coordinates_world'][0],
                'y': carina_info['coordinates_world'][1],
                'z': carina_info['coordinates_world'][2]
            },
            'physical_coordinates_mm': {
                'x': carina_info['coordinates_world'][0] * self.spacing[0],
                'y': carina_info['coordinates_world'][1] * self.spacing[1],
                'z': carina_info['coordinates_world'][2] * self.spacing[2]
            },
            'diameter_info': {
                'avg_branch_diameter_mm': carina_info['avg_branch_diameter'],
                'diameter_at_node_mm': carina_info['diameter_at_node'],
                'degree': carina_info['degree']
            }
        }

        print("\n=== CARINA COORDINATES ===")
        print(f"Node ID: {coordinates['node_id']}")
        print(f"Voxel coordinates (z,y,x): ({coordinates['voxel_coordinates']['z']}, {coordinates['voxel_coordinates']['y']}, {coordinates['voxel_coordinates']['x']})")
        print(f"World coordinates (x,y,z): ({coordinates['world_coordinates']['x']:.1f}, {coordinates['world_coordinates']['y']:.1f}, {coordinates['world_coordinates']['z']:.1f})")
        print(f"Physical coordinates (mm): ({coordinates['physical_coordinates_mm']['x']:.1f}, {coordinates['physical_coordinates_mm']['y']:.1f}, {coordinates['physical_coordinates_mm']['z']:.1f})")
        print(f"Diameter: {coordinates['diameter_info']['avg_branch_diameter_mm']:.2f} mm")
        print(f"Degree: {coordinates['diameter_info']['degree']}")

        return coordinates

    def visualize_diameter_analysis_with_carina(self, save_path=None):
        """
        Visualizza l'analisi dei diametri con la carina evidenziata
        """
        if not hasattr(self, 'branch_metrics_df'):
            raise ValueError("Calculate metrics first")

        if not hasattr(self, 'carina_node'):
            self.identify_carina()

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        color_values = self.branch_metrics_df['diameter_mean_mm']
        norm = Normalize(vmin=color_values.min(), vmax=color_values.max())
        cmap = plt.cm.viridis

        plotted_count = 0
        for idx, row in self.branch_metrics_df.iterrows():
            branch_id = int(row['branch_id'])
            diameter = row['diameter_mean_mm']

            try:
                coords = self.skeleton_obj.path_coordinates(branch_id)

                if len(coords) > 1:
                    z_coords = coords[:, 0]
                    y_coords = coords[:, 1]
                    x_coords = coords[:, 2]

                    color = cmap(norm(diameter))

                    ax.plot(x_coords, y_coords, z_coords, 
                        color=color, linewidth=3, alpha=0.8)
                    plotted_count += 1

            except Exception as e:
                continue

        carina_pos = self.graph.nodes[self.carina_node]['pos']
        ax.scatter(carina_pos[2], carina_pos[1], carina_pos[0],
                c='red', s=500, marker='*', edgecolors='black', linewidths=3,
                label=f'Carina (Ø={self.carina_info["avg_branch_diameter"]:.1f}mm)')

        ax.text(carina_pos[2], carina_pos[1], carina_pos[0] + 20, 
            f'CARINA\nØ={self.carina_info["avg_branch_diameter"]:.1f}mm', 
            fontsize=12, fontweight='bold', color='red', ha='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))

        print(f"Branches plotted: {plotted_count}")

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Mean Diameter (mm)', rotation=270, labelpad=20)

        ax.set_xlabel('X (voxel)')
        ax.set_ylabel('Y (voxel)')
        ax.set_zlabel('Z (voxel)')
        ax.set_title(f'Bronchial Tree - Diameter Analysis\nCarina highlighted in red (largest diameter)')
        ax.legend()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Diameter visualization with carina saved: {save_path}")

        plt.show()
        return fig, ax

    def plot_weibel_tapering_analysis(self, save_path=None):
        """
        Creates comprehensive visualization of diameter tapering across generations
        """
        if not hasattr(self, 'weibel_analysis_df'):
            raise ValueError("Run analyze_weibel_tapering() first")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        ax1 = axes[0, 0]
        generations = self.weibel_analysis_df['generation']
        diameters = self.weibel_analysis_df['diameter_mean_mm']
        errors = self.weibel_analysis_df['diameter_std_mm']

        ax1.errorbar(generations, diameters, yerr=errors, 
                    marker='o', markersize=8, capsize=5, linewidth=2)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Mean Diameter (mm)')
        ax1.set_title('Airway Diameter Tapering Across Generations\n(Weibel Model)')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(generations)

        if len(generations) > 1:
            gen_0_diameter = diameters.iloc[0]
            theoretical_diameters = [gen_0_diameter * (2**(-1/3))**g for g in generations]
            ax1.plot(generations, theoretical_diameters, 
                    'r--', linewidth=2, alpha=0.7, label='Weibel theoretical')
            ax1.legend()

        ax2 = axes[0, 1]
        if hasattr(self, 'tapering_ratios_df') and len(self.tapering_ratios_df) > 0:
            x_labels = [f"{row['from_generation']}→{row['to_generation']}" 
                       for _, row in self.tapering_ratios_df.iterrows()]
            ratios = self.tapering_ratios_df['diameter_ratio']

            ax2.bar(range(len(ratios)), ratios, alpha=0.7, edgecolor='black')
            ax2.axhline(y=2**(-1/3), color='r', linestyle='--', 
                       linewidth=2, label='Weibel theoretical (0.793)')
            ax2.set_xlabel('Generation Transition')
            ax2.set_ylabel('Diameter Ratio')
            ax2.set_title('Diameter Reduction Ratios Between Generations')
            ax2.set_xticks(range(len(ratios)))
            ax2.set_xticklabels(x_labels, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')

        ax3 = axes[1, 0]
        n_branches = self.weibel_analysis_df['n_branches']
        ax3.bar(generations, n_branches, alpha=0.7, edgecolor='black', color='green')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Number of Branches')
        ax3.set_title('Branch Count by Generation')
        ax3.set_xticks(generations)
        ax3.grid(True, alpha=0.3, axis='y')

        theoretical_branches = [2**(g+1) for g in generations]
        ax3.plot(generations, theoretical_branches, 
                'r--', linewidth=2, alpha=0.7, label='Perfect dichotomy (2^(n+1))')
        ax3.legend()

        ax4 = axes[1, 1]
        lengths = self.weibel_analysis_df['length_mean_mm']
        length_errors = self.weibel_analysis_df['length_std_mm']

        ax4.errorbar(generations, lengths, yerr=length_errors,
                    marker='s', markersize=8, capsize=5, 
                    linewidth=2, color='purple')
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Mean Branch Length (mm)')
        ax4.set_title('Branch Length Across Generations')
        ax4.set_xticks(generations)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Weibel analysis plot saved: {save_path}")

        plt.show()

    def visualize_distance_from_carina(self, save_path=None, colormap='viridis', max_branches=1000):
        """
        Visualizes branches colored by their PATH distance from the carina
        """
        print("\n=== DISTANCE FROM CARINA VISUALIZATION ===")

        if 'distance_from_carina_distal_mm' not in self.branch_metrics_df.columns:
            raise ValueError("Calculate distances from carina first")

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        color_values = self.branch_metrics_df['distance_from_carina_distal_mm'].dropna()

        norm = Normalize(vmin=color_values.min(), vmax=color_values.max())
        cmap = plt.colormaps[colormap]

        plotted_count = 0
        for idx, row in self.branch_metrics_df.iterrows():
            if plotted_count >= max_branches:
                break

            if pd.isna(row['distance_from_carina_distal_mm']):
                continue

            branch_id = int(row['branch_id'])
            distance = row['distance_from_carina_distal_mm']

            try:
                coords = self.skeleton_obj.path_coordinates(branch_id)

                if len(coords) > 1:
                    z_coords = coords[:, 0]
                    y_coords = coords[:, 1]
                    x_coords = coords[:, 2]

                    color = cmap(norm(distance))

                    ax.plot(x_coords, y_coords, z_coords, 
                           color=color, linewidth=2, alpha=0.8)
                    plotted_count += 1

            except Exception as e:
                continue

        print(f"Branches plotted: {plotted_count}")

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Path Distance from Carina (mm)', rotation=270, labelpad=20)

        ax.set_xlabel('X (voxel)')
        ax.set_ylabel('Y (voxel)')
        ax.set_zlabel('Z (voxel)')
        ax.set_title(f'Bronchial Tree - Path Distance from Carina\n{plotted_count} branches visualized')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved: {save_path}")

        plt.show()

    def plot_distance_distributions(self, save_path=None):
        """
        Plots distributions of path distances from carina
        """
        if 'distance_from_carina_distal_mm' not in self.branch_metrics_df.columns:
            raise ValueError("Calculate distances from carina first")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        valid_distances = self.branch_metrics_df['distance_from_carina_distal_mm'].dropna()
        axes[0, 0].hist(valid_distances, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 0].set_xlabel('Path Distance from Carina (mm)')
        axes[0, 0].set_ylabel('Number of Branches')
        axes[0, 0].set_title('Distribution of Branch Distances from Carina')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].scatter(self.branch_metrics_df['distance_from_carina_distal_mm'],
                          self.branch_metrics_df['diameter_mean_mm'],
                          alpha=0.5, s=20)
        axes[0, 1].set_xlabel('Path Distance from Carina (mm)')
        axes[0, 1].set_ylabel('Mean Diameter (mm)')
        axes[0, 1].set_title('Airway Diameter vs Path Distance from Carina')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].scatter(self.branch_metrics_df['distance_from_carina_distal_mm'],
                          self.branch_metrics_df['length_mm'],
                          alpha=0.5, s=20, color='green')
        axes[1, 0].set_xlabel('Path Distance from Carina (mm)')
        axes[1, 0].set_ylabel('Branch Length (mm)')
        axes[1, 0].set_title('Branch Length vs Path Distance from Carina')
        axes[1, 0].grid(True, alpha=0.3)

        if 'generation' in self.branch_metrics_df.columns:
            gen_data = self.branch_metrics_df[['generation', 'distance_from_carina_distal_mm']].dropna()
            axes[1, 1].scatter(gen_data['generation'],
                             gen_data['distance_from_carina_distal_mm'],
                             alpha=0.5, s=20, color='coral')
            axes[1, 1].set_xlabel('Generation')
            axes[1, 1].set_ylabel('Path Distance from Carina (mm)')
            axes[1, 1].set_title('Distance vs Generation')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Distance distributions saved: {save_path}")

        plt.show()

    def visualize_skeleton_3d(self, save_path=None, show_bifurcations=False):
        """Visualizes 3D skeleton"""
        print("\n=== 3D Visualization ===")

        if self.skeleton is None:
            raise ValueError("Compute skeleton first")

        skeleton_coords = np.argwhere(self.skeleton > 0)

        subsample = max(1, len(skeleton_coords) // 10000)
        skeleton_coords = skeleton_coords[::subsample]

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(skeleton_coords[:, 2], skeleton_coords[:, 1], skeleton_coords[:, 0],
                  c='blue', marker='.', s=1, alpha=0.6, label='Skeleton')

        if show_bifurcations and hasattr(self, 'bifurcations_df') and len(self.bifurcations_df) > 0:
            bif = self.bifurcations_df
            ax.scatter(bif['x'], bif['y'], bif['z'],
                      c='red', marker='o', s=100, alpha=0.8, label='Bifurcations')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Skeleton of Bronchial Tree')
        ax.legend()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved: {save_path}")

        plt.show()

    def visualize_branches_3d(self, save_path=None, colormap='viridis', max_branches=1000, color_by='diameter'):
        """Visualizes 3D branches colored by diameter or length"""
        print("\n=== 3D Branches Visualization ===")

        if not hasattr(self, 'branch_metrics_df'):
            raise ValueError("Calculate metrics first")

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        if color_by == 'length':
            color_values = self.branch_metrics_df['length_mm']
            color_label = 'Length (mm)'
        else:
            color_values = self.branch_metrics_df['diameter_mean_mm']
            color_label = 'Mean Diameter (mm)'

        norm = Normalize(vmin=color_values.min(), vmax=color_values.max())
        cmap = plt.colormaps[colormap]

        plotted_count = 0
        for idx, row in self.branch_metrics_df.iterrows():
            if plotted_count >= max_branches:
                break

            branch_id = int(row['branch_id'])
            color_value = row[color_label.split()[0].lower() + '_mm' if color_by == 'length' else 'diameter_mean_mm']

            try:
                coords = self.skeleton_obj.path_coordinates(branch_id)

                if len(coords) > 1:
                    z_coords = coords[:, 0]
                    y_coords = coords[:, 1]
                    x_coords = coords[:, 2]

                    color = cmap(norm(color_value))

                    ax.plot(x_coords, y_coords, z_coords, 
                           color=color, linewidth=2, alpha=0.8)
                    plotted_count += 1

            except Exception as e:
                continue

        print(f"Branches plotted: {plotted_count}")

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label(color_label, rotation=270, labelpad=20)

        ax.set_xlabel('X (voxel)')
        ax.set_ylabel('Y (voxel)')
        ax.set_zlabel('Z (voxel)')
        ax.set_title(f'Bronchial Tree Branches - Colored by {color_label}\n{plotted_count} branches')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved: {save_path}")

        plt.show()

    def visualize_connected_components(self, save_path=None, max_components=10, min_voxels=5):
        """Visualizes connected components"""
        print("\n=== CONNECTED COMPONENTS VISUALIZATION ===")

        if self.connected_components is None:
            raise ValueError("Analyze connected components first")

        large_components = self.component_stats[self.component_stats['voxel_count'] >= min_voxels]
        components_to_plot = min(max_components, len(large_components))

        print(f"Visualizing {components_to_plot} components with at least {min_voxels} voxels")

        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')

        colors = plt.cm.tab20(np.linspace(0, 1, components_to_plot))

        total_plotted = 0
        for i in range(components_to_plot):
            comp_id = large_components.iloc[i]['component_id']
            voxel_count = large_components.iloc[i]['voxel_count']

            component_coords = np.argwhere(self.connected_components == comp_id)

            if len(component_coords) > 0:
                if voxel_count < 100:
                    subsample = 1
                    marker_size = 20
                else:
                    subsample = max(1, len(component_coords) // 500)
                    marker_size = 10

                component_coords_subsampled = component_coords[::subsample]

                ax.scatter(component_coords_subsampled[:, 2], 
                          component_coords_subsampled[:, 1], 
                          component_coords_subsampled[:, 0],
                          c=[colors[i]], marker='o', s=marker_size, alpha=0.8,
                          label=f'Comp {comp_id} ({voxel_count} voxels)')

                total_plotted += len(component_coords_subsampled)

        ax.set_xlabel('X (voxel)')
        ax.set_ylabel('Y (voxel)')
        ax.set_zlabel('Z (voxel)')
        ax.set_title(f'Connected Components of Skeleton\n{components_to_plot} components, {total_plotted} points')
        ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Component visualization saved: {save_path}")

        plt.show()

    def plot_diameter_distribution(self, save_path=None):
        """Plots diameter distribution"""
        if not hasattr(self, 'branch_metrics_df'):
            print("Calculate metrics first")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(self.branch_metrics_df['diameter_mean_mm'], bins=30, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Mean diameter (mm)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Branch Diameters Distribution')
        axes[0].grid(True, alpha=0.3)

        axes[1].scatter(self.branch_metrics_df['length_mm'], 
                       self.branch_metrics_df['diameter_mean_mm'], 
                       alpha=0.6, s=20)
        axes[1].set_xlabel('Branch length (mm)')
        axes[1].set_ylabel('Mean diameter (mm)')
        axes[1].set_title('Diameter vs Branch Length')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved: {save_path}")

        plt.show()

    def plot_length_distribution(self, save_path=None):
        """Plots branch length distribution"""
        if not hasattr(self, 'branch_metrics_df'):
            print("Calculate metrics first")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(self.branch_metrics_df['length_mm'], bins=30, edgecolor='black', alpha=0.7, color='green')
        axes[0].set_xlabel('Branch length (mm)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Branch Length Distribution')
        axes[0].grid(True, alpha=0.3)

        axes[1].scatter(self.branch_metrics_df['length_mm'], 
                       self.branch_metrics_df['volume_mm3'], 
                       alpha=0.6, s=20, c='green')
        axes[1].set_xlabel('Branch length (mm)')
        axes[1].set_ylabel('Branch volume (mm³)')
        axes[1].set_title('Length vs Volume')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved: {save_path}")

        plt.show()

    def get_branch_summary(self):
        """Returns a dataframe with summary of all branches"""
        if not hasattr(self, 'branch_metrics_df'):
            raise ValueError("Calculate and merge metrics first")

        summary = self.branch_metrics_df.copy()
        print("\n=== BRANCH SUMMARY (first 10) ===")
        for i, (_, row) in enumerate(summary.iterrows()):
            if i >= 10:
                break
            gen_str = f", Gen {int(row['generation'])}" if 'generation' in row and not pd.isna(row['generation']) else ""
            print(f"Branch {row['branch_id']}: "
                  f"Length = {row['length_mm']:.2f} mm, "
                  f"Mean diameter = {row['diameter_mean_mm']:.2f} mm, "
                  f"Volume = {row['volume_mm3']:.2f} mm³{gen_str}")

        return summary

    def save_results(self, output_dir):
        """Saves all results to files"""
        os.makedirs(output_dir, exist_ok=True)

        skeleton_sitk = sitk.GetImageFromArray(self.skeleton.astype(np.uint8))
        skeleton_sitk.CopyInformation(self.sitk_image)
        skeleton_path = os.path.join(output_dir, "skeleton.nii.gz")
        sitk.WriteImage(skeleton_sitk, skeleton_path)
        print(f"Skeleton saved: {skeleton_path}")

        if self.connected_components is not None:
            components_sitk = sitk.GetImageFromArray(self.connected_components.astype(np.int16))
            components_sitk.CopyInformation(self.sitk_image)
            components_path = os.path.join(output_dir, "skeleton_components.nii.gz")
            sitk.WriteImage(components_sitk, components_path)
            print(f"Connected components saved: {components_path}")

        if hasattr(self, 'branch_metrics_df'):
            metrics_path = os.path.join(output_dir, "branch_metrics_complete.csv")
            self.branch_metrics_df.to_csv(metrics_path, index=False)
            print(f"Complete branch metrics saved: {metrics_path}")

        if hasattr(self, 'bifurcations_df'):
            bifurcations_path = os.path.join(output_dir, "bifurcations.csv")
            self.bifurcations_df.to_csv(bifurcations_path, index=False)
            print(f"Bifurcations saved: {bifurcations_path}")

        if hasattr(self, 'component_stats'):
            components_path = os.path.join(output_dir, "component_statistics.csv")
            self.component_stats.to_csv(components_path, index=False)
            print(f"Component statistics saved: {components_path}")

        if hasattr(self, 'weibel_analysis_df'):
            weibel_path = os.path.join(output_dir, "weibel_generation_analysis.csv")
            self.weibel_analysis_df.to_csv(weibel_path, index=False)
            print(f"Weibel generation analysis saved: {weibel_path}")

        if hasattr(self, 'tapering_ratios_df'):
            tapering_path = os.path.join(output_dir, "weibel_tapering_ratios.csv")
            self.tapering_ratios_df.to_csv(tapering_path, index=False)
            print(f"Tapering ratios saved: {tapering_path}")

        summary_path = os.path.join(output_dir, "analysis_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("=== BRONCHIAL TREE ANALYSIS ===\n\n")
            f.write(f"Input file: {self.mask_path}\n")
            f.write(f"Spacing (x,y,z): {self.spacing} mm\n")
            f.write(f"Shape (z,y,x): {self.mask.shape}\n\n")

            if self.skeleton is not None:
                f.write(f"Skeleton voxels: {np.sum(self.skeleton > 0)}\n\n")

            if hasattr(self, 'component_stats'):
                f.write("SKELETON CONNECTED COMPONENTS:\n")
                f.write(f"  Total components: {len(self.component_stats)}\n")
                f.write(f"  Largest component: {self.component_stats['voxel_count'].iloc[0]} voxels\n\n")

            if hasattr(self, 'branch_metrics_df'):
                f.write("BRANCH METRICS:\n")
                f.write(f"  Number of branches: {len(self.branch_metrics_df)}\n")
                f.write(f"  Mean diameter: {self.branch_metrics_df['diameter_mean_mm'].mean():.2f} mm\n")
                f.write(f"  Min diameter: {self.branch_metrics_df['diameter_mean_mm'].min():.2f} mm\n")
                f.write(f"  Max diameter: {self.branch_metrics_df['diameter_mean_mm'].max():.2f} mm\n")
                f.write(f"  Mean length: {self.branch_metrics_df['length_mm'].mean():.2f} mm\n")
                f.write(f"  Total airway length: {self.branch_metrics_df['length_mm'].sum():.2f} mm\n")
                f.write(f"  Total airway volume: {self.branch_metrics_df['volume_mm3'].sum():.2f} mm³\n\n")

            if hasattr(self, 'bifurcations_df'):
                f.write("BIFURCATIONS:\n")
                f.write(f"  Number: {len(self.bifurcations_df)}\n")
                if len(self.bifurcations_df) > 0:
                    f.write(f"  Average degree: {self.bifurcations_df['degree'].mean():.2f}\n\n")

            if hasattr(self, 'weibel_analysis_df'):
                f.write("WEIBEL GENERATION ANALYSIS:\n")
                f.write(f"  Maximum generation: {int(self.weibel_analysis_df['generation'].max())}\n")
                f.write(f"  Number of generations: {len(self.weibel_analysis_df)}\n")
                if hasattr(self, 'tapering_ratios_df') and len(self.tapering_ratios_df) > 0:
                    mean_ratio = self.tapering_ratios_df['diameter_ratio'].mean()
                    f.write(f"  Mean tapering ratio: {mean_ratio:.3f}\n")
                    f.write(f"  Weibel theoretical ratio: {2**(-1/3):.3f}\n")

        print(f"Summary saved: {summary_path}")

    def run_full_analysis(self, output_dir, visualize=True, 
                         max_reconnect_distance_mm=15.0, 
                         min_voxels_for_reconnect=5,
                         max_voxels_for_keep=100,
                         original_mask_path=None):
        """
        Runs complete analysis pipeline with Weibel generation analysis

        Args:
            output_dir: Directory for saving results
            visualize: Whether to generate visualizations
            max_reconnect_distance_mm: Max distance for reconnecting components
            min_voxels_for_reconnect: Min voxels needed to attempt reconnection
            max_voxels_for_keep: Max voxels to keep isolated component
            original_mask_path: Optional path to original mask for accurate metrics
                               (dual-mask strategy: refined for skeleton, original for metrics)
        """
        print("\n" + "="*60)
        print("COMPLETE BRONCHIAL TREE ANALYSIS PIPELINE")
        print("WITH WEIBEL GENERATION ANALYSIS")
        if original_mask_path:
            print("DUAL-MASK STRATEGY: Refined for skeleton, Original for metrics")
        print("="*60)

        if original_mask_path and self.original_mask is None:
            print(f"\nLoading ORIGINAL mask for accurate metrics: {original_mask_path}")
            original_sitk = sitk.ReadImage(original_mask_path)
            self.original_mask = sitk.GetArrayFromImage(original_sitk)
            print(f"Original mask positive voxels: {np.sum(self.original_mask > 0)}")

        self.compute_skeleton()

        self.analyze_connected_components()

        self.smart_component_management(
            max_reconnect_distance_mm=max_reconnect_distance_mm,
            min_voxels_for_reconnect=min_voxels_for_reconnect,
            max_voxels_for_keep=max_voxels_for_keep,
            remove_tiny_components=True
        )

        self.analyze_connected_components()

        self.build_graph()

        self.identify_carina()

        self.assign_generations_weibel()

        self.calculate_branch_lengths()

        if self.original_mask is not None and self.original_distance_transform is None:
            print("\nComputing distance transform on ORIGINAL mask for accurate diameter metrics...")
            original_binary = (self.original_mask > 0).astype(np.uint8)
            self.original_distance_transform = distance_transform_edt(original_binary, sampling=self.spacing)

        self.analyze_diameters()

        self.merge_branch_metrics()

        self.analyze_weibel_tapering()

        self.calculate_distances_from_carina()

        self.identify_bifurcations()

        self.save_results(output_dir)

        if visualize:
            self.visualize_skeleton_3d(
                save_path=os.path.join(output_dir, "skeleton_3d.png"),
                show_bifurcations=False
            )

            self.visualize_connected_components(
                save_path=os.path.join(output_dir, "connected_components.png")
            )

            self.visualize_branches_3d(
                save_path=os.path.join(output_dir, "branches_3d_diameter.png"),
                color_by='diameter'
            )

            self.visualize_branches_3d(
                save_path=os.path.join(output_dir, "branches_3d_length.png"),
                color_by='length'
            )

            self.visualize_weibel_generations(
                save_path=os.path.join(output_dir, "weibel_generations.png")
            )

            self.plot_weibel_tapering_analysis(
                save_path=os.path.join(output_dir, "weibel_tapering_analysis.png")
            )

            self.visualize_distance_from_carina(
                save_path=os.path.join(output_dir, "distance_from_carina.png")
            )

            self.plot_distance_distributions(
                save_path=os.path.join(output_dir, "distance_distributions.png")
            )

            self.plot_diameter_distribution(
                save_path=os.path.join(output_dir, "diameter_distribution.png")
            )

            self.plot_length_distribution(
                save_path=os.path.join(output_dir, "length_distribution.png")
            )

            self.visualize_weibel_generations_with_carina(
                save_path=os.path.join(output_dir, "weibel_generations_with_carina.png")
            )

        self.get_branch_summary()

        print("\n" + "="*60)
        print("ANALYSIS COMPLETED!")
        print("="*60)

        return {
            'skeleton': self.skeleton,
            'connected_components': self.connected_components,
            'component_stats': self.component_stats,
            'graph': self.graph,
            'branch_metrics': self.branch_metrics_df,
            'bifurcations': self.bifurcations_df,
            'weibel_analysis': self.weibel_analysis_df,
            'tapering_ratios': self.tapering_ratios_df
        }