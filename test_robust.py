import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label, center_of_mass, binary_erosion, binary_dilation
from scipy.ndimage import distance_transform_edt, generate_binary_structure
from skimage.morphology import skeletonize, ball
from skan import Skeleton, summarize
import networkx as nx
from collections import defaultdict
from airway_graph import AirwayGraphAnalyzer
import json


class EnhancedCarinaDetector:
    """
    Sistema ultra-conservativo per detection della carina con:
    - Pre-cut fisso a z=390
    - Identificazione intelligente della trachea vera
    - Preservazione completa di tutti i rami bronchiali
    """
    
    def __init__(self, mask, spacing, verbose=True, precut_z=390, trachea_remove_fraction=0.15):
        """
        Args:
            mask: Maschera 3D delle vie aeree (numpy array)
            spacing: Tuple (x,y,z) spacing in mm
            verbose: Print debug information
            precut_z: Z-level per pre-cut iniziale (default 390)
            trachea_remove_fraction: Fraction of the identified trachea length to remove
                starting from the superior end (default 0.15 = remove only top 15% - VERY CONSERVATIVE!).
        """
        self.mask = mask
        self.spacing = spacing
        self.verbose = verbose
        self.precut_z = precut_z
        # Fraction (0..1) of trachea length to remove from the superior/top end
        # Smaller values preserve more trachea near the carina.
        self.trachea_remove_fraction = float(trachea_remove_fraction)
        
        # Risultati
        self.precut_mask = None
        self.cleaned_mask = None
        self.carina_z = None
        self.carina_y = None
        self.carina_x = None
        self.confidence_score = 0.0
        self.detection_method = None
        self.trachea_top_z = None
        self.trachea_bottom_z = None
        self.trachea_length = None
        self.trachea_mask = None
        
    def detect_carina_robust(self):
        """
        Detection robusta con pre-cut e identificazione trachea
        """
        if self.verbose:
            print("\n" + "="*70)
            print(" "*20 + "ENHANCED CARINA DETECTION")
            print("="*70)
        
        # STEP 0: Pre-cut fisso
        self._apply_precut()
        
        # STEP 1: Identificazione conservativa della trachea
        self._identify_trachea_conservative()
        
        # STEP 2: Pulizia intelligente basata su trachea identificata
        self._intelligent_trachea_cleaning()
        
        # STEP 3: Detection multi-metodo
        candidates = []
        
        carina_cc = self._detect_by_connected_components()
        if carina_cc is not None:
            candidates.append(('connected_components', carina_cc, 3.0))
        
        carina_diameter = self._detect_by_diameter_analysis()
        if carina_diameter is not None:
            candidates.append(('diameter', carina_diameter, 2.0))
        
        carina_topo = self._detect_by_topology()
        if carina_topo is not None:
            candidates.append(('topology', carina_topo, 2.5))
        
        carina_slice = self._detect_by_slice_analysis()
        if carina_slice is not None:
            candidates.append(('slice_analysis', carina_slice, 1.5))
        
        # STEP 4: Voting e selezione
        if not candidates:
            if self.verbose:
                print("\n⚠ WARNING: No candidates found, using fallback")
            return self._fallback_detection()
        
        best_candidate = self._vote_and_select(candidates)
        
        self.carina_z, self.carina_y, self.carina_x = best_candidate['position']
        self.confidence_score = best_candidate['confidence']
        self.detection_method = best_candidate['method']
        
        if self.verbose:
            print(f"\n✓ CARINA DETECTED:")
            print(f"  Position: z={self.carina_z}, y={self.carina_y}, x={self.carina_x}")
            print(f"  Method: {self.detection_method}")
            print(f"  Confidence: {self.confidence_score:.2f}/5.0")
        
        return self.carina_z, self.carina_y, self.carina_x, self.confidence_score
    
    def _apply_precut(self):
        """
        Applica pre-cut fisso a z=precut_z
        """
        if self.verbose:
            print(f"\nStep 0: Applying fixed pre-cut at z={self.precut_z}...")
        
        self.precut_mask = self.mask.copy()
        
        # Applica cut solo se necessario
        if self.precut_z < self.mask.shape[0]:
            original_voxels = np.sum(self.mask > 0)
            self.precut_mask[self.precut_z:, :, :] = 0
            removed_voxels = original_voxels - np.sum(self.precut_mask > 0)
            
            if self.verbose:
                print(f"  Cut applied at z={self.precut_z}")
                print(f"  Removed: {removed_voxels:,} voxels ({removed_voxels/original_voxels*100:.1f}%)")
        else:
            if self.verbose:
                print(f"  No cut needed (precut_z >= volume height)")
    
    def _identify_trachea_conservative(self):
        """
        Identifica la trachea in modo ultra-conservativo:
        - Cerca la componente tubolare centrale
        - Distingue trachea da rami bronchiali
        - Non tocca strutture laterali
        """
        if self.verbose:
            print("\nStep 1: Conservative trachea identification...")
        
        binary_mask = (self.precut_mask > 0).astype(np.uint8)
        
        # Parametri anatomici (basati su conoscenze mediche)
        center_y = binary_mask.shape[1] // 2
        center_x = binary_mask.shape[2] // 2
        max_trachea_deviation = min(binary_mask.shape[1], binary_mask.shape[2]) * 0.20  # 20% dal centro
        
        # Analisi slice-by-slice dall'alto verso il basso
        trachea_slices = []
        trachea_components = []
        
        z_max = binary_mask.shape[0] - 1
        z_min = max(0, int(binary_mask.shape[0] * 0.3))  # Cerca fino al 30% inferiore
        
        for z in range(z_max, z_min, -1):
            slice_2d = binary_mask[z, :, :]
            
            if np.sum(slice_2d) < 10:
                continue
            
            labeled, num_objects = label(slice_2d)
            
            # Cerca la componente più centrale
            best_component = None
            min_deviation = float('inf')
            
            for obj_id in range(1, num_objects + 1):
                obj_mask = (labeled == obj_id)
                obj_area = np.sum(obj_mask)
                
                # Filtro dimensionale: la trachea non è troppo piccola né troppo grande
                if obj_area < 30 or obj_area > 3000:
                    continue
                
                # Centro di massa della componente
                cy, cx = center_of_mass(obj_mask)
                deviation = np.sqrt((cy - center_y)**2 + (cx - center_x)**2)
                
                # La trachea deve essere vicina al centro anatomico
                if deviation < max_trachea_deviation and deviation < min_deviation:
                    # Verifica forma: la trachea è ragionevolmente compatta
                    coords = np.argwhere(obj_mask)
                    y_extent = np.max(coords[:, 0]) - np.min(coords[:, 0]) + 1
                    x_extent = np.max(coords[:, 1]) - np.min(coords[:, 1]) + 1
                    bbox_area = y_extent * x_extent
                    compactness = obj_area / bbox_area if bbox_area > 0 else 0
                    
                    # Trachea abbastanza compatta (non troppo ramificata)
                    if compactness > 0.3:
                        min_deviation = deviation
                        best_component = {
                            'z': z,
                            'label': obj_id,
                            'area': obj_area,
                            'centroid': (cy, cx),
                            'deviation': deviation,
                            'compactness': compactness
                        }
            
            if best_component is not None:
                trachea_slices.append(z)
                trachea_components.append(best_component)
        
        if len(trachea_slices) < 5:
            if self.verbose:
                print("  ⚠ Cannot identify clear trachea, will be very conservative")
            self.trachea_mask = None
            return
        
        # Analizza continuità della trachea
        areas = [comp['area'] for comp in trachea_components]
        centroids = [comp['centroid'] for comp in trachea_components]
        
        # Cerca dove la trachea inizia a biforcare (area aumenta bruscamente)
        window_size = 5
        if len(areas) > window_size:
            smoothed_areas = np.convolve(areas, np.ones(window_size)/window_size, mode='valid')
            derivatives = np.diff(smoothed_areas)
            
            # Trova primo grande aumento di area (probabile biforcazione)
            mean_deriv = np.mean(np.abs(derivatives))
            std_deriv = np.std(derivatives)
            threshold = mean_deriv + 2 * std_deriv
            
            bifurcation_idx = None
            for i, deriv in enumerate(derivatives):
                if deriv > threshold:
                    bifurcation_idx = i
                    break
            
            if bifurcation_idx is not None:
                # La trachea finisce poco prima della biforcazione
                self.trachea_bottom_z = trachea_slices[min(bifurcation_idx + window_size, len(trachea_slices) - 1)]
            else:
                # Usa una percentuale conservativa
                self.trachea_bottom_z = trachea_slices[len(trachea_slices) // 2]
        else:
            self.trachea_bottom_z = trachea_slices[len(trachea_slices) // 2]
        
        self.trachea_top_z = trachea_slices[0]
        self.trachea_length = self.trachea_top_z - self.trachea_bottom_z
        
        # Crea maschera della trachea identificata
        self.trachea_mask = np.zeros_like(binary_mask)
        for z in range(self.trachea_bottom_z, self.trachea_top_z + 1):
            if z in trachea_slices:
                idx = trachea_slices.index(z)
                comp = trachea_components[idx]
                slice_2d = binary_mask[z, :, :]
                labeled, _ = label(slice_2d)
                
                # Trova componente corrispondente per centroid matching
                min_dist = float('inf')
                best_label = 1
                for obj_id in range(1, np.max(labeled) + 1):
                    obj_mask = (labeled == obj_id)
                    cy, cx = center_of_mass(obj_mask)
                    dist = np.sqrt((cy - comp['centroid'][0])**2 + (cx - comp['centroid'][1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_label = obj_id
                
                self.trachea_mask[z, :, :] = (labeled == best_label)
        
        if self.verbose:
            print(f"  ✓ Trachea identified:")
            print(f"    Top: z={self.trachea_top_z}")
            print(f"    Bottom: z={self.trachea_bottom_z}")
            print(f"    Length: {self.trachea_length} slices ({self.trachea_length * self.spacing[2]:.1f} mm)")
            print(f"    Voxels: {np.sum(self.trachea_mask):,}")
    
    def _intelligent_trachea_cleaning(self):
        """
        Pulizia intelligente: rimuove solo trachea vera, preserva tutto il resto
        """
        if self.verbose:
            print("\nStep 2: Intelligent trachea cleaning...")
        
        if self.trachea_mask is None:
            # Fallback ultra-conservativo: rimuovi solo top 5%
            if self.verbose:
                print("  Using ultra-conservative fallback (no trachea identified)")
            
            binary_mask = (self.precut_mask > 0).astype(np.uint8)
            coords = np.argwhere(binary_mask)
            if len(coords) > 0:
                z_max = np.max(coords[:, 0])
                z_min = np.min(coords[:, 0])
                height = z_max - z_min
                remove_slices = max(3, int(height * 0.05))
                cutoff = z_max - remove_slices
                
                self.cleaned_mask = self.precut_mask.copy()
                self.cleaned_mask[cutoff:, :, :] = 0
                
                if self.verbose:
                    print(f"  Removed top {remove_slices} slices (z >= {cutoff})")
            else:
                self.cleaned_mask = self.precut_mask.copy()
            return
        
        # Rimuovi solo la trachea identificata
        self.cleaned_mask = self.precut_mask.copy()
        
        # Rimuovi solo la porzione superiore della trachea: usare una frazione configurabile
        # MA con un margine di sicurezza di almeno 15mm sopra la carina
        safety_margin_mm = 15.0
        safety_margin_slices = int(safety_margin_mm / self.spacing[2])
        
        remove_slices = max(1, int(self.trachea_length * self.trachea_remove_fraction))
        # Calcola indice di inizio rimozione (inclusive)
        # IMPORTANTE: non scendere sotto trachea_bottom + safety_margin
        removal_start_z = max(self.trachea_bottom_z + safety_margin_slices, 
                             self.trachea_top_z - remove_slices + 1)

        for z in range(removal_start_z, self.trachea_top_z + 1):
            # Rimuovi solo la componente tracheale, non tutto lo slice
            trachea_slice = self.trachea_mask[z, :, :]
            self.cleaned_mask[z, :, :] = self.cleaned_mask[z, :, :] * (1 - trachea_slice)

        removed_voxels = np.sum(self.precut_mask > 0) - np.sum(self.cleaned_mask > 0)

        if self.verbose:
            print(f"  ✓ Intelligent cleaning complete:")
            print(f"  Trachea removal fraction: {self.trachea_remove_fraction:.2f}")
            print(f"  Removal range: z={removal_start_z} to z={self.trachea_top_z} ({remove_slices} slices)")
            print(f"  Removed: {removed_voxels:,} voxels (trachea only)")
            print(f"  Preserved: all bronchial branches")
    
    def _detect_by_connected_components(self):
        """Detection basata su componenti connesse"""
        if self.verbose:
            print("\nMethod 1: Connected components analysis...")
        
        binary_mask = (self.cleaned_mask > 0).astype(np.uint8)
        candidates = []
        
        # Cerca nell'area dove ci aspettiamo la carina
        z_start = min(binary_mask.shape[0] - 1, self.trachea_bottom_z if self.trachea_bottom_z else binary_mask.shape[0] - 1)
        z_min = max(0, int(binary_mask.shape[0] * 0.2))
        
        for z in range(z_start, z_min, -1):
            slice_2d = binary_mask[z, :, :]
            if np.sum(slice_2d) < 50:
                continue
            
            labeled, num_objects = label(slice_2d)
            
            if num_objects >= 2:
                sizes = []
                centroids = []
                for obj_id in range(1, num_objects + 1):
                    obj_mask = (labeled == obj_id)
                    size = np.sum(obj_mask)
                    sizes.append(size)
                    centroids.append(center_of_mass(obj_mask))
                
                sorted_indices = np.argsort(sizes)[::-1]
                
                if len(sizes) >= 2:
                    size1, size2 = sizes[sorted_indices[0]], sizes[sorted_indices[1]]
                    size_ratio = size2 / size1 if size1 > 0 else 0
                    
                    # Deve essere una vera biforcazione (componenti simili)
                    if 0.2 < size_ratio < 0.8:
                        score = size_ratio * (2.0 if num_objects == 2 else 1.5)
                        
                        cent1 = centroids[sorted_indices[0]]
                        cent2 = centroids[sorted_indices[1]]
                        carina_y = int((cent1[0] + cent2[0]) / 2)
                        carina_x = int((cent1[1] + cent2[1]) / 2)
                        
                        candidates.append({
                            'z': z,
                            'y': carina_y,
                            'x': carina_x,
                            'score': score,
                            'num_objects': num_objects,
                            'size_ratio': size_ratio
                        })
        
        if not candidates:
            if self.verbose:
                print("  No valid candidates found")
            return None
        
        # Prendi il più alto (più vicino alla carina anatomica)
        best = max(candidates, key=lambda x: x['z'])
        
        if self.verbose:
            print(f"  Best: z={best['z']}, score={best['score']:.3f}")
        
        return (best['z'], best['y'], best['x'])
    
    def _detect_by_diameter_analysis(self):
        """Detection basata su analisi diametro"""
        if self.verbose:
            print("\nMethod 2: Diameter analysis...")
        
        try:
            binary_mask = (self.cleaned_mask > 0).astype(np.uint8)
            skeleton = skeletonize(binary_mask)
            if np.sum(skeleton) < 10:
                return None
            
            spacing_zyx = (self.spacing[2], self.spacing[1], self.spacing[0])
            distance_transform = distance_transform_edt(binary_mask, sampling=spacing_zyx)
            
            skeleton_obj = Skeleton(skeleton, spacing=spacing_zyx)
            coordinates = skeleton_obj.coordinates
            
            if len(coordinates) == 0:
                return None
            
            candidates = []
            z_threshold = binary_mask.shape[0] * 0.3
            
            for idx in range(len(coordinates)):
                pos = coordinates[idx]
                z, y, x = int(pos[0]), int(pos[1]), int(pos[2])
                
                if z < z_threshold:
                    continue
                
                if not (0 <= z < distance_transform.shape[0] and
                       0 <= y < distance_transform.shape[1] and
                       0 <= x < distance_transform.shape[2]):
                    continue
                
                diameter = distance_transform[z, y, x] * 2
                
                candidates.append({
                    'z': z,
                    'y': y,
                    'x': x,
                    'diameter': diameter,
                    'score': diameter
                })
            
            if not candidates:
                return None
            
            candidates.sort(key=lambda x: x['score'], reverse=True)
            top_candidates = candidates[:max(5, len(candidates) // 10)]
            best = max(top_candidates, key=lambda x: x['z'])
            
            if self.verbose:
                print(f"  Best: z={best['z']}, diameter={best['diameter']:.2f}mm")
            
            return (best['z'], best['y'], best['x'])
            
        except Exception as e:
            if self.verbose:
                print(f"  Error: {e}")
            return None
    
    def _detect_by_topology(self):
        """Detection basata su topologia"""
        if self.verbose:
            print("\nMethod 3: Topology analysis...")
        
        try:
            binary_mask = (self.cleaned_mask > 0).astype(np.uint8)
            skeleton = skeletonize(binary_mask)
            if np.sum(skeleton) < 10:
                return None
            
            spacing_zyx = (self.spacing[2], self.spacing[1], self.spacing[0])
            skeleton_obj = Skeleton(skeleton, spacing=spacing_zyx)
            branch_data = summarize(skeleton_obj)
            
            G = nx.Graph()
            coordinates = skeleton_obj.coordinates
            
            for idx in range(len(coordinates)):
                pos = coordinates[idx]
                G.add_node(idx, pos=pos)
            
            for _, row in branch_data.iterrows():
                node1 = int(row['node-id-src'])
                node2 = int(row['node-id-dst'])
                length = row['branch-distance']
                G.add_edge(node1, node2, length=length)
            
            if len(G.nodes()) < 3:
                return None
            
            try:
                centrality = nx.betweenness_centrality(G, weight='length')
            except:
                return None
            
            z_threshold = binary_mask.shape[0] * 0.3
            bifurcation_nodes = [n for n in G.nodes() if G.degree(n) >= 3 and G.nodes[n]['pos'][0] >= z_threshold]
            
            if not bifurcation_nodes:
                bifurcation_nodes = [n for n in G.nodes() if G.degree(n) >= 2 and G.nodes[n]['pos'][0] >= z_threshold]
            
            if not bifurcation_nodes:
                return None
            
            bifurcation_nodes.sort(key=lambda n: G.nodes[n]['pos'][0], reverse=True)
            candidates = bifurcation_nodes[:5]
            
            best_node = max(candidates, key=lambda n: centrality.get(n, 0))
            best_pos = G.nodes[best_node]['pos']
            
            z, y, x = int(best_pos[0]), int(best_pos[1]), int(best_pos[2])
            
            if self.verbose:
                print(f"  Best: z={z}, centrality={centrality[best_node]:.3f}")
            
            return (z, y, x)
            
        except Exception as e:
            if self.verbose:
                print(f"  Error: {e}")
            return None
    
    def _detect_by_slice_analysis(self):
        """Analisi slice-by-slice"""
        if self.verbose:
            print("\nMethod 4: Slice analysis...")
        
        binary_mask = (self.cleaned_mask > 0).astype(np.uint8)
        candidates = []
        
        z_start = binary_mask.shape[0] - 1
        z_min = max(0, int(binary_mask.shape[0] * 0.3))
        
        for z in range(z_start, z_min, -1):
            slice_2d = binary_mask[z, :, :]
            area = np.sum(slice_2d)
            
            if area < 30:
                continue
            
            labeled, num_objects = label(slice_2d)
            
            if area > 0:
                coords = np.argwhere(slice_2d)
                y_extent = np.max(coords[:, 0]) - np.min(coords[:, 0])
                x_extent = np.max(coords[:, 1]) - np.min(coords[:, 1])
                extent = (y_extent + 1) * (x_extent + 1)
                compactness = area / extent if extent > 0 else 0
            else:
                compactness = 0
            
            score = 0
            if num_objects == 2:
                score = 2.5 * (1.0 - compactness)
            elif num_objects > 2:
                score = 1.5 * (1.0 - compactness)
            
            if score > 0.4:
                cy, cx = center_of_mass(slice_2d)
                candidates.append({
                    'z': z,
                    'y': int(cy),
                    'x': int(cx),
                    'score': score,
                    'num_objects': num_objects
                })
        
        if not candidates:
            return None
        
        best = max(candidates, key=lambda x: x['z'])
        
        if self.verbose:
            print(f"  Best: z={best['z']}, score={best['score']:.3f}")
        
        return (best['z'], best['y'], best['x'])
    
    def _vote_and_select(self, candidates):
        """Voting tra candidati"""
        if self.verbose:
            print(f"\nVoting among {len(candidates)} candidates...")
        
        best_idx = None
        best_consensus_score = -1
        
        for i, (method, pos, weight) in enumerate(candidates):
            consensus = 0
            for j, (_, other_pos, other_weight) in enumerate(candidates):
                if i == j:
                    continue
                
                dist = np.linalg.norm(np.array(pos) - np.array(other_pos))
                
                if dist < 25:
                    consensus += other_weight * (1.0 - dist / 25)
            
            total_score = weight + consensus
            
            if self.verbose:
                print(f"  {method}: pos={pos}, weight={weight:.1f}, "
                      f"consensus={consensus:.2f}, total={total_score:.2f}")
            
            if total_score > best_consensus_score:
                best_consensus_score = total_score
                best_idx = i
        
        best_method, best_pos, best_weight = candidates[best_idx]
        confidence = min(5.0, best_consensus_score / len(candidates))
        
        return {
            'method': best_method,
            'position': best_pos,
            'confidence': confidence
        }
    
    def _fallback_detection(self):
        """Fallback conservativo"""
        if self.verbose:
            print("\nUsing fallback detection...")
        
        binary_mask = (self.cleaned_mask > 0).astype(np.uint8)
        coords = np.argwhere(binary_mask)
        
        if len(coords) == 0:
            self.carina_z = binary_mask.shape[0] // 2
            self.carina_y = binary_mask.shape[1] // 2
            self.carina_x = binary_mask.shape[2] // 2
            self.confidence_score = 0.5
            self.detection_method = 'fallback_center'
            return self.carina_z, self.carina_y, self.carina_x, 0.5
        
        min_z, max_z = np.min(coords[:, 0]), np.max(coords[:, 0])
        self.carina_z = int(min_z + (max_z - min_z) * 0.6)  # 60% dall'alto
        
        self.carina_y = int(np.mean(coords[:, 1]))
        self.carina_x = int(np.mean(coords[:, 2]))
        
        self.confidence_score = 1.5
        self.detection_method = 'fallback_anatomical'
        
        return self.carina_z, self.carina_y, self.carina_x, 1.5
    
    def get_bronchi_mask(self):
        """
        Ritorna maschera finale dei bronchi con carina preservata
        """
        if self.carina_z is None:
            raise ValueError("Run detect_carina_robust() first")
        
        # La maschera pulita già contiene bronchi + carina
        # Applichiamo solo smoothing morfologico leggero
        bronchi_mask = self.cleaned_mask.copy()
        
        struct = generate_binary_structure(3, 1)
        bronchi_mask_binary = (bronchi_mask > 0).astype(np.uint8)
        bronchi_mask_binary = binary_dilation(bronchi_mask_binary, structure=struct, iterations=1)
        bronchi_mask_binary = binary_erosion(bronchi_mask_binary, structure=struct, iterations=1)
        
        bronchi_mask = bronchi_mask_binary.astype(self.mask.dtype) * np.max(self.mask)
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("FINAL BRONCHI MASK:")
            print(f"{'='*70}")
            print(f"  Voxels: {np.sum(bronchi_mask > 0):,}")
            print(f"  Carina preserved at: z={self.carina_z}")
            if self.trachea_bottom_z is not None and self.trachea_top_z is not None and self.trachea_length is not None:
                removal_start = max(self.trachea_bottom_z, self.trachea_top_z - max(1, int(self.trachea_length * self.trachea_remove_fraction)) + 1)
                print(f"  Trachea removed from: z={removal_start} to z={self.trachea_top_z}")
        
        return bronchi_mask


def integrate_with_pipeline(mask_path, spacing=None, precut_z=390, save_output=True, output_dir=None):
    """
    Integrazione con pipeline
    
    Args:
        mask_path: Path to airway mask
        spacing: Voxel spacing
        precut_z: Z-level for precut
        save_output: Whether to save outputs
        output_dir: Directory where to save outputs (default: current directory)
    """
    import os
    
    sitk_image = sitk.ReadImage(mask_path)
    mask = sitk.GetArrayFromImage(sitk_image)
    
    if spacing is None:
        spacing = sitk_image.GetSpacing()
    
    detector = EnhancedCarinaDetector(mask, spacing, verbose=True, precut_z=precut_z)
    carina_z, carina_y, carina_x, confidence = detector.detect_carina_robust()
    bronchi_mask = detector.get_bronchi_mask()
    
    if save_output:
        # Use output_dir if provided, otherwise current directory
        if output_dir is None:
            output_dir = "."
        os.makedirs(output_dir, exist_ok=True)
        
        output_sitk = sitk.GetImageFromArray(bronchi_mask.astype(np.uint8))
        output_sitk.CopyInformation(sitk_image)
        output_path = os.path.join(output_dir, "bronchi_enhanced_conservative.nii.gz")
        sitk.WriteImage(output_sitk, output_path)
        print(f"\n✓ Saved: {output_path}")
        # Generate graph visualization with carina highlighted (red dot)
        try:
            print("\nGenerating airway graph visualization (this may take a moment)...")
            # Pass the carina coordinates detected above so the graph analyzer
            # uses the same reference point (voxel coordinates: z,y,x)
            graph_analyzer = AirwayGraphAnalyzer(output_path, carina_coords=(int(carina_z), int(carina_y), int(carina_x)))
            graph_analyzer.compute_skeleton()
            graph_analyzer.analyze_connected_components()
            graph_analyzer.build_graph()
            # Ensure carina is identified in the graph analyzer (will compute if needed)
            graph_analyzer.identify_carina()
            # Save carina coordinates to JSON
            try:
                carina_info = graph_analyzer.get_carina_coordinates()
                json_path = os.path.join(output_dir, "carina_coordinates.json")
                with open(json_path, 'w') as jf:
                    json.dump(carina_info, jf, indent=2)
                print(f"✓ Saved carina coordinates JSON: {json_path}")
            except Exception as e:
                print(f"⚠ Warning: could not save carina JSON: {e}")

            # For a clearer, ordered visualization draw continuous branches
            # instead of just nodes. Compute branch metrics (length/diameter)
            # and generation assignment so that the plot shows well-defined
            # branches and a prominent carina marker.
            try:
                # Calculate lengths and diameters along branches
                graph_analyzer.calculate_branch_lengths()
                graph_analyzer.analyze_diameters()
                graph_analyzer.merge_branch_metrics()

                # Assign Weibel generations (used to color/organize branches)
                try:
                    graph_analyzer.assign_generations_weibel()
                except Exception:
                    # If generation assignment fails, still attempt branch plot
                    pass

                graph_img_path = os.path.join(output_dir, "bronchi_graph_with_carina.png")
                # This method draws full branches (continuous lines) and highlights
                # the carina with a large red marker, producing a much clearer tree
                # image than the node-scatter view.
                graph_analyzer.visualize_weibel_generations_with_carina(save_path=graph_img_path)
                print(f"✓ Saved graph image with carina: {graph_img_path}")
            except Exception as e:
                print(f"⚠ Warning: could not generate branch-ordered graph visualization: {e}")

            # Additionally, save a 3D skeleton view with the carina overlaid (PNG)
            try:
                import matplotlib.pyplot as plt

                skel = graph_analyzer.skeleton
                if skel is not None:
                    skel_coords = np.argwhere(skel > 0)
                    if len(skel_coords) > 0:
                        subsample = max(1, len(skel_coords) // 20000)
                        skel_coords = skel_coords[::subsample]

                        fig = plt.figure(figsize=(12, 10))
                        ax = fig.add_subplot(111, projection='3d')
                        ax.scatter(skel_coords[:, 2], skel_coords[:, 1], skel_coords[:, 0],
                                   c='lightblue', marker='.', s=1, alpha=0.6, label='Skeleton')

                        # Carina voxel coordinates
                        try:
                            carina_vox = graph_analyzer.get_carina_coordinates()['voxel_coordinates']
                            cz, cy, cx = int(carina_vox['z']), int(carina_vox['y']), int(carina_vox['x'])
                            ax.scatter(cx, cy, cz, c='red', s=200, marker='o', edgecolors='black', linewidths=1.5,
                                       label='Carina')
                            ax.text(cx, cy, cz + 5, f'Carina\nZ:{cz} Y:{cy} X:{cx}', color='red', fontsize=9,
                                    ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                        except Exception:
                            pass

                        ax.set_xlabel('X (voxel)')
                        ax.set_ylabel('Y (voxel)')
                        ax.set_zlabel('Z (voxel)')
                        ax.set_title('3D Skeleton with Carina Highlighted')
                        ax.legend()

                        skel_img_path = os.path.join(output_dir, "skeleton_with_carina.png")
                        plt.savefig(skel_img_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        print(f"✓ Saved skeleton image with carina: {skel_img_path}")
            except Exception as e:
                print(f"⚠ Warning: could not generate skeleton PNG: {e}")
        except Exception as e:
            print(f"⚠ Warning: could not generate graph visualization: {e}")
    
    return bronchi_mask, (carina_z, carina_y, carina_x), confidence, detector

if __name__ == "__main__":
    import os
    import sys
    
    print("="*70)
    print(" "*10 + "ENHANCED CONSERVATIVE CARINA DETECTION")
    print("="*70)
    
    # Test con file di esempio
    mask_path = "airway_segmentation/1.2.840.113704.1.111.5964.1132388375.7_airwayfull.nii.gz"
    
    if not os.path.exists(mask_path):
        print(f"❌ ERROR: File not found: {mask_path}")
        sys.exit(1)
    
    print(f"Processing: {mask_path}")
    
    # Parametri conservativi
    precut_z = 390  # Pre-cut fisso per rimuovere trachea superiore
    
    try:
        bronchi_mask, carina_coords, confidence, detector = integrate_with_pipeline(
            mask_path, 
            spacing=None,
            precut_z=precut_z,
            save_output=True
        )
        
        print(f"\n{'='*70}")
        print("FINAL ENHANCED RESULTS:")
        print(f"{'='*70}")
        print(f"Carina: z={carina_coords[0]}, y={carina_coords[1]}, x={carina_coords[2]}")
        print(f"Confidence: {confidence:.2f}/5.0")
        print(f"Pre-cut applied at: z={precut_z}")
        
        if detector.trachea_top_z and detector.trachea_bottom_z:
            print(f"Trachea identified: z={detector.trachea_top_z} to z={detector.trachea_bottom_z}")
            print(f"Trachea length: {detector.trachea_length} slices ({detector.trachea_length * detector.spacing[2]:.1f} mm)")
        
        print(f"Bronchi voxels preserved: {np.sum(bronchi_mask > 0):,}")
        
        # Statistiche di pulizia
        original_voxels = np.sum(detector.mask > 0)
        final_voxels = np.sum(bronchi_mask > 0)
        removed_voxels = original_voxels - final_voxels
        
        print(f"Original voxels: {original_voxels:,}")
        print(f"Removed voxels: {removed_voxels:,} ({removed_voxels/original_voxels*100:.1f}%)")
        print(f"Preservation rate: {final_voxels/original_voxels*100:.1f}%")
        
    except Exception as e:
        print(f"\n❌ ERROR during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)