# narrative_attractor_dynamics.py
# -*- coding: utf-8 -*-
"""
Dynamic Narrative Space Explorer - Complexity Science Approach

This script extends the narrative similarity concept by modeling narratives as
points in a dynamic embedding space with attractor basins. It identifies:
1. Global attractors (narrative gravitational centers)
2. Basin boundaries and phase transitions
3. Trajectory analysis (how narratives evolve sentence-by-sentence)
4. Multi-scale attractor hierarchies

Usage:
  python narrative_attractor_dynamics.py explore --input_path data/narratives.jsonl --out_dir artifacts/dynamics
  python narrative_attractor_dynamics.py visualize --space_path artifacts/dynamics/space.joblib --out_dir viz
"""

from core.core import argparse
import os
from core.core import warnings
warnings.filterwarnings("ignore")

from core.core import numpy as np
from core.core import pandas as pd
from core.core import regex as re
from core.sentence_transformers import SentenceTransformer
from core.sklearn.cluster import DBSCAN, KMeans
from core.sklearn.decomposition import PCA
from core.sklearn.manifold import TSNE
from core.core import joblib
from core.core import json

try:
    import matplotlib.pyplot as plt
    from core.mpl_toolkits.mplot3d import Axes3D
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# -------------------- Utils from core.original --------------------

def _is_jsonl(path: str) -> bool:
    return path.lower().endswith(".jsonl") or path.lower().endswith(".ndjson")

def read_table(path: str) -> pd.DataFrame:
    if _is_jsonl(path):
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)

def write_table(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if _is_jsonl(path):
        df.to_json(path, orient="records", lines=True, force_ascii=False)
    else:
        df.to_csv(path, index=False)

def sent_split(text: str):
    sents = re.split(r'(?<=[\.!?])\s+|\n+', (text or "").strip())
    sents = [s.strip() for s in sents if s.strip()]
    return sents if sents else [text.strip()]


# -------------------- Encoder --------------------

class Encoder:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", device=None):
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str):
        return self.model.encode(text or "", normalize_embeddings=True, convert_to_numpy=True)

    def embed_sents(self, sents):
        if not sents: sents = [""]
        return self.model.encode(sents, normalize_embeddings=True, convert_to_numpy=True)


# -------------------- Attractor Physics --------------------

class AttractorBasin:
    """Represents a single attractor in narrative space"""
    def __init__(self, center, points_idx, strength=1.0, parent_id=None, level=0):
        self.center = center  # embedding vector
        self.points_idx = points_idx  # indices of points in this basin
        self.strength = strength  # basin depth/strength
        self.radius = None  # will be computed
        self.basin_depth = None  # stability measure
        self.escape_energy = None  # energy needed to leave basin
        self.parent_id = parent_id  # for hierarchical structure
        self.level = level  # hierarchy level (0=global, 1=sub-basin, etc)
        self.children = []  # sub-basins

    def compute_radius(self, all_points):
        """Compute effective radius of basin"""
        if len(self.points_idx) == 0:
            self.radius = 0.0
            return
        basin_points = all_points[self.points_idx]
        dists = np.linalg.norm(basin_points - self.center[None, :], axis=1)
        self.radius = float(np.percentile(dists, 95))  # 95th percentile

    def compute_basin_depth(self, all_points, other_attractors):
        """Compute basin stability (depth of potential well)"""
        if len(self.points_idx) == 0:
            self.basin_depth = 0.0
            self.escape_energy = 0.0
            return

        basin_points = all_points[self.points_idx]

        # Basin depth: average distance from core.center
        internal_dists = np.linalg.norm(basin_points - self.center[None, :], axis=1)
        self.basin_depth = float(np.mean(internal_dists))

        # Escape energy: minimum energy to reach nearest attractor
        if other_attractors:
            nearest_dists = []
            for point in basin_points:
                min_other_dist = min([np.linalg.norm(point - other.center)
                                     for other in other_attractors if other is not self])
                nearest_dists.append(min_other_dist)
            self.escape_energy = float(np.mean(nearest_dists))
        else:
            self.escape_energy = self.radius

    def potential_energy(self, point):
        """Compute potential energy at a point (distance-based)"""
        dist = np.linalg.norm(point - self.center)
        # Harmonic oscillator potential: E = 0.5 * k * r^2
        return 0.5 * self.strength * (dist ** 2)

    def force_vector(self, point):
        """Compute force vector pulling toward center"""
        diff = self.center - point
        dist = np.linalg.norm(diff)
        if dist < 1e-8:
            return np.zeros_like(diff)
        # F = -∇E = -k*r (restoring force)
        return self.strength * diff


class NarrativeSpace:
    """Models the full dynamic narrative embedding space"""

    def __init__(self, encoder_name="sentence-transformers/all-mpnet-base-v2"):
        self.encoder = Encoder(encoder_name)
        self.narratives = []  # list of dict with text, id, etc
        self.doc_embeddings = None  # (N, D) document embeddings
        self.sent_embeddings = []  # list of (n_sents, D) per document
        self.attractors = []  # list of AttractorBasin objects
        self.global_attractor = None  # overall center of mass

    def add_narratives(self, texts, ids=None, metadata=None):
        """Add narratives to the space"""
        if ids is None:
            ids = list(range(len(self.narratives), len(self.narratives) + len(texts)))
        if metadata is None:
            metadata = [{}] * len(texts)

        for i, text in enumerate(texts):
            self.narratives.append({
                'id': ids[i],
                'text': text,
                'metadata': metadata[i]
            })

    def compute_embeddings(self):
        """Compute all embeddings"""
        print(f"[*] Computing embeddings for {len(self.narratives)} narratives...")

        # Document-level embeddings
        doc_embs = []
        sent_embs = []

        for i, narr in enumerate(self.narratives):
            if i % 50 == 0:
                print(f"    {i}/{len(self.narratives)}...")

            text = narr['text']
            doc_emb = self.encoder.embed_text(text)
            doc_embs.append(doc_emb)

            sents = sent_split(text)
            sent_emb = self.encoder.embed_sents(sents)
            sent_embs.append(sent_emb)

        self.doc_embeddings = np.vstack(doc_embs)
        self.sent_embeddings = sent_embs

        # Compute global center of mass
        self.global_attractor = self.doc_embeddings.mean(axis=0)
        print(f"[✓] Embeddings computed (dim={self.doc_embeddings.shape[1]})")

    def discover_attractors(self, method='kmeans', n_attractors=5, eps=0.3, min_samples=3):
        """Discover attractor basins using clustering"""
        print(f"[*] Discovering attractors using {method}...")

        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_attractors, random_state=42, n_init=10)
            labels = clusterer.fit_predict(self.doc_embeddings)
            centers = clusterer.cluster_centers_

        elif method == 'dbscan':
            clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
            labels = clusterer.fit_predict(self.doc_embeddings)
            # Compute centers for each cluster
            unique_labels = set(labels) - {-1}
            centers = []
            valid_labels = []
            for lbl in sorted(unique_labels):
                mask = labels == lbl
                center = self.doc_embeddings[mask].mean(axis=0)
                centers.append(center)
                valid_labels.append(lbl)
            centers = np.vstack(centers) if centers else np.array([])
            # Remap labels to contiguous
            label_map = {old: new for new, old in enumerate(valid_labels)}
            label_map[-1] = -1
            labels = np.array([label_map[l] for l in labels])

        else:
            raise ValueError(f"Unknown method: {method}")

        # Create AttractorBasin objects
        self.attractors = []
        unique_labels = set(labels) - {-1}

        for lbl in sorted(unique_labels):
            mask = labels == lbl
            points_idx = np.where(mask)[0]
            center = centers[lbl] if method == 'kmeans' else centers[list(sorted(unique_labels)).index(lbl)]

            basin = AttractorBasin(center, points_idx)
            basin.compute_radius(self.doc_embeddings)
            basin.strength = len(points_idx) / len(self.doc_embeddings)  # proportional to occupancy
            self.attractors.append(basin)

        print(f"[✓] Found {len(self.attractors)} attractors")
        for i, att in enumerate(self.attractors):
            print(f"    Attractor {i}: {len(att.points_idx)} narratives, radius={att.radius:.3f}")

        return labels

    def compute_trajectory_features(self, narrative_idx):
        """Compute how a narrative evolves sentence-by-sentence"""
        sent_embs = self.sent_embeddings[narrative_idx]

        # Distance to global attractor over time
        dists_to_global = np.linalg.norm(sent_embs - self.global_attractor[None, :], axis=1)

        # Distance to nearest attractor over time
        dists_to_nearest = []
        nearest_attractor_ids = []

        for sent_emb in sent_embs:
            min_dist = float('inf')
            nearest_id = -1
            for i, att in enumerate(self.attractors):
                dist = np.linalg.norm(sent_emb - att.center)
                if dist < min_dist:
                    min_dist = dist
                    nearest_id = i
            dists_to_nearest.append(min_dist)
            nearest_attractor_ids.append(nearest_id)

        # Trajectory curvature (rate of change in direction)
        if len(sent_embs) > 2:
            deltas = np.diff(sent_embs, axis=0)
            delta_norms = np.linalg.norm(deltas, axis=1, keepdims=True) + 1e-8
            delta_dirs = deltas / delta_norms
            curvatures = np.linalg.norm(np.diff(delta_dirs, axis=0), axis=1)
            mean_curvature = float(curvatures.mean())
        else:
            mean_curvature = 0.0

        # Basin transitions (how many times the nearest attractor changes)
        transitions = sum(1 for i in range(len(nearest_attractor_ids)-1)
                         if nearest_attractor_ids[i] != nearest_attractor_ids[i+1])

        return {
            'dists_to_global': dists_to_global.tolist(),
            'dists_to_nearest': dists_to_nearest,
            'nearest_attractor_ids': nearest_attractor_ids,
            'mean_curvature': mean_curvature,
            'basin_transitions': transitions,
            'trajectory_length': float(np.sum(np.linalg.norm(np.diff(sent_embs, axis=0), axis=1))),
            'start_to_end_dist': float(np.linalg.norm(sent_embs[-1] - sent_embs[0])) if len(sent_embs) > 1 else 0.0
        }

    def compute_all_trajectories(self):
        """Compute trajectory features for all narratives"""
        print("[*] Computing narrative trajectories...")
        trajectories = []
        for i in range(len(self.narratives)):
            if i % 50 == 0:
                print(f"    {i}/{len(self.narratives)}...")
            traj = self.compute_trajectory_features(i)
            traj['narrative_id'] = self.narratives[i]['id']
            trajectories.append(traj)
        print("[✓] Trajectories computed")
        return trajectories

    def compute_basin_statistics(self):
        """Compute statistics about attractor basins"""
        # First compute basin depths and escape energies
        for att in self.attractors:
            att.compute_basin_depth(self.doc_embeddings, self.attractors)

        stats = []
        for i, att in enumerate(self.attractors):
            basin_points = self.doc_embeddings[att.points_idx]

            # Cohesion (avg distance to center)
            cohesion = float(np.mean(np.linalg.norm(basin_points - att.center[None, :], axis=1)))

            # Separation (distance to nearest other attractor)
            separations = [np.linalg.norm(att.center - other.center)
                          for j, other in enumerate(self.attractors) if i != j]
            min_separation = float(min(separations)) if separations else 0.0

            # Variance (spread of points)
            variance = float(np.mean(np.var(basin_points, axis=0)))

            stats.append({
                'attractor_id': i,
                'n_narratives': len(att.points_idx),
                'strength': att.strength,
                'radius': att.radius,
                'basin_depth': att.basin_depth,
                'escape_energy': att.escape_energy,
                'cohesion': cohesion,
                'min_separation': min_separation,
                'variance': variance,
                'stability_ratio': att.escape_energy / (att.basin_depth + 1e-8)  # stability measure
            })

        return stats

    def compute_lyapunov_exponent(self, narrative_idx, epsilon=1e-3):
        """
        Estimate local Lyapunov exponent for a narrative trajectory.
        Measures sensitivity to initial conditions / trajectory stability.
        """
        sent_embs = self.sent_embeddings[narrative_idx]
        if len(sent_embs) < 3:
            return 0.0

        # Compute divergence rate along trajectory
        log_divergences = []

        for i in range(len(sent_embs) - 1):
            # Current state
            x_i = sent_embs[i]

            # Perturb slightly
            perturbation = np.random.randn(*x_i.shape) * epsilon
            perturbation = perturbation / (np.linalg.norm(perturbation) + 1e-8) * epsilon
            x_i_perturbed = x_i + perturbation

            # Evolve both forward one step
            x_next = sent_embs[i + 1]

            # Compute "evolved" perturbed state by finding force field
            # We approximate the dynamical system using attractor forces
            force = np.zeros_like(x_i_perturbed)
            for att in self.attractors:
                force += att.force_vector(x_i_perturbed)

            # Simple Euler step
            dt = 0.1
            x_next_perturbed = x_i_perturbed + force * dt

            # Measure divergence
            d0 = epsilon
            d1 = np.linalg.norm(x_next - x_next_perturbed)

            if d1 > 1e-10:
                log_divergences.append(np.log(d1 / (d0 + 1e-10)))

        if not log_divergences:
            return 0.0

        # Average logarithmic divergence rate
        lyapunov = float(np.mean(log_divergences))
        return lyapunov

    def discover_hierarchical_attractors(self, max_depth=2, min_cluster_size=3):
        """
        Discover nested attractor basins (fractals/hierarchies).
        Each major basin is recursively subdivided to find sub-attractors.
        """
        print(f"[*] Discovering hierarchical attractors (max_depth={max_depth})...")

        if not self.attractors:
            print("[!] No attractors found. Run discover_attractors() first.")
            return

        def subdivide_basin(parent_basin, parent_id, depth):
            """Recursively subdivide a basin"""
            if depth >= max_depth:
                return []

            if len(parent_basin.points_idx) < min_cluster_size * 2:
                return []

            # Get points in this basin
            basin_points = self.doc_embeddings[parent_basin.points_idx]

            # Try to split into sub-basins using k-means (k=2)
            from core.sklearn.cluster import KMeans
            n_sub = min(3, len(parent_basin.points_idx) // min_cluster_size)
            if n_sub < 2:
                return []

            kmeans = KMeans(n_clusters=n_sub, random_state=42, n_init=10)
            sub_labels = kmeans.fit_predict(basin_points)

            sub_basins = []
            for sub_lbl in range(n_sub):
                mask = sub_labels == sub_lbl
                if np.sum(mask) < min_cluster_size:
                    continue

                # Map back to global indices
                sub_points_idx = parent_basin.points_idx[mask]
                sub_center = kmeans.cluster_centers_[sub_lbl]

                sub_basin = AttractorBasin(
                    sub_center,
                    sub_points_idx,
                    strength=len(sub_points_idx) / len(self.doc_embeddings),
                    parent_id=parent_id,
                    level=depth + 1
                )
                sub_basin.compute_radius(self.doc_embeddings)
                sub_basins.append(sub_basin)

                # Add to parent's children
                parent_basin.children.append(sub_basin)

                # Recurse
                deeper_basins = subdivide_basin(sub_basin, len(self.attractors) + len(sub_basins) - 1, depth + 1)
                sub_basins.extend(deeper_basins)

            return sub_basins

        # Start subdivision from core.each top-level attractor
        all_sub_basins = []
        for i, att in enumerate(self.attractors):
            sub_basins = subdivide_basin(att, i, 0)
            all_sub_basins.extend(sub_basins)

        print(f"[✓] Found {len(all_sub_basins)} sub-attractors across {max_depth} levels")
        return all_sub_basins

    def compute_bifurcation_signature(self, narrative_idx):
        """
        Analyze if a narrative shows bifurcation behavior (sudden phase transitions).
        Detects where trajectory abruptly switches attractor basins.
        """
        sent_embs = self.sent_embeddings[narrative_idx]
        if len(sent_embs) < 3:
            return {
                'has_bifurcation': False,
                'bifurcation_points': [],
                'phase_transition_strength': 0.0
            }

        # Find which attractor each sentence is closest to
        attractor_sequence = []
        for sent_emb in sent_embs:
            dists = [np.linalg.norm(sent_emb - att.center) for att in self.attractors]
            attractor_sequence.append(np.argmin(dists))

        # Detect transitions
        bifurcation_points = []
        for i in range(len(attractor_sequence) - 1):
            if attractor_sequence[i] != attractor_sequence[i + 1]:
                # Measure transition strength (distance between attractor centers)
                att_from core.= self.attractors[attractor_sequence[i]]
                att_to = self.attractors[attractor_sequence[i + 1]]
                transition_dist = np.linalg.norm(att_from.center - att_to.center)

                bifurcation_points.append({
                    'sentence_idx': i,
                    'from_attractor': int(attractor_sequence[i]),
                    'to_attractor': int(attractor_sequence[i + 1]),
                    'transition_strength': float(transition_dist)
                })

        has_bifurcation = len(bifurcation_points) > 0
        avg_strength = float(np.mean([bp['transition_strength'] for bp in bifurcation_points])) if bifurcation_points else 0.0

        return {
            'has_bifurcation': has_bifurcation,
            'n_transitions': len(bifurcation_points),
            'bifurcation_points': bifurcation_points,
            'phase_transition_strength': avg_strength
        }

    def reduce_dimensions(self, method='pca', n_components=3):
        """Reduce dimensionality for visualization"""
        print(f"[*] Reducing to {n_components}D using {method}...")

        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
            reduced = reducer.fit_transform(self.doc_embeddings)
            reduced_attractors = reducer.transform(np.vstack([a.center for a in self.attractors]))
            reduced_global = reducer.transform(self.global_attractor.reshape(1, -1))[0]

        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(self.doc_embeddings)-1))
            # Include attractors in the embedding
            all_points = np.vstack([
                self.doc_embeddings,
                np.vstack([a.center for a in self.attractors]),
                self.global_attractor.reshape(1, -1)
            ])
            reduced_all = reducer.fit_transform(all_points)
            reduced = reduced_all[:len(self.doc_embeddings)]
            reduced_attractors = reduced_all[len(self.doc_embeddings):len(self.doc_embeddings)+len(self.attractors)]
            reduced_global = reduced_all[-1]

        elif method == 'umap' and HAS_UMAP:
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            all_points = np.vstack([
                self.doc_embeddings,
                np.vstack([a.center for a in self.attractors]),
                self.global_attractor.reshape(1, -1)
            ])
            reduced_all = reducer.fit_transform(all_points)
            reduced = reduced_all[:len(self.doc_embeddings)]
            reduced_attractors = reduced_all[len(self.doc_embeddings):len(self.doc_embeddings)+len(self.attractors)]
            reduced_global = reduced_all[-1]

        else:
            raise ValueError(f"Unknown or unavailable method: {method}")

        print("[✓] Dimension reduction complete")
        return reduced, reduced_attractors, reduced_global


# -------------------- Exploration & Analysis --------------------

def explore_narrative_space(input_path, out_dir, encoder_name, n_attractors, cluster_method,
                           compute_advanced=True, hierarchical=True):
    """Main exploration pipeline with complexity science features"""
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    df = read_table(input_path)

    # Determine text column
    text_col = None
    for col in ['text', 'anchor_text', 'narrative', 'content']:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        raise ValueError("Could not find text column. Expected one of: text, anchor_text, narrative, content")

    texts = df[text_col].tolist()
    ids = df['id'].tolist() if 'id' in df.columns else list(range(len(texts)))

    # Build space
    space = NarrativeSpace(encoder_name)
    space.add_narratives(texts, ids)
    space.compute_embeddings()

    # Discover attractors
    labels = space.discover_attractors(method=cluster_method, n_attractors=n_attractors)

    # Analyze trajectories
    trajectories = space.compute_all_trajectories()
    traj_df = pd.DataFrame(trajectories)

    # Basin statistics (includes depth and escape energy)
    basin_stats = space.compute_basin_statistics()
    basin_df = pd.DataFrame(basin_stats)

    # Advanced complexity science features
    if compute_advanced:
        print("[*] Computing advanced complexity features...")

        # Lyapunov exponents
        print("    - Lyapunov exponents...")
        lyapunovs = []
        for i in range(len(space.narratives)):
            if i % 50 == 0:
                print(f"      {i}/{len(space.narratives)}...")
            lyap = space.compute_lyapunov_exponent(i)
            lyapunovs.append(lyap)

        # Bifurcation analysis
        print("    - Bifurcation signatures...")
        bifurcations = []
        for i in range(len(space.narratives)):
            if i % 50 == 0:
                print(f"      {i}/{len(space.narratives)}...")
            bif = space.compute_bifurcation_signature(i)
            bifurcations.append({
                'narrative_id': space.narratives[i]['id'],
                'has_bifurcation': bif['has_bifurcation'],
                'n_transitions': bif['n_transitions'],
                'phase_transition_strength': bif['phase_transition_strength']
            })

        # Add to assignments
        assign_df = pd.DataFrame({
            'narrative_id': ids,
            'attractor_id': labels,
            'dist_to_global': [float(np.linalg.norm(space.doc_embeddings[i] - space.global_attractor))
                              for i in range(len(ids))],
            'lyapunov_exponent': lyapunovs
        })

        bifurc_df = pd.DataFrame(bifurcations)
        assign_df = assign_df.merge(bifurc_df, on='narrative_id', how='left')

        write_table(assign_df, os.path.join(out_dir, "attractor_assignments.csv"))

        print("[✓] Advanced features computed")
    else:
        # Save basic attractor assignments
        assign_df = pd.DataFrame({
            'narrative_id': ids,
            'attractor_id': labels,
            'dist_to_global': [float(np.linalg.norm(space.doc_embeddings[i] - space.global_attractor))
                              for i in range(len(ids))]
        })
        write_table(assign_df, os.path.join(out_dir, "attractor_assignments.csv"))

    # Hierarchical attractor discovery
    sub_basins = []
    if hierarchical and len(space.attractors) > 0:
        sub_basins = space.discover_hierarchical_attractors(max_depth=2, min_cluster_size=5)

        if sub_basins:
            # Save hierarchical structure
            hierarchy_data = []
            for i, sub in enumerate(sub_basins):
                hierarchy_data.append({
                    'sub_attractor_id': i,
                    'parent_attractor_id': sub.parent_id,
                    'level': sub.level,
                    'n_narratives': len(sub.points_idx),
                    'radius': sub.radius
                })
            hierarchy_df = pd.DataFrame(hierarchy_data)
            write_table(hierarchy_df, os.path.join(out_dir, "hierarchical_attractors.csv"))

    # Save results
    joblib.dump(space, os.path.join(out_dir, "narrative_space.joblib"))
    write_table(traj_df, os.path.join(out_dir, "trajectories.jsonl"))
    write_table(basin_df, os.path.join(out_dir, "basin_statistics.csv"))

    # Summary report
    report = {
        'n_narratives': len(texts),
        'n_attractors': len(space.attractors),
        'n_sub_attractors': len(sub_basins),
        'embedding_dim': space.encoder.dim,
        'global_attractor_center': space.global_attractor.tolist(),
        'attractor_summary': basin_stats,
        'advanced_features_computed': compute_advanced,
        'hierarchical_analysis': hierarchical and len(sub_basins) > 0
    }

    with open(os.path.join(out_dir, "summary.json"), 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n[✓] Exploration complete! Results saved to {out_dir}/")
    print(f"    - narrative_space.joblib (pickled space object)")
    print(f"    - trajectories.jsonl (sentence-level dynamics)")
    print(f"    - basin_statistics.csv (attractor properties + stability)")
    print(f"    - attractor_assignments.csv (assignments + Lyapunov + bifurcations)")
    if hierarchical and sub_basins:
        print(f"    - hierarchical_attractors.csv (nested basin structure)")
    print(f"    - summary.json (overview)")


# -------------------- Visualization --------------------

def visualize_space(space_path, out_dir, method='pca'):
    """Generate visualizations of the narrative space"""
    if not HAS_PLT:
        print("[!] matplotlib not available. Skipping visualization.")
        return

    os.makedirs(out_dir, exist_ok=True)
    space = joblib.load(space_path)

    # Reduce dimensions
    reduced, reduced_attractors, reduced_global = space.reduce_dimensions(method=method, n_components=3)

    # Assign colors based on nearest attractor
    labels = []
    for i in range(len(space.doc_embeddings)):
        dists = [np.linalg.norm(space.doc_embeddings[i] - att.center) for att in space.attractors]
        labels.append(np.argmin(dists))
    labels = np.array(labels)

    # 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot narratives
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2],
                        c=labels, cmap='tab10', alpha=0.6, s=30)

    # Plot attractors
    ax.scatter(reduced_attractors[:, 0], reduced_attractors[:, 1], reduced_attractors[:, 2],
              c='red', marker='*', s=500, edgecolors='black', linewidth=2, label='Attractors')

    # Plot global attractor
    ax.scatter([reduced_global[0]], [reduced_global[1]], [reduced_global[2]],
              c='gold', marker='D', s=300, edgecolors='black', linewidth=2, label='Global Center')

    ax.set_xlabel(f'{method.upper()} 1')
    ax.set_ylabel(f'{method.upper()} 2')
    ax.set_zlabel(f'{method.upper()} 3')
    ax.set_title('Narrative Space - Attractor Basins')
    ax.legend()
    plt.colorbar(scatter, label='Basin ID')

    plt.savefig(os.path.join(out_dir, "space_3d.png"), dpi=150, bbox_inches='tight')
    print(f"[✓] Saved 3D visualization to {out_dir}/space_3d.png")

    # 2D projection
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.6, s=30)
    ax.scatter(reduced_attractors[:, 0], reduced_attractors[:, 1],
              c='red', marker='*', s=500, edgecolors='black', linewidth=2, label='Attractors')
    ax.scatter([reduced_global[0]], [reduced_global[1]],
              c='gold', marker='D', s=300, edgecolors='black', linewidth=2, label='Global Center')

    ax.set_xlabel(f'{method.upper()} 1')
    ax.set_ylabel(f'{method.upper()} 2')
    ax.set_title('Narrative Space - 2D Projection')
    ax.legend()
    plt.colorbar(scatter, label='Basin ID')

    plt.savefig(os.path.join(out_dir, "space_2d.png"), dpi=150, bbox_inches='tight')
    print(f"[✓] Saved 2D visualization to {out_dir}/space_2d.png")

    plt.close('all')


# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser(description="Explore dynamic narrative attractor spaces")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Explore command
    ap_exp = sub.add_parser("explore", help="Build and analyze narrative space")
    ap_exp.add_argument("--input_path", required=True, help="Input JSONL/CSV with narratives")
    ap_exp.add_argument("--out_dir", default="artifacts/dynamics", help="Output directory")
    ap_exp.add_argument("--encoder_name", default="sentence-transformers/all-mpnet-base-v2")
    ap_exp.add_argument("--n_attractors", type=int, default=5, help="Number of attractors (kmeans)")
    ap_exp.add_argument("--cluster_method", choices=['kmeans', 'dbscan'], default='kmeans')
    ap_exp.add_argument("--no-advanced", dest="compute_advanced", action="store_false",
                       help="Disable Lyapunov & bifurcation analysis (faster)")
    ap_exp.add_argument("--no-hierarchical", dest="hierarchical", action="store_false",
                       help="Disable hierarchical attractor discovery")
    ap_exp.set_defaults(compute_advanced=True, hierarchical=True)

    # Visualize command
    ap_viz = sub.add_parser("visualize", help="Visualize narrative space")
    ap_viz.add_argument("--space_path", required=True, help="Path to narrative_space.joblib")
    ap_viz.add_argument("--out_dir", default="viz", help="Output directory for plots")
    ap_viz.add_argument("--method", choices=['pca', 'tsne', 'umap'], default='pca')

    args = ap.parse_args()

    if args.cmd == "explore":
        explore_narrative_space(
            args.input_path,
            args.out_dir,
            args.encoder_name,
            args.n_attractors,
            args.cluster_method,
            args.compute_advanced,
            args.hierarchical
        )

    elif args.cmd == "visualize":
        visualize_space(args.space_path, args.out_dir, args.method)


if __name__ == "__main__":
    main()
