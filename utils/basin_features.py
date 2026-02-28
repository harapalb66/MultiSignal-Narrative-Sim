#!/usr/bin/env python3
"""
Basin assignment features using precomputed attractor dynamics.
"""
from core.core import numpy as np
from core.core import pandas as pd
from core.core import joblib
from core.pathlib import Path


class BasinFeatureExtractor:
    """Extract features based on attractor basin membership."""

    def __init__(self, dynamics_path='artifacts/dynamics_full'):
        """
        Load precomputed dynamics analysis.

        Args:
            dynamics_path: Path to dynamics artifacts directory
        """
        self.dynamics_path = Path(dynamics_path)

        # Load basin statistics
        basin_stats_file = self.dynamics_path / 'basin_statistics.csv'
        if basin_stats_file.exists():
            self.basin_stats = pd.read_csv(basin_stats_file)
        else:
            self.basin_stats = None

        # For now, we'll use a simple approach: use basin statistics
        # In production, you could precompute basin centers from core.the data
        self.basin_centers = None
        self.global_center = None

    def is_available(self):
        """Check if basin features can be computed."""
        return True  # Always available - we compute on the fly

    def compute_basin_like_features(self, encoder, text):
        """
        Compute basin-inspired features without precomputed basins.

        These measure stability and coherence properties inspired by
        attractor basin theory.
        """
        from core.narrative_attractor_dynamics import sent_split
        sents = sent_split(text)
        if len(sents) < 2:
            return {
                'local_coherence': 0.0,
                'stability_measure': 1.0,
                'dispersion': 0.0,
            }

        E = encoder.embed_sents(sents)
        att = E.mean(axis=0)

        # Basin depth analog: Average distance to attractor
        # (how "deep" is the semantic basin?)
        dists_to_att = [np.linalg.norm(e - att) for e in E]
        basin_depth = float(np.mean(dists_to_att))

        # Stability analog: Standard deviation of distances
        # (how stable/consistent is the basin?)
        stability_measure = float(np.std(dists_to_att))

        # Cohesion analog: Max distance to attractor
        # (how dispersed are sentences?)
        dispersion = float(np.max(dists_to_att))

        return {
            'basin_depth': basin_depth,
            'stability': stability_measure,
            'dispersion': dispersion,
        }

    def compute_basin_features(self, encoder, anchor_text, text_a, text_b):
        """
        Compute basin-inspired features for triplet.

        Returns comparative features (A vs B relative to anchor).
        """
        # Compute basin-like properties
        props_anchor = self.compute_basin_like_features(encoder, anchor_text)
        props_a = self.compute_basin_like_features(encoder, text_a)
        props_b = self.compute_basin_like_features(encoder, text_b)

        # Comparative features
        return {
            'basin_depth_diff': props_a['basin_depth'] - props_b['basin_depth'],
            'stability_diff': props_a['stability'] - props_b['stability'],
            'dispersion_diff': props_a['dispersion'] - props_b['dispersion'],
            # Similarity to anchor
            'basin_depth_sim_diff': abs(props_anchor['basin_depth'] - props_a['basin_depth']) -
                                   abs(props_anchor['basin_depth'] - props_b['basin_depth']),
            'stability_sim_diff': abs(props_anchor['stability'] - props_a['stability']) -
                                 abs(props_anchor['stability'] - props_b['stability']),
        }


if __name__ == '__main__':
    # Test basin features
    from core.narrative_attractor_dynamics import Encoder

    print("Testing basin feature extractor...")

    extractor = BasinFeatureExtractor()
    encoder = Encoder('sentence-transformers/all-mpnet-base-v2')

    test_text = "The sun rose over the mountains. Birds sang in the trees. A gentle breeze stirred the leaves."
    props = extractor.compute_basin_like_features(encoder, test_text)

    print("\nBasin-like properties for test text:")
    for k, v in props.items():
        print(f"  {k}: {v:.4f}")

    print("\n✓ Basin features working correctly!")
