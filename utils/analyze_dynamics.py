#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick analysis script to explore narrative dynamics results.
Provides insights into the discovered attractor landscape.
"""

from core.core import pandas as pd
from core.core import numpy as np
from core.core import json
from core import sys
from core.pathlib import Path

def analyze_results(dynamics_dir):
    """Analyze narrative dynamics results"""
    dynamics_dir = Path(dynamics_dir)

    print("=" * 70)
    print("NARRATIVE ATTRACTOR DYNAMICS ANALYSIS")
    print("=" * 70)

    # Load summary
    with open(dynamics_dir / "summary.json") as f:
        summary = json.load(f)

    print(f"\n DATASET OVERVIEW")
    print(f"   Total narratives: {summary['n_narratives']}")
    print(f"   Main attractors: {summary['n_attractors']}")
    print(f"   Sub-attractors: {summary['n_sub_attractors']}")
    print(f"   Embedding dimension: {summary['embedding_dim']}")

    # Load basin statistics
    basin_stats = pd.read_csv(dynamics_dir / "basin_statistics.csv")

    print(f"\n ATTRACTOR LANDSCAPE")
    print(f"   Most stable basin: #{basin_stats['stability_ratio'].idxmax()} "
          f"(stability={basin_stats['stability_ratio'].max():.3f})")
    print(f"   Deepest basin: #{basin_stats['basin_depth'].idxmax()} "
          f"(depth={basin_stats['basin_depth'].max():.3f})")
    print(f"   Largest basin: #{basin_stats['radius'].idxmax()} "
          f"(radius={basin_stats['radius'].max():.3f})")
    print(f"   Most populous: #{basin_stats['n_narratives'].idxmax()} "
          f"({basin_stats['n_narratives'].max()} narratives)")

    # Load assignments
    assignments = pd.read_csv(dynamics_dir / "attractor_assignments.csv")

    print(f"\n NARRATIVE DYNAMICS")
    if 'lyapunov_exponent' in assignments.columns:
        avg_lyap = assignments['lyapunov_exponent'].mean()
        print(f"   Average Lyapunov exponent: {avg_lyap:.3f}")
        print(f"   Interpretation: {'Chaotic/unpredictable' if avg_lyap > 0 else 'Stable/predictable'}")

    if 'has_bifurcation' in assignments.columns:
        bifurc_rate = assignments['has_bifurcation'].mean()
        print(f"   Bifurcation rate: {bifurc_rate*100:.1f}%")
        print(f"   ({int(bifurc_rate * len(assignments))} narratives show phase transitions)")

        if 'n_transitions' in assignments.columns:
            avg_trans = assignments[assignments['has_bifurcation']]['n_transitions'].mean()
            print(f"   Average transitions per bifurcating narrative: {avg_trans:.2f}")

    # Distance analysis
    print(f"\n📏 DISTANCE METRICS")
    avg_dist = assignments['dist_to_global'].mean()
    max_dist = assignments['dist_to_global'].max()
    min_dist = assignments['dist_to_global'].min()

    print(f"   Average distance to global attractor: {avg_dist:.3f}")
    print(f"   Most typical narrative (min dist): ID {assignments.loc[assignments['dist_to_global'].idxmin(), 'narrative_id']}")
    print(f"   Most unusual narrative (max dist): ID {assignments.loc[assignments['dist_to_global'].idxmax(), 'narrative_id']}")

    # Basin distribution
    print(f"\n BASIN DISTRIBUTION")
    basin_counts = assignments['attractor_id'].value_counts().sort_index()
    for basin_id, count in basin_counts.items():
        pct = (count / len(assignments)) * 100
        depth = basin_stats.loc[basin_stats['attractor_id'] == basin_id, 'basin_depth'].values[0]
        stab = basin_stats.loc[basin_stats['attractor_id'] == basin_id, 'stability_ratio'].values[0]
        print(f"   Basin {basin_id}: {count:3d} narratives ({pct:5.1f}%) | "
              f"depth={depth:.3f} | stability={stab:.3f}")

    # Hierarchical structure
    if (dynamics_dir / "hierarchical_attractors.csv").exists():
        hierarchy = pd.read_csv(dynamics_dir / "hierarchical_attractors.csv")
        print(f"\n HIERARCHICAL STRUCTURE")
        levels = hierarchy['level'].value_counts().sort_index()
        for level, count in levels.items():
            print(f"   Level {level}: {count} sub-basins")

        # Find deepest nested structure
        max_level = hierarchy['level'].max()
        print(f"   Maximum nesting depth: {max_level}")

    # Trajectory insights
    if (dynamics_dir / "trajectories.jsonl").exists():
        trajs = pd.read_json(dynamics_dir / "trajectories.jsonl", lines=True)

        print(f"\n TRAJECTORY CHARACTERISTICS")
        avg_traj_len = trajs['trajectory_length'].mean()
        avg_curvature = trajs['mean_curvature'].mean()
        avg_transitions = trajs['basin_transitions'].mean()

        print(f"   Average trajectory length: {avg_traj_len:.3f}")
        print(f"   Average curvature: {avg_curvature:.4f}")
        print(f"   Average basin transitions: {avg_transitions:.2f}")

        # Find most interesting trajectories
        most_curved_id = trajs.loc[trajs['mean_curvature'].idxmax(), 'narrative_id']
        most_trans_id = trajs.loc[trajs['basin_transitions'].idxmax(), 'narrative_id']
        longest_id = trajs.loc[trajs['trajectory_length'].idxmax(), 'narrative_id']

        print(f"\n   Most curved trajectory: ID {most_curved_id}")
        print(f"   Most transitions: ID {most_trans_id}")
        print(f"   Longest trajectory: ID {longest_id}")

    print("\n" + "=" * 70)
    print(" INTERPRETATION TIPS:")
    print("   - High stability ratio (>1.2): Deep, well-defined thematic cluster")
    print("   - Many bifurcations: Narrative has dramatic topic shifts")
    print("   - High Lyapunov (>7): Unpredictable, nonlinear narrative flow")
    print("   - Large dist_to_global: Unusual/outlier narrative")
    print("=" * 70 + "\n")


def compare_narratives(dynamics_dir, id1, id2):
    """Compare two narratives in the attractor space"""
    dynamics_dir = Path(dynamics_dir)

    assignments = pd.read_csv(dynamics_dir / "attractor_assignments.csv")

    n1 = assignments[assignments['narrative_id'] == id1].iloc[0]
    n2 = assignments[assignments['narrative_id'] == id2].iloc[0]

    print(f"\n COMPARISON: Narrative {id1} vs {id2}")
    print(f"   Attractor basin: {n1['attractor_id']} vs {n2['attractor_id']}")
    print(f"   Distance to global: {n1['dist_to_global']:.3f} vs {n2['dist_to_global']:.3f}")

    if 'lyapunov_exponent' in n1:
        print(f"   Lyapunov exponent: {n1['lyapunov_exponent']:.3f} vs {n2['lyapunov_exponent']:.3f}")

    if 'n_transitions' in n1:
        print(f"   Basin transitions: {n1['n_transitions']} vs {n2['n_transitions']}")

    if n1['attractor_id'] == n2['attractor_id']:
        print(f"   ✓ Both in same attractor basin (similar narratives)")
    else:
        print(f"   ✗ Different attractor basins (distinct narratives)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python analyze_dynamics.py <dynamics_dir>")
        print("  python analyze_dynamics.py <dynamics_dir> compare <id1> <id2>")
        sys.exit(1)

    dynamics_dir = sys.argv[1]

    if len(sys.argv) > 2 and sys.argv[2] == "compare":
        id1 = int(sys.argv[3])
        id2 = int(sys.argv[4])
        compare_narratives(dynamics_dir, id1, id2)
    else:
        analyze_results(dynamics_dir)
