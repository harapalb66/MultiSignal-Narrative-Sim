#!/usr/bin/env python3
"""
Visualize narrative trajectories in 3D semantic space.

For selected examples:
1. Compute chunk embeddings (trajectories in 768D)
2. Compute centroids (attractors)
3. Reduce to 3D using UMAP
4. Visualize as connected paths
5. Compute trajectory similarity metrics
"""
from core.core import numpy as np
from core.core import pandas as pd
from core.core import matplotlib.pyplot as plt
from core.mpl_toolkits.mplot3d import Axes3D
from core.narrative_attractor_dynamics import Encoder, sent_split
from core.sklearn.decomposition import PCA
from core.scipy.spatial.distance import directed_hausdorff
from core.core import json


def create_chunks(text, chunk_size=3):
    """Split text into overlapping chunks of sentences."""
    sents = sent_split(text)
    if len(sents) <= chunk_size:
        return [text]

    chunks = []
    stride = max(1, chunk_size // 2)
    for i in range(0, len(sents) - chunk_size + 1, stride):
        chunk = ' '.join(sents[i:i + chunk_size])
        chunks.append(chunk)

    if len(sents) > chunk_size:
        last_chunk = ' '.join(sents[-chunk_size:])
        if last_chunk != chunks[-1]:
            chunks.append(last_chunk)

    return chunks


def frechet_distance(P, Q):
    """
    Compute Fréchet distance between two trajectories.
    Simplified discrete version.
    """
    n, m = len(P), len(Q)

    # Dynamic programming table
    ca = np.full((n, m), -1.0)

    def c(i, j):
        if ca[i, j] > -1:
            return ca[i, j]

        d = np.linalg.norm(P[i] - Q[j])

        if i == 0 and j == 0:
            ca[i, j] = d
        elif i > 0 and j == 0:
            ca[i, j] = max(c(i-1, 0), d)
        elif i == 0 and j > 0:
            ca[i, j] = max(c(0, j-1), d)
        elif i > 0 and j > 0:
            ca[i, j] = max(min(c(i-1, j), c(i-1, j-1), c(i, j-1)), d)
        else:
            ca[i, j] = float('inf')

        return ca[i, j]

    return c(n-1, m-1)


def dtw_distance(P, Q):
    """Dynamic Time Warping distance between trajectories."""
    n, m = len(P), len(Q)
    dtw_matrix = np.zeros((n + 1, m + 1))

    for i in range(1, n + 1):
        dtw_matrix[i, 0] = float('inf')
    for j in range(1, m + 1):
        dtw_matrix[0, j] = float('inf')

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(P[i-1] - Q[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],    # insertion
                dtw_matrix[i, j-1],    # deletion
                dtw_matrix[i-1, j-1]   # match
            )

    return dtw_matrix[n, m]


def compute_trajectory_data(encoder, example, chunk_size=3):
    """
    Compute chunk trajectories and centroids for one example.

    Returns:
        data dict with embeddings, chunks, centroids
    """
    # Create chunks
    anchor_chunks = create_chunks(example['anchor_text'], chunk_size)
    a_chunks = create_chunks(example['text_a'], chunk_size)
    b_chunks = create_chunks(example['text_b'], chunk_size)

    # Embed chunks
    anchor_embeds = encoder.embed_sents(anchor_chunks)
    a_embeds = encoder.embed_sents(a_chunks)
    b_embeds = encoder.embed_sents(b_chunks)

    # Compute centroids
    anchor_centroid = anchor_embeds.mean(axis=0)
    a_centroid = a_embeds.mean(axis=0)
    b_centroid = b_embeds.mean(axis=0)

    # Compute distances
    dist_a = float(np.linalg.norm(anchor_centroid - a_centroid))
    dist_b = float(np.linalg.norm(anchor_centroid - b_centroid))

    return {
        'anchor_embeds': anchor_embeds,
        'a_embeds': a_embeds,
        'b_embeds': b_embeds,
        'anchor_centroid': anchor_centroid,
        'a_centroid': a_centroid,
        'b_centroid': b_centroid,
        'anchor_chunks': anchor_chunks,
        'a_chunks': a_chunks,
        'b_chunks': b_chunks,
        'dist_a': dist_a,
        'dist_b': dist_b,
        'center_predicts_a': dist_a < dist_b,
    }


def compute_trajectory_similarities(traj_data):
    """
    Compute trajectory-based similarity metrics.

    Returns dict with various trajectory comparison metrics.
    """
    anchor_traj = traj_data['anchor_embeds']
    a_traj = traj_data['a_embeds']
    b_traj = traj_data['b_embeds']

    # Fréchet distance
    frechet_a = frechet_distance(anchor_traj, a_traj)
    frechet_b = frechet_distance(anchor_traj, b_traj)

    # DTW distance
    dtw_a = dtw_distance(anchor_traj, a_traj)
    dtw_b = dtw_distance(anchor_traj, b_traj)

    # Hausdorff distance (shape similarity)
    hausdorff_a = max(directed_hausdorff(anchor_traj, a_traj)[0],
                      directed_hausdorff(a_traj, anchor_traj)[0])
    hausdorff_b = max(directed_hausdorff(anchor_traj, b_traj)[0],
                      directed_hausdorff(b_traj, anchor_traj)[0])

    return {
        'frechet_a': float(frechet_a),
        'frechet_b': float(frechet_b),
        'frechet_predicts_a': frechet_a < frechet_b,
        'dtw_a': float(dtw_a),
        'dtw_b': float(dtw_b),
        'dtw_predicts_a': dtw_a < dtw_b,
        'hausdorff_a': float(hausdorff_a),
        'hausdorff_b': float(hausdorff_b),
        'hausdorff_predicts_a': hausdorff_a < hausdorff_b,
    }


def visualize_example_3d(traj_data, example_id, case_name, ground_truth, output_path):
    """
    Visualize trajectories in 3D using PCA.
    """
    # Combine all embeddings for PCA
    all_embeds = np.vstack([
        traj_data['anchor_embeds'],
        traj_data['a_embeds'],
        traj_data['b_embeds'],
        traj_data['anchor_centroid'].reshape(1, -1),
        traj_data['a_centroid'].reshape(1, -1),
        traj_data['b_centroid'].reshape(1, -1),
    ])

    # PCA to 3D
    pca = PCA(n_components=3)
    all_3d = pca.fit_transform(all_embeds)

    # Split back
    n_anchor = len(traj_data['anchor_embeds'])
    n_a = len(traj_data['a_embeds'])
    n_b = len(traj_data['b_embeds'])

    anchor_3d = all_3d[:n_anchor]
    a_3d = all_3d[n_anchor:n_anchor+n_a]
    b_3d = all_3d[n_anchor+n_a:n_anchor+n_a+n_b]
    anchor_cent_3d = all_3d[-3]
    a_cent_3d = all_3d[-2]
    b_cent_3d = all_3d[-1]

    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectories
    ax.plot(anchor_3d[:, 0], anchor_3d[:, 1], anchor_3d[:, 2],
            'o-', color='black', linewidth=2, markersize=6, label='Anchor trajectory', alpha=0.7)
    ax.plot(a_3d[:, 0], a_3d[:, 1], a_3d[:, 2],
            'o-', color='blue', linewidth=2, markersize=6, label='Text A trajectory', alpha=0.7)
    ax.plot(b_3d[:, 0], b_3d[:, 1], b_3d[:, 2],
            'o-', color='red', linewidth=2, markersize=6, label='Text B trajectory', alpha=0.7)

    # Plot centroids
    ax.scatter([anchor_cent_3d[0]], [anchor_cent_3d[1]], [anchor_cent_3d[2]],
              color='black', s=300, marker='*', edgecolors='gold', linewidths=2,
              label='Anchor centroid', zorder=10)
    ax.scatter([a_cent_3d[0]], [a_cent_3d[1]], [a_cent_3d[2]],
              color='blue', s=300, marker='*', edgecolors='gold', linewidths=2,
              label='Text A centroid', zorder=10)
    ax.scatter([b_cent_3d[0]], [b_cent_3d[1]], [b_cent_3d[2]],
              color='red', s=300, marker='*', edgecolors='gold', linewidths=2,
              label='Text B centroid', zorder=10)

    # Draw lines from core.anchor centroid to A and B centroids
    ax.plot([anchor_cent_3d[0], a_cent_3d[0]],
           [anchor_cent_3d[1], a_cent_3d[1]],
           [anchor_cent_3d[2], a_cent_3d[2]],
           'b--', linewidth=1.5, alpha=0.5, label=f'Center dist A: {traj_data["dist_a"]:.3f}')
    ax.plot([anchor_cent_3d[0], b_cent_3d[0]],
           [anchor_cent_3d[1], b_cent_3d[1]],
           [anchor_cent_3d[2], b_cent_3d[2]],
           'r--', linewidth=1.5, alpha=0.5, label=f'Center dist B: {traj_data["dist_b"]:.3f}')

    # Labels and title
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')

    title = f"Example {example_id}: {case_name}\n"
    title += f"Ground Truth: {'A' if ground_truth else 'B'} is closer | "
    title += f"Center predicts: {'A' if traj_data['center_predicts_a'] else 'B'}"
    ax.set_title(title, fontsize=12, fontweight='bold')

    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")

    return pca.explained_variance_ratio_


def main():
    print("=" * 80)
    print("TRAJECTORY VISUALIZATION AND PATH COMPARISON")
    print("=" * 80)

    # Load selected IDs
    with open('selected_example_ids.txt', 'r') as f:
        selected_ids = list(map(int, f.read().strip().split(',')))

    print(f"\nSelected example IDs: {selected_ids}")

    # Load dev set
    print("\n[1/5] Loading dev set data...")
    dev_df = pd.read_json('SemEval2026-Task_4-dev-v1/dev_track_a.jsonl', lines=True)
    pred_df = pd.read_json('chunk_enhanced_dev_predictions_full.jsonl', lines=True)

    # Initialize encoder
    print("[2/5] Initializing encoder...")
    encoder = Encoder('sentence-transformers/all-mpnet-base-v2')

    # Process each example
    print("[3/5] Computing trajectories...")
    results = []

    case_names = [
        "Agreement + Correct",
        "Chunks Override + Correct",
        "Chunks Override + Wrong"
    ]

    for idx, (example_id, case_name) in enumerate(zip(selected_ids, case_names)):
        print(f"\n  Processing Example {example_id}: {case_name}...")

        example = dev_df.iloc[example_id]
        pred = pred_df[pred_df['id'] == example_id].iloc[0]

        # Compute trajectory data
        traj_data = compute_trajectory_data(encoder, example, chunk_size=3)

        # Compute trajectory similarities
        traj_sim = compute_trajectory_similarities(traj_data)

        # Combine results
        result = {
            'id': example_id,
            'case_name': case_name,
            'ground_truth_a_closer': example['text_a_is_closer'],
            'center_predicts_a': traj_data['center_predicts_a'],
            'chunks_predicts_a': pred['text_a_is_closer'],
            'decision_type': pred['decision_type'],
            'chunk_confidence': pred['confidence'],
            'n_anchor_chunks': len(traj_data['anchor_chunks']),
            'n_a_chunks': len(traj_data['a_chunks']),
            'n_b_chunks': len(traj_data['b_chunks']),
            **traj_sim,
        }
        results.append(result)

        # Print summary
        print(f"    Ground truth: {'A' if result['ground_truth_a_closer'] else 'B'} is closer")
        print(f"    Center: {'A' if result['center_predicts_a'] else 'B'} (dist_a={traj_data['dist_a']:.3f}, dist_b={traj_data['dist_b']:.3f})")
        print(f"    Chunks: {'A' if result['chunks_predicts_a'] else 'B'} (conf={result['chunk_confidence']:.3f}, {result['decision_type']})")
        print(f"    Fréchet: {'A' if result['frechet_predicts_a'] else 'B'} (a={result['frechet_a']:.3f}, b={result['frechet_b']:.3f})")
        print(f"    DTW: {'A' if result['dtw_predicts_a'] else 'B'} (a={result['dtw_a']:.3f}, b={result['dtw_b']:.3f})")
        print(f"    Hausdorff: {'A' if result['hausdorff_predicts_a'] else 'B'} (a={result['hausdorff_a']:.3f}, b={result['hausdorff_b']:.3f})")

    # Visualize
    print("\n[4/5] Creating 3D visualizations...")
    for idx, (example_id, case_name) in enumerate(zip(selected_ids, case_names)):
        example = dev_df.iloc[example_id]
        traj_data = compute_trajectory_data(encoder, example, chunk_size=3)
        output_path = f"trajectory_viz_example_{example_id}.png"
        visualize_example_3d(traj_data, example_id, case_name,
                           example['text_a_is_closer'], output_path)

    # Save results
    print("\n[5/5] Saving results...")
    results_df = pd.DataFrame(results)
    results_df.to_json('trajectory_analysis_results.jsonl', orient='records',
                      lines=True, force_ascii=False)
    print("  Saved: trajectory_analysis_results.jsonl")

    # Print summary table
    print("\n" + "=" * 80)
    print("TRAJECTORY PATH COMPARISON SUMMARY")
    print("=" * 80)
    print("\nAccuracy by method:")

    for method in ['center', 'chunks', 'frechet', 'dtw', 'hausdorff']:
        pred_col = f'{method}_predicts_a'
        if pred_col in results_df.columns:
            correct = (results_df[pred_col] == results_df['ground_truth_a_closer']).sum()
            acc = correct / len(results_df) * 100
            print(f"  {method.capitalize():12s}: {acc:.1f}% ({correct}/{len(results_df)})")

    print("\n" + "=" * 80)
    print("VISUALIZATION FILES CREATED:")
    print("=" * 80)
    for example_id in selected_ids:
        print(f"  - trajectory_viz_example_{example_id}.png")
    print(f"  - trajectory_analysis_results.jsonl")
    print("=" * 80)


if __name__ == '__main__':
    main()
