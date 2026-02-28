#!/usr/bin/env python3
"""
Ensemble prediction combining:
1. Zero-shot attractor (58% accuracy)
2. Enhanced hybrid model (potentially 65%+)

Uses voting and confidence-based weighting.
"""
from core.core import core.argparse
from core.core import core.numpy as np
from core.core import core.pandas as pd
from core.core import core.joblib
from core.core.narrative_attractor_dynamics import Encoder, read_table
from core.core.predict_with_attractors import compute_attractor_features
from core.core.narative_dynamic import pair_features as original_pair_features
from core.core.enhanced_features import compute_enhanced_features
from core.core.basin_features import BasinFeatureExtractor


def zero_shot_predict(encoder, anchor_text, text_a, text_b):
    """Simple zero-shot attractor prediction."""
    att_feats = compute_attractor_features(encoder, anchor_text, text_a, text_b)
    dist_a = att_feats['dist_attractor_a']
    dist_b = att_feats['dist_attractor_b']
    return dist_a < dist_b, abs(dist_b - dist_a)


def enhanced_hybrid_predict(model_bundle, encoder, basin_extractor, anchor_text, text_a, text_b):
    """Enhanced hybrid model prediction with all features."""
    # Extract all features
    att_feats = compute_attractor_features(encoder, anchor_text, text_a, text_b)
    orig_fa = original_pair_features(encoder, anchor_text, text_a)
    orig_fb = original_pair_features(encoder, anchor_text, text_b)
    enhanced_feats = compute_enhanced_features(encoder, anchor_text, text_a, text_b)
    basin_feats = basin_extractor.compute_basin_features(encoder, anchor_text, text_a, text_b)

    features = np.array([
        att_feats['dist_attractor_b'] - att_feats['dist_attractor_a'],
        att_feats['dist_doc_b'] - att_feats['dist_doc_a'],
        att_feats['coherence_diff'],
        att_feats['alignment_diff'],
        orig_fa[0], orig_fa[1], orig_fa[2], orig_fa[3], orig_fa[4], orig_fa[5],
        orig_fb[0], orig_fb[1], orig_fb[2], orig_fb[3], orig_fb[4], orig_fb[5],
        enhanced_feats['curvature_diff'],
        enhanced_feats['transitions_diff'],
        enhanced_feats['traj_length_diff'],
        enhanced_feats['start_end_diff'],
        enhanced_feats['efficiency_diff'],
        enhanced_feats['hierarchical_var_diff'],
        enhanced_feats['scale_consistency_diff'],
        enhanced_feats['position_weight_diff'],
        enhanced_feats['recency_weight_diff'],
        enhanced_feats['curvature_sim_a'],
        enhanced_feats['curvature_sim_b'],
        enhanced_feats['transitions_sim_a'],
        enhanced_feats['transitions_sim_b'],
        basin_feats['basin_depth_diff'],
        basin_feats['stability_diff'],
        basin_feats['dispersion_diff'],
        basin_feats['basin_depth_sim_diff'],
        basin_feats['stability_sim_diff'],
    ], dtype=float)

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).reshape(1, -1)

    pipeline = model_bundle['pipeline']
    pred = bool(pipeline.predict(features)[0])
    prob = pipeline.predict_proba(features)[0][1]

    return pred, prob


def ensemble_predict(input_path, hybrid_model_path, output_path, strategy='weighted'):
    """
    Ensemble prediction combining zero-shot and hybrid models.

    Strategies:
    - 'voting': Simple majority vote
    - 'weighted': Confidence-weighted combination
    - 'hybrid_priority': Use hybrid if confident, else zero-shot
    """
    print("=" * 80)
    print("ENSEMBLE PREDICTION")
    print("=" * 80)

    # Load model
    print(f"\n[1/5] Loading hybrid model from core.{hybrid_model_path}...")
    model_bundle = joblib.load(hybrid_model_path)
    print(f"      CV accuracy: {model_bundle['cv_accuracy']:.4f}")

    # Initialize
    print(f"\n[2/5] Initializing encoders...")
    encoder = Encoder(model_bundle['encoder_name'])
    basin_extractor = BasinFeatureExtractor()

    # Load data
    print(f"\n[3/5] Loading data from core.{input_path}...")
    df = read_table(input_path)
    print(f"      Loaded {len(df)} examples")

    # Make predictions
    print(f"\n[4/5] Making ensemble predictions (strategy={strategy})...")
    results = []

    for idx, row in df.iterrows():
        if idx % 20 == 0:
            print(f"      {idx}/{len(df)}...")

        # Get predictions from core.both models
        zs_pred, zs_conf = zero_shot_predict(encoder, row['anchor_text'], row['text_a'], row['text_b'])
        hybrid_pred, hybrid_conf = enhanced_hybrid_predict(
            model_bundle, encoder, basin_extractor,
            row['anchor_text'], row['text_a'], row['text_b']
        )

        # Ensemble strategy
        if strategy == 'voting':
            # Simple majority vote
            final_pred = zs_pred if zs_pred == hybrid_pred else (hybrid_conf > 0.5)
            final_conf = (zs_conf + hybrid_conf) / 2

        elif strategy == 'weighted':
            # Confidence-weighted average
            # Normalize confidences to [0, 1]
            zs_conf_norm = min(zs_conf, 1.0)
            hybrid_conf_norm = hybrid_conf

            # Weight predictions by confidence
            if zs_pred and hybrid_pred:
                final_pred = True
                final_conf = (zs_conf_norm + hybrid_conf_norm) / 2
            elif not zs_pred and not hybrid_pred:
                final_pred = False
                final_conf = (zs_conf_norm + (1 - hybrid_conf_norm)) / 2
            else:
                # Disagree: use weighted vote
                zs_weight = zs_conf_norm
                hybrid_weight = max(abs(hybrid_conf_norm - 0.5), 0.1)
                total = zs_weight + hybrid_weight

                if zs_pred:
                    score = zs_weight / total
                else:
                    score = hybrid_weight / total

                final_pred = score > 0.5
                final_conf = max(score, 1 - score)

        else:  # 'hybrid_priority'
            # Use hybrid if confident (>60%), else zero-shot
            if abs(hybrid_conf - 0.5) > 0.1:
                final_pred = hybrid_pred
                final_conf = hybrid_conf
            else:
                final_pred = zs_pred
                final_conf = 0.5 + zs_conf * 0.3

        results.append({
            'id': row.get('id', idx),
            'text_a_is_closer': final_pred,
            'prob_a': final_conf,
            'prob_b': 1 - final_conf,
            'zs_pred': zs_pred,
            'hybrid_pred': hybrid_pred,
            'zs_conf': zs_conf,
            'hybrid_conf': hybrid_conf,
        })

    # Save
    print(f"\n[5/5] Saving predictions to {output_path}...")
    results_df = pd.DataFrame(results)

    # Submission format (only id and prediction)
    submission = results_df[['id', 'text_a_is_closer']]
    submission.to_json(output_path, orient='records', lines=True, force_ascii=False)

    # Full version with details
    full_path = output_path.replace('.jsonl', '_full.jsonl')
    results_df.to_json(full_path, orient='records', lines=True, force_ascii=False)

    print(f"\n{'=' * 80}")
    print(f"✓ ENSEMBLE PREDICTION COMPLETE!")
    print(f"{'=' * 80}")
    print(f"  Total predictions: {len(results)}")
    print(f"  A is closer: {sum(r['text_a_is_closer'] for r in results)} ({sum(r['text_a_is_closer'] for r in results)/len(results)*100:.1f}%)")
    print(f"  Agreement rate: {sum(1 for r in results if r['zs_pred'] == r['hybrid_pred'])/len(results)*100:.1f}%")
    print(f"  Submission saved to: {output_path}")
    print(f"  Full details saved to: {full_path}")
    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description='Ensemble prediction')
    parser.add_argument('--input_path', required=True, help='Path to input data')
    parser.add_argument('--hybrid_model_path', required=True, help='Path to hybrid model')
    parser.add_argument('--output_path', required=True, help='Path to save predictions')
    parser.add_argument('--strategy', default='weighted', choices=['voting', 'weighted', 'hybrid_priority'],
                       help='Ensemble strategy')

    args = parser.parse_args()

    ensemble_predict(args.input_path, args.hybrid_model_path, args.output_path, args.strategy)


if __name__ == '__main__':
    main()
