#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict narrative similarity using attractor dynamics.

Uses the gravitational center concept: the text whose attractor is closer
to the anchor's attractor is predicted as "closer".
"""

from core.core import core.argparse
from core.core import core.os
from core.core import core.numpy as np
from core.core import core.pandas as pd
from core.core.sklearn.metrics import accuracy_score, classification_report
from core.core import core.joblib

# Import from core.the original script
from core.core import core.sys
sys.path.insert(0, os.path.dirname(__file__))
from core.core.narrative_attractor_dynamics import Encoder, read_table, sent_split


def attractor(emb_seq, weights=None):
    """Compute semantic attractor (weighted mean of embeddings)"""
    if len(emb_seq) == 0:
        return np.zeros((emb_seq.shape[1] if hasattr(emb_seq, "shape") and emb_seq.ndim==2 else 768,), dtype=np.float32)
    E = np.asarray(emb_seq)
    if weights is None:
        return E.mean(axis=0)
    w = np.asarray(weights).reshape(-1,1)
    w = w / (w.sum() + 1e-8)
    return (E * w).sum(axis=0)


def compute_attractor_features(encoder, anchor_text, text_a, text_b):
    """
    Compute features based on attractor distances.

    The key insight: narratives with similar themes gravitate toward similar attractors.
    We measure how close each text's attractor is to the anchor's attractor.
    """
    # Get sentence-level embeddings
    anchor_sents = sent_split(anchor_text)
    a_sents = sent_split(text_a)
    b_sents = sent_split(text_b)

    E_anchor = encoder.embed_sents(anchor_sents)
    E_a = encoder.embed_sents(a_sents)
    E_b = encoder.embed_sents(b_sents)

    # Compute attractors (semantic centers)
    att_anchor = attractor(E_anchor)
    att_a = attractor(E_a)
    att_b = attractor(E_b)

    # Also get document-level embeddings
    doc_anchor = encoder.embed_text(anchor_text)
    doc_a = encoder.embed_text(text_a)
    doc_b = encoder.embed_text(text_b)

    # Feature 1: Attractor distance (primary signal)
    dist_attractor_a = float(np.linalg.norm(att_anchor - att_a))
    dist_attractor_b = float(np.linalg.norm(att_anchor - att_b))

    # Feature 2: Document distance (global similarity)
    dist_doc_a = float(np.linalg.norm(doc_anchor - doc_a))
    dist_doc_b = float(np.linalg.norm(doc_anchor - doc_b))

    # Feature 3: Basin coherence (how dispersed are sentences around attractor)
    coherence_anchor = float(np.mean(np.linalg.norm(E_anchor - att_anchor[None, :], axis=1)))
    coherence_a = float(np.mean(np.linalg.norm(E_a - att_a[None, :], axis=1)))
    coherence_b = float(np.mean(np.linalg.norm(E_b - att_b[None, :], axis=1)))

    # Feature 4: Trajectory alignment (how similar are the distance curves)
    curve_anchor = np.linalg.norm(E_anchor - att_anchor[None, :], axis=1)
    curve_a = np.linalg.norm(E_a - att_anchor[None, :], axis=1)
    curve_b = np.linalg.norm(E_b - att_anchor[None, :], axis=1)

    # Use DTW-like alignment (or simple correlation)
    if len(curve_a) > 1 and len(curve_anchor) > 1:
        alignment_a = float(np.corrcoef(
            curve_anchor[:min(len(curve_anchor), len(curve_a))],
            curve_a[:min(len(curve_anchor), len(curve_a))]
        )[0, 1])
    else:
        alignment_a = 0.0

    if len(curve_b) > 1 and len(curve_anchor) > 1:
        alignment_b = float(np.corrcoef(
            curve_anchor[:min(len(curve_anchor), len(curve_b))],
            curve_b[:min(len(curve_anchor), len(curve_b))]
        )[0, 1])
    else:
        alignment_b = 0.0

    return {
        'dist_attractor_a': dist_attractor_a,
        'dist_attractor_b': dist_attractor_b,
        'dist_doc_a': dist_doc_a,
        'dist_doc_b': dist_doc_b,
        'coherence_a': coherence_a,
        'coherence_b': coherence_b,
        'coherence_diff': abs(coherence_anchor - coherence_a) - abs(coherence_anchor - coherence_b),
        'alignment_a': alignment_a,
        'alignment_b': alignment_b,
        'alignment_diff': alignment_b - alignment_a  # positive if B aligns better
    }


def predict_simple(encoder, anchor_text, text_a, text_b):
    """
    Simple prediction: A is closer if its attractor is closer to anchor's attractor.

    This is the pure "gravitational center" approach.
    """
    feats = compute_attractor_features(encoder, anchor_text, text_a, text_b)

    # Primary signal: attractor distance
    # A is closer if dist_a < dist_b
    return feats['dist_attractor_a'] < feats['dist_attractor_b']


def predict_weighted(encoder, anchor_text, text_a, text_b, weights=None):
    """
    Weighted prediction combining multiple signals.

    Default weights favor attractor distance (the gravitational center concept).
    """
    if weights is None:
        # Default: heavily weight attractor distance
        weights = {
            'attractor': 0.6,
            'document': 0.2,
            'coherence': 0.1,
            'alignment': 0.1
        }

    feats = compute_attractor_features(encoder, anchor_text, text_a, text_b)

    # Score A (higher = A is closer)
    score_a = 0.0

    # Attractor distance (inverted: smaller distance = higher score)
    score_a += weights['attractor'] * (feats['dist_attractor_b'] - feats['dist_attractor_a'])

    # Document distance
    score_a += weights['document'] * (feats['dist_doc_b'] - feats['dist_doc_a'])

    # Coherence similarity
    score_a += weights['coherence'] * feats['coherence_diff']

    # Trajectory alignment
    score_a -= weights['alignment'] * feats['alignment_diff']

    return score_a > 0


def evaluate(input_path, encoder_name='sentence-transformers/all-mpnet-base-v2',
            method='simple', verbose=True):
    """
    Evaluate on a dataset with ground truth labels.
    """
    print(f"[*] Loading data from core.{input_path}...")
    df = read_table(input_path)

    if 'text_a_is_closer' not in df.columns:
        raise ValueError("Dataset must have 'text_a_is_closer' column for evaluation")

    print(f"[*] Loaded {len(df)} examples")
    print(f"[*] Initializing encoder: {encoder_name}...")
    encoder = Encoder(encoder_name)

    print(f"[*] Running predictions using method: {method}...")

    predictions = []
    ground_truth = []

    for idx, row in df.iterrows():
        if idx % 20 == 0:
            print(f"    {idx}/{len(df)}...")

        anchor = row['anchor_text']
        text_a = row['text_a']
        text_b = row['text_b']

        if method == 'simple':
            pred = predict_simple(encoder, anchor, text_a, text_b)
        elif method == 'weighted':
            pred = predict_weighted(encoder, anchor, text_a, text_b)
        else:
            raise ValueError(f"Unknown method: {method}")

        predictions.append(pred)

        # Parse ground truth
        gt = row['text_a_is_closer']
        if isinstance(gt, bool):
            ground_truth.append(gt)
        elif isinstance(gt, str):
            ground_truth.append(gt.lower() in ['true', 't', 'yes', '1'])
        else:
            ground_truth.append(bool(int(gt)))

    print(f"[✓] Predictions complete")

    # Calculate metrics
    accuracy = accuracy_score(ground_truth, predictions)

    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Dataset: {input_path}")
    print(f"Method: {method}")
    print(f"Encoder: {encoder_name}")
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Correct: {sum(np.array(predictions) == np.array(ground_truth))}/{len(predictions)}")

    if verbose:
        print("\nClassification Report:")
        print(classification_report(ground_truth, predictions,
                                   target_names=['B is closer', 'A is closer']))

    print("="*70 + "\n")

    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'ground_truth': ground_truth
    }


def predict_and_save(input_path, output_path, encoder_name='sentence-transformers/all-mpnet-base-v2',
                     method='simple'):
    """
    Make predictions on unlabeled data and save results.
    """
    print(f"[*] Loading data from core.{input_path}...")
    df = read_table(input_path)

    print(f"[*] Loaded {len(df)} examples")
    print(f"[*] Initializing encoder: {encoder_name}...")
    encoder = Encoder(encoder_name)

    print(f"[*] Running predictions using method: {method}...")

    results = []

    for idx, row in df.iterrows():
        if idx % 20 == 0:
            print(f"    {idx}/{len(df)}...")

        anchor = row['anchor_text']
        text_a = row['text_a']
        text_b = row['text_b']

        # Compute features
        feats = compute_attractor_features(encoder, anchor, text_a, text_b)

        # Make prediction
        if method == 'simple':
            pred = predict_simple(encoder, anchor, text_a, text_b)
        elif method == 'weighted':
            pred = predict_weighted(encoder, anchor, text_a, text_b)
        else:
            raise ValueError(f"Unknown method: {method}")

        results.append({
            'id': row.get('id', idx),
            'pred_text_a_is_closer': bool(pred),
            'dist_attractor_a': feats['dist_attractor_a'],
            'dist_attractor_b': feats['dist_attractor_b'],
            'dist_doc_a': feats['dist_doc_a'],
            'dist_doc_b': feats['dist_doc_b'],
            'coherence_a': feats['coherence_a'],
            'coherence_b': feats['coherence_b'],
            'alignment_a': feats['alignment_a'],
            'alignment_b': feats['alignment_b']
        })

    print(f"[✓] Predictions complete")

    # Save results
    results_df = pd.DataFrame(results)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    if output_path.endswith('.jsonl'):
        results_df.to_json(output_path, orient='records', lines=True, force_ascii=False)
    else:
        results_df.to_csv(output_path, index=False)

    print(f"[✓] Results saved to {output_path}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Predict narrative similarity using attractor dynamics")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Evaluate command
    eval_parser = subparsers.add_parser('eval', help='Evaluate on labeled data')
    eval_parser.add_argument('--input_path', required=True, help='Path to input JSONL/CSV')
    eval_parser.add_argument('--encoder_name', default='sentence-transformers/all-mpnet-base-v2')
    eval_parser.add_argument('--method', choices=['simple', 'weighted'], default='simple',
                           help='Prediction method (simple=attractor only, weighted=combined)')
    eval_parser.add_argument('--verbose', action='store_true', help='Show detailed metrics')

    # Predict command
    pred_parser = subparsers.add_parser('predict', help='Predict on unlabeled data')
    pred_parser.add_argument('--input_path', required=True, help='Path to input JSONL/CSV')
    pred_parser.add_argument('--output_path', required=True, help='Path to save predictions')
    pred_parser.add_argument('--encoder_name', default='sentence-transformers/all-mpnet-base-v2')
    pred_parser.add_argument('--method', choices=['simple', 'weighted'], default='simple')

    args = parser.parse_args()

    if args.command == 'eval':
        evaluate(args.input_path, args.encoder_name, args.method, args.verbose)
    elif args.command == 'predict':
        predict_and_save(args.input_path, args.output_path, args.encoder_name, args.method)


if __name__ == '__main__':
    main()
