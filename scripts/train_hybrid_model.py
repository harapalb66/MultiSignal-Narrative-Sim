#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Model: Combines attractor dynamics features with original narative_dynamic.py features.

This trains a supervised model on the combined feature set.
"""

from core.core import core.argparse
from core.core import core.os
from core.core import core.warnings
warnings.filterwarnings("ignore")

from core.core import core.numpy as np
from core.core import core.pandas as pd
from core.core.sklearn.linear_model import LogisticRegression
from core.core.sklearn.preprocessing import StandardScaler
from core.core.sklearn.pipeline import Pipeline
from core.core.sklearn.model_selection import KFold, train_test_split
from core.core.sklearn.metrics import accuracy_score, classification_report
from core.core import core.joblib

# Import from core.both scripts
from core.core import core.sys
sys.path.insert(0, os.path.dirname(__file__))
from core.core.narrative_attractor_dynamics import Encoder, read_table, sent_split
from core.core.predict_with_attractors import compute_attractor_features
from core.core.narative_dynamic import pair_features as original_pair_features


def extract_hybrid_features(encoder, anchor_text, text_a, text_b):
    """
    Combine attractor features + original features into one feature vector.

    Returns:
        Feature vector with 14 dimensions:
        - 4 attractor features (distances, coherence, alignment)
        - 10 original features (2x: cos_global, cos_att, neg_basin, dtw, order_k, ner)
    """
    # Attractor features (4 features)
    att_feats = compute_attractor_features(encoder, anchor_text, text_a, text_b)

    # Original features from core.narative_dynamic.py (6 features each for A and B)
    orig_fa = original_pair_features(encoder, anchor_text, text_a)
    orig_fb = original_pair_features(encoder, anchor_text, text_b)

    # Combine into single feature vector
    features = np.array([
        # Attractor differences (A vs B)
        att_feats['dist_attractor_b'] - att_feats['dist_attractor_a'],  # positive = A closer
        att_feats['dist_doc_b'] - att_feats['dist_doc_a'],
        att_feats['coherence_diff'],
        att_feats['alignment_diff'],

        # Original features for A
        orig_fa[0],  # cos_global
        orig_fa[1],  # cos_att
        orig_fa[2],  # neg_basin
        orig_fa[3],  # dtw_sim
        orig_fa[4],  # order_k
        orig_fa[5],  # ner_overlap

        # Original features for B
        orig_fb[0],  # cos_global
        orig_fb[1],  # cos_att
        orig_fb[2],  # neg_basin
        orig_fb[3],  # dtw_sim
        orig_fb[4],  # order_k
        orig_fb[5],  # ner_overlap
    ], dtype=float)

    return features


def train_hybrid_model(train_path, output_path, encoder_name='sentence-transformers/all-mpnet-base-v2',
                       test_size=0.2, cv_folds=5, random_state=42):
    """
    Train a hybrid model combining attractor + original features.
    """
    print("="*70)
    print("HYBRID MODEL TRAINING")
    print("="*70)
    print(f"Training data: {train_path}")
    print(f"Encoder: {encoder_name}")
    print()

    # Load data
    print("[*] Loading training data...")
    df = read_table(train_path)

    if 'text_a_is_closer' not in df.columns:
        raise ValueError("Training data must have 'text_a_is_closer' column")

    print(f"[*] Loaded {len(df)} examples")

    # Initialize encoder
    print(f"[*] Initializing encoder...")
    encoder = Encoder(encoder_name)

    # Extract features
    print("[*] Extracting hybrid features...")
    X = []
    y = []

    for idx, row in df.iterrows():
        if idx % 20 == 0:
            print(f"    {idx}/{len(df)}...")

        anchor = row['anchor_text']
        text_a = row['text_a']
        text_b = row['text_b']

        # Extract hybrid features
        features = extract_hybrid_features(encoder, anchor, text_a, text_b)
        X.append(features)

        # Parse label
        label = row['text_a_is_closer']
        if isinstance(label, bool):
            y.append(int(label))
        elif isinstance(label, str):
            y.append(int(label.lower() in ['true', 't', 'yes', '1']))
        else:
            y.append(int(label))

    X = np.vstack(X)
    y = np.array(y)

    print(f"[✓] Extracted {len(X)} feature vectors (dim={X.shape[1]})")
    print(f"    Positive labels: {np.sum(y)}/{len(y)} ({np.mean(y)*100:.1f}%)")

    # Split train/test
    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"\n[*] Split: {len(X_train)} train, {len(X_test)} test")
    else:
        X_train, y_train = X, y
        X_test, y_test = None, None
        print(f"\n[*] Using all {len(X_train)} examples for training (no test split)")

    # Build pipeline
    print(f"\n[*] Training model with {cv_folds}-fold cross-validation...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=random_state,
            solver='lbfgs'
        ))
    ])

    # Cross-validation on training set
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        pipeline.fit(X_train[train_idx], y_train[train_idx])
        val_pred = pipeline.predict(X_train[val_idx])
        val_acc = accuracy_score(y_train[val_idx], val_pred)
        cv_scores.append(val_acc)
        print(f"    Fold {fold+1}/{cv_folds}: {val_acc:.4f}")

    mean_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)
    print(f"\n[✓] Cross-validation accuracy: {mean_cv:.4f} ± {std_cv:.4f}")

    # Train final model on all training data
    print(f"\n[*] Training final model on full training set...")
    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"[✓] Training accuracy: {train_acc:.4f}")

    # Evaluate on test set if available
    if X_test is not None:
        print(f"\n[*] Evaluating on test set...")
        test_pred = pipeline.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)

        print(f"[✓] Test accuracy: {test_acc:.4f}")
        print(f"\nTest Set Classification Report:")
        print(classification_report(y_test, test_pred,
                                   target_names=['B is closer', 'A is closer']))
    else:
        test_acc = None

    # Feature importance (from core.logistic regression coefficients)
    print(f"\n[*] Feature importance (logistic regression coefficients):")
    feature_names = [
        'attractor_dist_diff',
        'doc_dist_diff',
        'coherence_diff',
        'alignment_diff',
        'cos_global_A',
        'cos_att_A',
        'neg_basin_A',
        'dtw_sim_A',
        'order_k_A',
        'ner_overlap_A',
        'cos_global_B',
        'cos_att_B',
        'neg_basin_B',
        'dtw_sim_B',
        'order_k_B',
        'ner_overlap_B'
    ]

    coefs = pipeline.named_steps['classifier'].coef_[0]
    importance = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)

    print(f"\n    Top 10 most important features:")
    for i, (name, coef) in enumerate(importance[:10], 1):
        print(f"    {i:2d}. {name:25s} : {coef:+.4f}")

    # Save model
    print(f"\n[*] Saving model to {output_path}...")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    model_bundle = {
        'pipeline': pipeline,
        'encoder_name': encoder_name,
        'feature_names': feature_names,
        'cv_accuracy': mean_cv,
        'cv_std': std_cv,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'n_features': X.shape[1],
        'random_state': random_state
    }

    joblib.dump(model_bundle, output_path)
    print(f"[✓] Model saved successfully")

    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Model type: Hybrid (Attractor + Original features)")
    print(f"Features: {X.shape[1]} (4 attractor + 10 original)")
    print(f"Training examples: {len(X_train)}")
    if X_test is not None:
        print(f"Test examples: {len(X_test)}")
    print(f"\nCross-validation accuracy: {mean_cv:.4f} ± {std_cv:.4f}")
    print(f"Training accuracy: {train_acc:.4f}")
    if test_acc is not None:
        print(f"Test accuracy: {test_acc:.4f}")
    print(f"\nModel saved to: {output_path}")
    print("="*70 + "\n")

    return model_bundle


def predict_with_hybrid(input_path, model_path, output_path):
    """
    Make predictions using trained hybrid model.
    """
    print(f"[*] Loading model from core.{model_path}...")
    bundle = joblib.load(model_path)

    pipeline = bundle['pipeline']
    encoder_name = bundle['encoder_name']

    print(f"[*] Model info:")
    print(f"    - CV accuracy: {bundle['cv_accuracy']:.4f}")
    print(f"    - Features: {bundle['n_features']}")
    print(f"    - Encoder: {encoder_name}")

    # Initialize encoder
    print(f"\n[*] Initializing encoder...")
    encoder = Encoder(encoder_name)

    # Load data
    print(f"[*] Loading data from core.{input_path}...")
    df = read_table(input_path)
    print(f"[*] Loaded {len(df)} examples")

    # Extract features and predict
    print(f"[*] Making predictions...")
    predictions = []
    probabilities = []

    for idx, row in df.iterrows():
        if idx % 20 == 0:
            print(f"    {idx}/{len(df)}...")

        anchor = row['anchor_text']
        text_a = row['text_a']
        text_b = row['text_b']

        # Extract features
        features = extract_hybrid_features(encoder, anchor, text_a, text_b)
        features = features.reshape(1, -1)

        # Predict
        pred = pipeline.predict(features)[0]
        prob = pipeline.predict_proba(features)[0]

        predictions.append(bool(pred))
        probabilities.append(prob[1])  # probability of A being closer

    print(f"[✓] Predictions complete")

    # Create output dataframe
    results = pd.DataFrame({
        'id': df.get('id', range(len(df))),
        'pred_text_a_is_closer': predictions,
        'prob_text_a': probabilities,
        'prob_text_b': [1.0 - p for p in probabilities]
    })

    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    if output_path.endswith('.jsonl'):
        results.to_json(output_path, orient='records', lines=True, force_ascii=False)
    else:
        results.to_csv(output_path, index=False)

    print(f"[✓] Results saved to {output_path}")

    return results


def evaluate_with_hybrid(input_path, model_path):
    """
    Evaluate hybrid model on labeled data.
    """
    print(f"[*] Loading model from core.{model_path}...")
    bundle = joblib.load(model_path)

    pipeline = bundle['pipeline']
    encoder_name = bundle['encoder_name']

    print(f"[*] Initializing encoder...")
    encoder = Encoder(encoder_name)

    # Load data
    print(f"[*] Loading data from core.{input_path}...")
    df = read_table(input_path)

    if 'text_a_is_closer' not in df.columns:
        raise ValueError("Evaluation data must have 'text_a_is_closer' column")

    print(f"[*] Loaded {len(df)} examples")

    # Extract features
    print(f"[*] Extracting features and predicting...")
    predictions = []
    ground_truth = []

    for idx, row in df.iterrows():
        if idx % 20 == 0:
            print(f"    {idx}/{len(df)}...")

        anchor = row['anchor_text']
        text_a = row['text_a']
        text_b = row['text_b']

        # Extract features
        features = extract_hybrid_features(encoder, anchor, text_a, text_b)
        features = features.reshape(1, -1)

        # Predict
        pred = pipeline.predict(features)[0]
        predictions.append(bool(pred))

        # Ground truth
        label = row['text_a_is_closer']
        if isinstance(label, bool):
            ground_truth.append(label)
        elif isinstance(label, str):
            ground_truth.append(label.lower() in ['true', 't', 'yes', '1'])
        else:
            ground_truth.append(bool(int(label)))

    print(f"[✓] Predictions complete")

    # Calculate metrics
    accuracy = accuracy_score(ground_truth, predictions)

    print("\n" + "="*70)
    print("EVALUATION RESULTS - HYBRID MODEL")
    print("="*70)
    print(f"Dataset: {input_path}")
    print(f"Model: {model_path}")
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Correct: {sum(np.array(predictions) == np.array(ground_truth))}/{len(predictions)}")

    print("\nClassification Report:")
    print(classification_report(ground_truth, predictions,
                               target_names=['B is closer', 'A is closer']))

    print("="*70 + "\n")

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Train and use hybrid narrative similarity model")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Train command
    train_parser = subparsers.add_parser('train', help='Train hybrid model')
    train_parser.add_argument('--train_path', required=True, help='Path to training data')
    train_parser.add_argument('--output_path', default='models/hybrid_model.joblib',
                             help='Path to save trained model')
    train_parser.add_argument('--encoder_name', default='sentence-transformers/all-mpnet-base-v2')
    train_parser.add_argument('--test_size', type=float, default=0.2,
                             help='Test split size (0 = no split)')
    train_parser.add_argument('--cv_folds', type=int, default=5, help='Cross-validation folds')
    train_parser.add_argument('--random_state', type=int, default=42)

    # Predict command
    pred_parser = subparsers.add_parser('predict', help='Predict with hybrid model')
    pred_parser.add_argument('--input_path', required=True, help='Input data')
    pred_parser.add_argument('--model_path', required=True, help='Path to trained model')
    pred_parser.add_argument('--output_path', required=True, help='Output predictions')

    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate hybrid model')
    eval_parser.add_argument('--input_path', required=True, help='Labeled data')
    eval_parser.add_argument('--model_path', required=True, help='Path to trained model')

    args = parser.parse_args()

    if args.command == 'train':
        train_hybrid_model(
            args.train_path,
            args.output_path,
            args.encoder_name,
            args.test_size,
            args.cv_folds,
            args.random_state
        )
    elif args.command == 'predict':
        predict_with_hybrid(args.input_path, args.model_path, args.output_path)
    elif args.command == 'eval':
        evaluate_with_hybrid(args.input_path, args.model_path)


if __name__ == '__main__':
    main()
