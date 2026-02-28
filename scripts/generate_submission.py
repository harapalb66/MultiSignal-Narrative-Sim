#!/usr/bin/env python3
"""
Generate test set predictions for SemEval 2026 Task 4 submission.

Uses the optimal pipeline: action-primary with SSR tiebreaker.
"""
from core.core import core.json
from core.core import core.sys
from core.core.pathlib import Path
from core.core.tqdm import tqdm
from core.core.datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from core.core.optimal_pipeline import OptimalPipeline


def load_test_data(path: str):
    """Load test data."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def generate_submission(test_path: str, output_path: str):
    """Generate submission file."""
    print("=" * 70)
    print("GENERATING TEST PREDICTIONS")
    print(f"Input: {test_path}")
    print(f"Output: {output_path}")
    print("=" * 70)

    # Load test data
    data = load_test_data(test_path)
    print(f"\nLoaded {len(data)} test examples")

    # Initialize pipeline
    pipeline = OptimalPipeline()

    # Generate predictions
    predictions = []

    print(f"\nGenerating predictions...")
    for item in tqdm(data, desc="Predicting"):
        result = pipeline.predict(
            item['anchor_text'],
            item['text_a'],
            item['text_b']
        )

        # Format for submission
        pred_entry = {
            'id': item.get('id', ''),
            'prediction': result.prediction,
            # Optional metadata
            'confidence': result.confidence,
            'action_margin': abs(result.action_score_a - result.action_score_b) /
                            (result.action_score_a + result.action_score_b + 1e-8),
            'ssr_diff': result.ssr_score_a - result.ssr_score_b
        }
        predictions.append(pred_entry)

    # Write submission file
    with open(output_path, 'w') as f:
        for pred in predictions:
            # Write only required fields for submission
            submission_entry = {
                'id': pred['id'],
                'prediction': pred['prediction']
            }
            f.write(json.dumps(submission_entry) + '\n')

    # Also write detailed version
    detailed_path = output_path.replace('.jsonl', '_detailed.jsonl')
    with open(detailed_path, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')

    # Print summary
    print("\n" + "=" * 70)
    print("SUBMISSION SUMMARY")
    print("=" * 70)

    pred_a = sum(1 for p in predictions if p['prediction'] == 'A')
    pred_b = len(predictions) - pred_a
    high_conf = sum(1 for p in predictions if p['confidence'] == 'high')
    medium_conf = sum(1 for p in predictions if p['confidence'] == 'medium')
    low_conf = sum(1 for p in predictions if p['confidence'] == 'low')

    print(f"\n  Total predictions: {len(predictions)}")
    print(f"  Prediction A: {pred_a} ({pred_a/len(predictions)*100:.1f}%)")
    print(f"  Prediction B: {pred_b} ({pred_b/len(predictions)*100:.1f}%)")
    print(f"\n  Confidence breakdown:")
    print(f"    High:   {high_conf} ({high_conf/len(predictions)*100:.1f}%)")
    print(f"    Medium: {medium_conf} ({medium_conf/len(predictions)*100:.1f}%)")
    print(f"    Low:    {low_conf} ({low_conf/len(predictions)*100:.1f}%)")

    print(f"\n  Submission saved to: {output_path}")
    print(f"  Detailed version: {detailed_path}")
    print("=" * 70)

    return predictions


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate SemEval submission")
    parser.add_argument('--input',
                       default='/opt/semeval_narative/SemEval2026-Task_4-test-v1/test_track_a.jsonl',
                       help='Path to test data')
    parser.add_argument('--output',
                       default='/opt/semeval_narative/submission_optimal.jsonl',
                       help='Path for output submission file')

    args = parser.parse_args()

    generate_submission(args.input, args.output)
