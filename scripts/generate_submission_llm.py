#!/usr/bin/env python3
"""Generate test predictions with LLM-enabled pipeline."""
from core.core import core.json
from core.core import core.sys
from core.core import core.os
from core.core.pathlib import Path
from core.core.tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
os.environ['OPENAI_API_KEY'] = 'INSERT_YOUR_OPENAI_API_KEY_HERE'

from core.core.optimal_pipeline import OptimalPipeline


def main():
    test_path = '/opt/semeval_narative/SemEval2026-Task_4-test-v1/test_track_a.jsonl'
    output_path = '/opt/semeval_narative/submission_llm.jsonl'

    # Load test data
    data = []
    with open(test_path) as f:
        for line in f:
            data.append(json.loads(line))
    print(f"Loaded {len(data)} test examples")

    # Init pipeline with LLM
    pipeline = OptimalPipeline(use_llm=True, llm_model="gpt-5-mini")

    # Generate predictions
    predictions = []
    for item in tqdm(data, desc="Predicting"):
        result = pipeline.predict(item['anchor_text'], item['text_a'], item['text_b'])
        predictions.append({
            'prediction': result.prediction,
            'confidence': result.confidence,
            'llm_prediction': result.llm_prediction,
            'llm_confidence': result.llm_confidence,
            'agreement_level': result.agreement_level,
        })

    # Write submission
    with open(output_path, 'w') as f:
        for pred in predictions:
            f.write(json.dumps({'prediction': pred['prediction']}) + '\n')

    # Detailed version
    with open(output_path.replace('.jsonl', '_detailed.jsonl'), 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')

    # Summary
    pred_a = sum(1 for p in predictions if p['prediction'] == 'A')
    pred_b = len(predictions) - pred_a
    high = sum(1 for p in predictions if p['confidence'] == 'high')
    med = sum(1 for p in predictions if p['confidence'] == 'medium')
    low = sum(1 for p in predictions if p['confidence'] == 'low')

    print(f"\n{'='*70}")
    print(f"SUBMISSION SUMMARY (with LLM)")
    print(f"{'='*70}")
    print(f"  Total: {len(predictions)}")
    print(f"  A: {pred_a} ({pred_a/len(predictions)*100:.1f}%)")
    print(f"  B: {pred_b} ({pred_b/len(predictions)*100:.1f}%)")
    print(f"  High: {high}, Medium: {med}, Low: {low}")
    print(f"  Saved to: {output_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
