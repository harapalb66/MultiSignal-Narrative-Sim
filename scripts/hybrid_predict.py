#!/usr/bin/env python3
"""
Hybrid Ensemble Prediction CLI.

Runs all combination strategies on the dataset and compares results.
"""
from core.core import core.argparse
from core.core import core.pandas as pd
from core.core import core.numpy as np
from core.core.narrative_attractor_dynamics import read_table, write_table
from core.core.hybrid_ensemble import HybridEnsemble


def run_evaluation(
    input_path: str,
    output_path: str,
    use_llm: bool = True,
    limit: int = None,
    extraction_model: str = "gpt-5-mini",
    comparison_model: str = "gpt-5-mini"
):
    """
    Run hybrid ensemble evaluation on dataset.

    Args:
        input_path: Path to input JSONL/CSV
        output_path: Path to save predictions
        use_llm: Whether to use LLM signal
        limit: Max examples to process (None = all)
        extraction_model: Model for LLM structure extraction
        comparison_model: Model for LLM structure comparison
    """
    print("=" * 80)
    print("HYBRID ENSEMBLE EVALUATION")
    print("=" * 80)

    # Initialize
    print("\n[1/4] Initializing ensemble...")
    print(f"      LLM Models: {extraction_model} / {comparison_model}")
    ensemble = HybridEnsemble(
        use_llm=use_llm,
        extraction_model=extraction_model,
        comparison_model=comparison_model
    )

    # Load data
    print(f"\n[2/4] Loading data from core.{input_path}...")
    df = read_table(input_path)
    if limit:
        df = df.head(limit)
    print(f"      Loaded {len(df)} examples")

    # Process examples
    print(f"\n[3/4] Processing examples...")
    results = []

    strategies = ['weighted', 'cascade', 'agreement', 'disagreement']

    for idx, row in df.iterrows():
        if idx % 10 == 0:
            print(f"      {idx}/{len(df)}...")

        try:
            # Compute all signals once
            signals = ensemble.compute_all_signals(
                row['anchor_text'],
                row['text_a'],
                row['text_b']
            )

            result = {
                'id': row.get('id', idx),
                'ground_truth_a': row.get('text_a_is_closer', None),
                # Individual signals
                'center_predicts_a': signals['center_predicts_a'],
                'action_predicts_a': signals['action_predicts_a'],
                'min_chunk_predicts_a': signals['min_chunk_predicts_a'],
                'llm_predicts_a': signals['llm_predicts_a'],
                'llm_confidence': signals['llm_confidence'],
            }

            # Apply each strategy
            for strategy in strategies:
                if strategy == 'weighted':
                    pred, score = ensemble.weighted_vote(signals)
                    result[f'{strategy}_predicts_a'] = pred
                    result[f'{strategy}_score'] = score
                elif strategy == 'cascade':
                    pred, reason = ensemble.confidence_cascade(signals)
                    result[f'{strategy}_predicts_a'] = pred
                    result[f'{strategy}_reason'] = reason
                elif strategy == 'agreement':
                    pred, reason = ensemble.agreement_amplification(signals)
                    result[f'{strategy}_predicts_a'] = pred
                    result[f'{strategy}_reason'] = reason
                elif strategy == 'disagreement':
                    pred, reason = ensemble.disagreement_analysis(signals)
                    result[f'{strategy}_predicts_a'] = pred
                    result[f'{strategy}_reason'] = reason

            results.append(result)

        except Exception as e:
            print(f"      Error on example {idx}: {e}")
            # Default result on error
            result = {
                'id': row.get('id', idx),
                'ground_truth_a': row.get('text_a_is_closer', None),
                'error': str(e)
            }
            for strategy in strategies:
                result[f'{strategy}_predicts_a'] = True  # Default
            results.append(result)

    results_df = pd.DataFrame(results)

    # Analyze results
    print(f"\n[4/4] Analyzing results...")
    has_ground_truth = 'text_a_is_closer' in df.columns and results_df['ground_truth_a'].notna().all()

    best_strategy = None
    best_acc = 0

    print(f"\n{'=' * 80}")
    print("RESULTS")
    print(f"{'=' * 80}")

    if has_ground_truth:
        print(f"\n### Accuracy Comparison ({len(results_df)} examples)")
        print("-" * 60)

        # Individual signals
        print("\nIndividual Signals:")
        for signal in ['center', 'action', 'min_chunk', 'llm']:
            col = f'{signal}_predicts_a'
            if col in results_df.columns:
                correct = (results_df[col] == results_df['ground_truth_a']).sum()
                acc = correct / len(results_df) * 100
                print(f"  {signal:15s}: {acc:5.1f}% ({correct}/{len(results_df)})")

        print("\nEnsemble Strategies:")
        best_strategy = None
        best_acc = 0

        for strategy in strategies:
            col = f'{strategy}_predicts_a'
            if col in results_df.columns:
                correct = (results_df[col] == results_df['ground_truth_a']).sum()
                acc = correct / len(results_df) * 100
                improvement = acc - 64.0  # vs LLM baseline
                print(f"  {strategy:15s}: {acc:5.1f}% ({correct}/{len(results_df)}) [{improvement:+.1f}% vs LLM]")

                if acc > best_acc:
                    best_acc = acc
                    best_strategy = strategy

        print(f"\n  BEST: {best_strategy} with {best_acc:.1f}%")

        # Agreement analysis
        print(f"\n### Agreement Analysis")
        print("-" * 60)

        # Count how many signals agree
        signal_cols = ['center_predicts_a', 'action_predicts_a', 'min_chunk_predicts_a', 'llm_predicts_a']
        results_df['votes_a'] = results_df[signal_cols].sum(axis=1)

        for votes in [4, 3, 2, 1, 0]:
            mask = results_df['votes_a'] == votes
            count = mask.sum()
            if count > 0:
                correct = (results_df.loc[mask, 'agreement_predicts_a'] == results_df.loc[mask, 'ground_truth_a']).sum()
                acc = correct / count * 100
                print(f"  {votes} vote(s) for A: {count:3d} cases ({count/len(results_df)*100:.1f}%), accuracy: {acc:.1f}%")

        # LLM confidence analysis
        print(f"\n### LLM Confidence Analysis")
        print("-" * 60)

        for conf in ['high', 'medium', 'low', 'error', 'disabled']:
            mask = results_df['llm_confidence'] == conf
            count = mask.sum()
            if count > 0:
                correct = (results_df.loc[mask, 'llm_predicts_a'] == results_df.loc[mask, 'ground_truth_a']).sum()
                acc = correct / count * 100
                print(f"  {conf:10s}: {count:3d} cases ({count/len(results_df)*100:.1f}%), accuracy: {acc:.1f}%")

    # Save results
    print(f"\n{'=' * 80}")
    print("SAVING RESULTS")
    print(f"{'=' * 80}")

    # Full results
    results_df.to_json(output_path, orient='records', lines=True, force_ascii=False)
    print(f"  Full results: {output_path}")

    # Best strategy submission format
    if best_strategy:
        submission = results_df[['id', f'{best_strategy}_predicts_a']].copy()
        submission.columns = ['id', 'text_a_is_closer']
        submission_path = output_path.replace('.jsonl', f'_{best_strategy}_submission.jsonl')
        submission.to_json(submission_path, orient='records', lines=True, force_ascii=False)
        print(f"  Best submission: {submission_path}")

    print(f"\n{'=' * 80}\n")

    return results_df, best_strategy, best_acc


def main():
    parser = argparse.ArgumentParser(
        description='Hybrid ensemble prediction for narrative similarity'
    )
    parser.add_argument(
        '--input_path',
        required=True,
        help='Path to input data (JSONL/CSV)'
    )
    parser.add_argument(
        '--output_path',
        required=True,
        help='Path to save predictions'
    )
    parser.add_argument(
        '--no-llm',
        dest='use_llm',
        action='store_false',
        help='Disable LLM signal (faster but less accurate)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Max examples to process'
    )
    parser.add_argument(
        '--extraction-model',
        type=str,
        default='gpt-5-mini',
        help='Model for LLM structure extraction'
    )
    parser.add_argument(
        '--comparison-model',
        type=str,
        default='gpt-5-mini',
        help='Model for LLM structure comparison'
    )

    args = parser.parse_args()

    run_evaluation(
        input_path=args.input_path,
        output_path=args.output_path,
        use_llm=args.use_llm,
        limit=args.limit,
        extraction_model=args.extraction_model,
        comparison_model=args.comparison_model
    )


if __name__ == '__main__':
    main()
