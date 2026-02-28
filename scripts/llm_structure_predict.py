#!/usr/bin/env python3
"""
LLM-based structured narrative comparison prediction.

Extracts narrative structures (events, agents, triples, causal links)
and uses LLM reasoning to compare them.
"""
from core.core import core.argparse
from core.core import core.pandas as pd
from core.core.narrative_attractor_dynamics import read_table, write_table
from core.core.narrative_structure_extractor import NarrativeStructureExtractor
from core.core.structure_comparator import StructureComparator


def predict(
    input_path: str,
    output_path: str,
    extraction_model: str = "gpt-4o-mini",
    comparison_model: str = "gpt-4o-mini",
    cache_dir: str = "/opt/semeval_narative/structure_cache",
    limit: int = None
):
    """
    Run LLM-based structured comparison on dataset.

    Args:
        input_path: Path to input JSONL/CSV
        output_path: Path to save predictions
        extraction_model: Model for structure extraction
        comparison_model: Model for comparison reasoning
        cache_dir: Directory to cache extracted structures
        limit: Max examples to process (None = all)
    """
    print("=" * 70)
    print("LLM STRUCTURED NARRATIVE COMPARISON")
    print("=" * 70)
    print(f"Extraction model: {extraction_model}")
    print(f"Comparison model: {comparison_model}")
    print(f"Cache directory: {cache_dir}")
    print("=" * 70)

    # Initialize
    print("\n[1/4] Initializing...")
    extractor = NarrativeStructureExtractor(
        model=extraction_model,
        cache_dir=cache_dir
    )
    comparator = StructureComparator(model=comparison_model)

    # Load data
    print(f"\n[2/4] Loading data from core.{input_path}...")
    df = read_table(input_path)
    if limit:
        df = df.head(limit)
    print(f"      Loaded {len(df)} examples")

    # Process examples
    print(f"\n[3/4] Processing examples...")
    results = []

    for idx, row in df.iterrows():
        if idx % 10 == 0:
            print(f"      {idx}/{len(df)}...")

        try:
            # Extract structures (uses cache if available)
            anchor_struct = extractor.extract(row['anchor_text'])
            a_struct = extractor.extract(row['text_a'])
            b_struct = extractor.extract(row['text_b'])

            # Compare structures
            comparison = comparator.compare(anchor_struct, a_struct, b_struct)

            # Convert answer to boolean
            text_a_is_closer = comparison.answer.upper() == 'A'

            results.append({
                'id': row.get('id', idx),
                'text_a_is_closer': text_a_is_closer,
                'confidence': comparison.confidence,
                'reasoning': comparison.reasoning,
            })

        except Exception as e:
            print(f"      Error on example {idx}: {e}")
            results.append({
                'id': row.get('id', idx),
                'text_a_is_closer': True,  # Default
                'confidence': 'error',
                'reasoning': str(e),
            })

    # Calculate accuracy if ground truth available
    results_df = pd.DataFrame(results)

    if 'text_a_is_closer' in df.columns:
        df_reset = df.reset_index(drop=True)
        correct = 0
        for i, row in results_df.iterrows():
            if i < len(df_reset) and row['text_a_is_closer'] == df_reset.iloc[i]['text_a_is_closer']:
                correct += 1
        accuracy = correct / len(results_df) * 100
        print(f"\n      Accuracy: {accuracy:.1f}% ({correct}/{len(results_df)})")

    # Save results
    print(f"\n[4/4] Saving predictions to {output_path}...")

    # Submission format (minimal)
    submission = results_df[['id', 'text_a_is_closer']]
    submission.to_json(output_path, orient='records', lines=True, force_ascii=False)

    # Full format with reasoning
    full_path = output_path.replace('.jsonl', '_full.jsonl')
    results_df.to_json(full_path, orient='records', lines=True, force_ascii=False)

    # Summary
    a_count = results_df['text_a_is_closer'].sum()
    b_count = len(results_df) - a_count

    print(f"\n{'=' * 70}")
    print(f"COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total predictions: {len(results_df)}")
    print(f"  A is closer: {a_count} ({a_count/len(results_df)*100:.1f}%)")
    print(f"  B is closer: {b_count} ({b_count/len(results_df)*100:.1f}%)")
    print(f"\n  Submission file: {output_path}")
    print(f"  Full details: {full_path}")
    print(f"{'=' * 70}\n")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='LLM-based structured narrative comparison'
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
        '--extraction_model',
        default='gpt-4o-mini',
        help='Model for structure extraction (default: gpt-4o-mini)'
    )
    parser.add_argument(
        '--comparison_model',
        default='gpt-4o-mini',
        help='Model for comparison (default: gpt-4o-mini)'
    )
    parser.add_argument(
        '--cache_dir',
        default='/opt/semeval_narative/structure_cache',
        help='Directory to cache structures'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Max examples to process'
    )

    args = parser.parse_args()

    predict(
        input_path=args.input_path,
        output_path=args.output_path,
        extraction_model=args.extraction_model,
        comparison_model=args.comparison_model,
        cache_dir=args.cache_dir,
        limit=args.limit
    )


if __name__ == '__main__':
    main()
