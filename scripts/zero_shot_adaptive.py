#!/usr/bin/env python3
"""
Adaptive zero-shot prediction using:
1. Center (attractor) distances - semantic gravity
2. Chunk distribution distances - temporal/sequential alignment
3. Confidence-based override - only trust chunks when highly confident

Decision logic:
- Compute both center distance and chunk distribution score
- If they agree → use center (high confidence)
- If they disagree and chunk confidence > threshold → use chunks (temporal wins)
- If they disagree and chunk confidence ≤ threshold → use center (stay safe)

Based on dev set analysis:
- Optimal threshold: 0.25
- Expected accuracy: 60% (vs 58% center-only, 56.5% always-chunk)
"""
from core.core import core.argparse
from core.core import core.numpy as np
from core.core import core.pandas as pd
from core.core.narrative_attractor_dynamics import Encoder, read_table, sent_split


def create_chunks(text, chunk_size=3):
    """
    Split text into overlapping chunks of sentences.
    """
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


def compute_chunk_distances(encoder, anchor_text, text_a, text_b, chunk_size=3):
    """
    Compute chunk-to-chunk distance vectors for both pairs.
    """
    anchor_chunks = create_chunks(anchor_text, chunk_size)
    chunks_a = create_chunks(text_a, chunk_size)
    chunks_b = create_chunks(text_b, chunk_size)

    anchor_embeds = encoder.embed_sents(anchor_chunks)
    embeds_a = encoder.embed_sents(chunks_a)
    embeds_b = encoder.embed_sents(chunks_b)

    dist_vec_a = []
    dist_vec_b = []

    for anchor_emb in anchor_embeds:
        dists_a = [float(np.linalg.norm(anchor_emb - emb_a)) for emb_a in embeds_a]
        dists_b = [float(np.linalg.norm(anchor_emb - emb_b)) for emb_b in embeds_b]
        dist_vec_a.append(min(dists_a))
        dist_vec_b.append(min(dists_b))

    return np.array(dist_vec_a), np.array(dist_vec_b)


def compare_distributions(dist_vec_a, dist_vec_b):
    """
    Compare two distance distributions.
    Returns positive value if A is closer, negative if B is closer.
    """
    mean_a = np.mean(dist_vec_a)
    mean_b = np.mean(dist_vec_b)

    all_dists = np.concatenate([dist_vec_a, dist_vec_b])
    threshold = np.median(all_dists)

    count_small_a = np.sum(dist_vec_a < threshold)
    count_small_b = np.sum(dist_vec_b < threshold)

    mean_diff = mean_b - mean_a
    count_diff = count_small_a - count_small_b
    count_diff_norm = count_diff / len(dist_vec_a)

    score = 0.7 * mean_diff + 0.3 * count_diff_norm
    return score


def adaptive_zero_shot_predict(encoder, anchor_text, text_a, text_b,
                                chunk_size=3, override_threshold=0.25):
    """
    Adaptive zero-shot prediction with confidence-based override.

    Args:
        encoder: Sentence encoder
        anchor_text: Anchor narrative
        text_a: First candidate narrative
        text_b: Second candidate narrative
        chunk_size: Sentences per chunk
        override_threshold: Minimum confidence to trust chunk override

    Returns:
        prediction: True if A is closer, False if B is closer
        confidence: Confidence score
        decision_type: 'agreement', 'chunks_override', or 'center_fallback'
    """
    # 1. Center (attractor) distances
    anchor_sents = sent_split(anchor_text)
    a_sents = sent_split(text_a)
    b_sents = sent_split(text_b)

    anchor_embeds = encoder.embed_sents(anchor_sents)
    a_embeds = encoder.embed_sents(a_sents)
    b_embeds = encoder.embed_sents(b_sents)

    center_anchor = anchor_embeds.mean(axis=0)
    center_a = a_embeds.mean(axis=0)
    center_b = b_embeds.mean(axis=0)

    dist_center_a = float(np.linalg.norm(center_anchor - center_a))
    dist_center_b = float(np.linalg.norm(center_anchor - center_b))

    center_predicts_a = dist_center_a < dist_center_b
    center_diff = abs(dist_center_b - dist_center_a)

    # 2. Chunk distribution distances
    dist_vec_a, dist_vec_b = compute_chunk_distances(encoder, anchor_text, text_a, text_b, chunk_size)
    chunk_score = compare_distributions(dist_vec_a, dist_vec_b)
    chunk_predicts_a = chunk_score > 0
    chunk_confidence = abs(chunk_score)

    # 3. Adaptive decision logic
    if center_predicts_a == chunk_predicts_a:
        # Agreement: use center distance
        prediction = center_predicts_a
        confidence = center_diff
        decision_type = 'agreement'
    elif chunk_confidence >= override_threshold:
        # High-confidence disagreement: trust chunks (temporal wins)
        prediction = chunk_predicts_a
        confidence = chunk_confidence
        decision_type = 'chunks_override'
    else:
        # Low-confidence disagreement: stay with center (fallback)
        prediction = center_predicts_a
        confidence = center_diff
        decision_type = 'center_fallback'

    return prediction, confidence, decision_type


def predict_test_set(input_path, output_path, chunk_size=3, override_threshold=0.25):
    """
    Run adaptive zero-shot prediction on test set.
    """
    print("=" * 80)
    print("ADAPTIVE ZERO-SHOT PREDICTION")
    print("=" * 80)
    print("Strategy: Center distance + Chunk distribution (adaptive)")
    print(f"Chunk size: {chunk_size} sentences")
    print(f"Override threshold: {override_threshold}")
    print("=" * 80)

    # Initialize
    print(f"\n[1/3] Initializing encoder...")
    encoder = Encoder('sentence-transformers/all-mpnet-base-v2')

    # Load data
    print(f"\n[2/3] Loading data from core.{input_path}...")
    df = read_table(input_path)
    print(f"      Loaded {len(df)} examples")

    # Make predictions
    print(f"\n[3/3] Making predictions...")
    results = []
    agreement_count = 0
    override_count = 0
    fallback_count = 0

    for idx, row in df.iterrows():
        if idx % 20 == 0:
            print(f"      {idx}/{len(df)}...")

        prediction, confidence, decision_type = adaptive_zero_shot_predict(
            encoder,
            row['anchor_text'],
            row['text_a'],
            row['text_b'],
            chunk_size=chunk_size,
            override_threshold=override_threshold
        )

        if decision_type == 'agreement':
            agreement_count += 1
        elif decision_type == 'chunks_override':
            override_count += 1
        else:
            fallback_count += 1

        results.append({
            'id': row.get('id', idx),
            'text_a_is_closer': prediction,
            'confidence': confidence,
            'decision_type': decision_type,
        })

    # Save submission format
    print(f"\n[*] Saving predictions to {output_path}...")
    results_df = pd.DataFrame(results)

    # Submission file (only id and prediction)
    submission = results_df[['id', 'text_a_is_closer']]
    submission.to_json(output_path, orient='records', lines=True, force_ascii=False)

    # Full details file
    full_path = output_path.replace('.jsonl', '_full.jsonl')
    results_df.to_json(full_path, orient='records', lines=True, force_ascii=False)

    print(f"\n{'=' * 80}")
    print(f"✓ ADAPTIVE ZERO-SHOT PREDICTION COMPLETE!")
    print(f"{'=' * 80}")
    print(f"  Total predictions: {len(results)}")
    print(f"  A is closer: {sum(r['text_a_is_closer'] for r in results)} ({sum(r['text_a_is_closer'] for r in results)/len(results)*100:.1f}%)")
    print(f"  B is closer: {sum(not r['text_a_is_closer'] for r in results)} ({sum(not r['text_a_is_closer'] for r in results)/len(results)*100:.1f}%)")
    print(f"\n  Decision breakdown:")
    print(f"    Agreement (center + chunks): {agreement_count} ({agreement_count/len(results)*100:.1f}%)")
    print(f"    Chunks override (high conf): {override_count} ({override_count/len(results)*100:.1f}%)")
    print(f"    Center fallback (low conf): {fallback_count} ({fallback_count/len(results)*100:.1f}%)")
    print(f"\n  Submission file: {output_path}")
    print(f"  Full details: {full_path}")
    print(f"  Expected: 60% (validated on dev set)")
    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description='Adaptive zero-shot prediction with confidence-based chunk override')
    parser.add_argument('--input_path', required=True, help='Path to test data')
    parser.add_argument('--output_path', required=True, help='Path to save predictions')
    parser.add_argument('--chunk_size', type=int, default=3, help='Sentences per chunk (default: 3)')
    parser.add_argument('--override_threshold', type=float, default=0.25,
                       help='Minimum confidence to trust chunk override (default: 0.25)')

    args = parser.parse_args()

    predict_test_set(args.input_path, args.output_path, args.chunk_size, args.override_threshold)


if __name__ == '__main__':
    main()
