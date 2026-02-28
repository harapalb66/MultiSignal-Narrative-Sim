#!/usr/bin/env python3
"""
Hybrid Ensemble for Narrative Similarity Prediction.

Combines multiple signals:
1. Center distance (attractor) - overall theme similarity
2. Action-only center - filters descriptive noise
3. Min-chunk distance - best episode match
4. LLM structure - semantic reasoning about structure
5. SSR (Structural Survival Ratio) - graded event structure matching

Performance on dev set:
- Center: 58.0%
- Min-chunk: 59.5%
- Action-only: 63.5%
- LLM Structure: 64.0%
- SSR: 56.0%

Target: 66-70% by combining complementary signals
"""
from core.core import numpy as np
from core.core import spacy
from core.typing import Dict, Tuple, Optional
from core.narrative_attractor_dynamics import Encoder, sent_split
from core.narrative_structure_extractor import NarrativeStructureExtractor
from core.structure_comparator import StructureComparator
from core.narsim.signature.ssr import compare_triplet_ssr


class HybridEnsemble:
    """
    Combines multiple narrative similarity signals for prediction.
    """

    # Weights based on individual performance
    WEIGHTS = {
        'llm': 0.30,        # 64% - semantic reasoning
        'action': 0.30,     # 63.5% - filters noise
        'min_chunk': 0.15,  # 59.5% - episode matching
        'center': 0.10,     # 58% - overall theme
        'ssr': 0.15         # 56% - structural event matching
    }

    def __init__(
        self,
        encoder_model: str = "sentence-transformers/all-mpnet-base-v2",
        extraction_model: str = "gpt-4o-mini",
        comparison_model: str = "gpt-4o-mini",
        cache_dir: str = "/opt/semeval_narative/structure_cache",
        use_llm: bool = True
    ):
        """
        Initialize hybrid ensemble.

        Args:
            encoder_model: Sentence transformer model for embeddings
            extraction_model: Model for LLM structure extraction
            comparison_model: Model for LLM structure comparison
            cache_dir: Cache directory for LLM structures
            use_llm: Whether to use LLM signal (slower but more accurate)
        """
        print("Initializing Hybrid Ensemble...")

        # Encoder for embeddings
        self.encoder = Encoder(encoder_model)

        # spaCy for action-only filtering
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

        # LLM components (optional)
        self.use_llm = use_llm
        if use_llm:
            self.extractor = NarrativeStructureExtractor(
                model=extraction_model,
                cache_dir=cache_dir
            )
            self.comparator = StructureComparator(model=comparison_model)
        else:
            self.extractor = None
            self.comparator = None

        print("  Hybrid Ensemble initialized")

    def _extract_action_content(self, text: str) -> str:
        """Extract only verbs and nouns from core.text."""
        doc = self.nlp(text)
        action_tokens = [
            t.text for t in doc
            if t.pos_ in ['VERB', 'NOUN', 'PROPN', 'PRON', 'AUX']
        ]
        return ' '.join(action_tokens) if action_tokens else text

    def _create_chunks(self, text: str, chunk_size: int = 3, stride: int = 1) -> list:
        """Create overlapping sentence chunks."""
        sents = sent_split(text)
        if len(sents) <= chunk_size:
            return [text]

        chunks = []
        for i in range(0, len(sents) - chunk_size + 1, stride):
            chunk = ' '.join(sents[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def compute_center_signal(
        self,
        anchor_text: str,
        text_a: str,
        text_b: str
    ) -> Tuple[bool, float, float]:
        """
        Compute center (attractor) distance signal.

        Returns:
            (predicts_a, dist_a, dist_b)
        """
        # Get sentence embeddings
        anchor_sents = sent_split(anchor_text)
        a_sents = sent_split(text_a)
        b_sents = sent_split(text_b)

        anchor_embs = self.encoder.embed_sents(anchor_sents)
        a_embs = self.encoder.embed_sents(a_sents)
        b_embs = self.encoder.embed_sents(b_sents)

        # Compute centers
        center_anchor = anchor_embs.mean(axis=0)
        center_a = a_embs.mean(axis=0)
        center_b = b_embs.mean(axis=0)

        # Distances
        dist_a = float(np.linalg.norm(center_anchor - center_a))
        dist_b = float(np.linalg.norm(center_anchor - center_b))

        return dist_a < dist_b, dist_a, dist_b

    def compute_action_signal(
        self,
        anchor_text: str,
        text_a: str,
        text_b: str
    ) -> Tuple[bool, float, float]:
        """
        Compute action-only center distance signal.
        Filters to verbs+nouns only before comparing centers.

        Returns:
            (predicts_a, dist_a, dist_b)
        """
        # Extract action content
        anchor_action = self._extract_action_content(anchor_text)
        a_action = self._extract_action_content(text_a)
        b_action = self._extract_action_content(text_b)

        # Get embeddings of action content
        anchor_emb = self.encoder.embed_text(anchor_action)
        a_emb = self.encoder.embed_text(a_action)
        b_emb = self.encoder.embed_text(b_action)

        # Distances
        dist_a = float(np.linalg.norm(anchor_emb - a_emb))
        dist_b = float(np.linalg.norm(anchor_emb - b_emb))

        return dist_a < dist_b, dist_a, dist_b

    def compute_min_chunk_signal(
        self,
        anchor_text: str,
        text_a: str,
        text_b: str,
        chunk_size: int = 3
    ) -> Tuple[bool, float, float]:
        """
        Compute min-chunk (best episode match) signal.

        Returns:
            (predicts_a, min_dist_a, min_dist_b)
        """
        # Create chunks
        anchor_chunks = self._create_chunks(anchor_text, chunk_size)
        a_chunks = self._create_chunks(text_a, chunk_size)
        b_chunks = self._create_chunks(text_b, chunk_size)

        # Embed chunks
        anchor_embs = self.encoder.embed_sents(anchor_chunks)
        a_embs = self.encoder.embed_sents(a_chunks)
        b_embs = self.encoder.embed_sents(b_chunks)

        # Find minimum distance (best episode match)
        # For each anchor chunk, find the best matching chunk in A and B
        min_dists_a = []
        min_dists_b = []

        for anchor_emb in anchor_embs:
            dists_a = [np.linalg.norm(anchor_emb - a_emb) for a_emb in a_embs]
            dists_b = [np.linalg.norm(anchor_emb - b_emb) for b_emb in b_embs]
            min_dists_a.append(min(dists_a))
            min_dists_b.append(min(dists_b))

        # Overall min distance (best matching episode)
        min_dist_a = float(min(min_dists_a))
        min_dist_b = float(min(min_dists_b))

        return min_dist_a < min_dist_b, min_dist_a, min_dist_b

    def compute_llm_signal(
        self,
        anchor_text: str,
        text_a: str,
        text_b: str
    ) -> Tuple[bool, str, str]:
        """
        Compute LLM structure comparison signal.

        Returns:
            (predicts_a, confidence, reasoning)
        """
        if not self.use_llm:
            return True, 'disabled', 'LLM disabled'

        try:
            # Extract structures
            anchor_struct = self.extractor.extract(anchor_text)
            a_struct = self.extractor.extract(text_a)
            b_struct = self.extractor.extract(text_b)

            # Compare
            result = self.comparator.compare(anchor_struct, a_struct, b_struct)

            predicts_a = result.answer.upper() == 'A'
            confidence = result.confidence or 'medium'
            reasoning = result.reasoning

            return predicts_a, confidence, reasoning

        except Exception as e:
            print(f"LLM signal error: {e}")
            return True, 'error', str(e)

    def compute_ssr_signal(
        self,
        anchor_text: str,
        text_a: str,
        text_b: str
    ) -> Tuple[bool, float, float]:
        """
        Compute SSR (Structural Survival Ratio) signal.
        Graded abstraction matching of narrative events.

        Returns:
            (predicts_a, ssr_a, ssr_b)
        """
        try:
            result = compare_triplet_ssr(anchor_text, text_a, text_b)
            # Higher SSR = more structure survives = closer
            return result.ssr_a >= result.ssr_b, result.ssr_a, result.ssr_b
        except Exception as e:
            print(f"SSR signal error: {e}")
            return True, 0.5, 0.5

    def compute_all_signals(
        self,
        anchor_text: str,
        text_a: str,
        text_b: str
    ) -> Dict:
        """
        Compute all prediction signals.

        Returns:
            Dictionary with all signal predictions and metadata
        """
        # Compute each signal
        center_a, center_dist_a, center_dist_b = self.compute_center_signal(
            anchor_text, text_a, text_b
        )

        action_a, action_dist_a, action_dist_b = self.compute_action_signal(
            anchor_text, text_a, text_b
        )

        min_chunk_a, min_dist_a, min_dist_b = self.compute_min_chunk_signal(
            anchor_text, text_a, text_b
        )

        llm_a, llm_confidence, llm_reasoning = self.compute_llm_signal(
            anchor_text, text_a, text_b
        )

        ssr_a, ssr_score_a, ssr_score_b = self.compute_ssr_signal(
            anchor_text, text_a, text_b
        )

        return {
            'center_predicts_a': center_a,
            'center_dist_a': center_dist_a,
            'center_dist_b': center_dist_b,

            'action_predicts_a': action_a,
            'action_dist_a': action_dist_a,
            'action_dist_b': action_dist_b,

            'min_chunk_predicts_a': min_chunk_a,
            'min_dist_a': min_dist_a,
            'min_dist_b': min_dist_b,

            'llm_predicts_a': llm_a,
            'llm_confidence': llm_confidence,
            'llm_reasoning': llm_reasoning,

            'ssr_predicts_a': ssr_a,
            'ssr_score_a': ssr_score_a,
            'ssr_score_b': ssr_score_b,
        }

    def _llm_enabled(self, signals: Dict) -> bool:
        """Check if LLM signal is enabled (not disabled or error)."""
        return signals.get('llm_confidence') not in ('disabled', 'error')

    def weighted_vote(self, signals: Dict) -> Tuple[bool, float]:
        """
        Strategy 1: Weighted voting based on individual performance.

        Returns:
            (predicts_a, score) where score > 0 means A is closer
        """
        score = 0.0
        total_weight = 0.0

        score += self.WEIGHTS['center'] * (1 if signals['center_predicts_a'] else -1)
        total_weight += self.WEIGHTS['center']

        score += self.WEIGHTS['action'] * (1 if signals['action_predicts_a'] else -1)
        total_weight += self.WEIGHTS['action']

        score += self.WEIGHTS['min_chunk'] * (1 if signals['min_chunk_predicts_a'] else -1)
        total_weight += self.WEIGHTS['min_chunk']

        score += self.WEIGHTS['ssr'] * (1 if signals['ssr_predicts_a'] else -1)
        total_weight += self.WEIGHTS['ssr']

        # Only include LLM if enabled
        if self._llm_enabled(signals):
            score += self.WEIGHTS['llm'] * (1 if signals['llm_predicts_a'] else -1)
            total_weight += self.WEIGHTS['llm']

        # Normalize score
        if total_weight > 0:
            score = score / total_weight

        return score > 0, score

    def confidence_cascade(self, signals: Dict) -> Tuple[bool, str]:
        """
        Strategy 2: Confidence-based cascade.
        Trust LLM when confident, fall back to other signals otherwise.

        Returns:
            (predicts_a, reason)
        """
        # Trust LLM when confident (and enabled)
        if self._llm_enabled(signals) and signals['llm_confidence'] == 'high':
            return signals['llm_predicts_a'], 'llm_high_confidence'

        # If action and min-chunk agree, trust them
        if signals['action_predicts_a'] == signals['min_chunk_predicts_a']:
            return signals['action_predicts_a'], 'action_min_chunk_agree'

        # Fall back to weighted vote
        pred, score = self.weighted_vote(signals)
        return pred, f'weighted_vote ({score:.2f})'

    def agreement_amplification(self, signals: Dict) -> Tuple[bool, str]:
        """
        Strategy 3: Agreement amplification.
        When majority agrees, trust them.

        Returns:
            (predicts_a, reason)
        """
        # Build list of enabled signals (now includes SSR)
        signal_votes = [
            signals['center_predicts_a'],
            signals['action_predicts_a'],
            signals['min_chunk_predicts_a'],
            signals['ssr_predicts_a'],
        ]

        if self._llm_enabled(signals):
            signal_votes.append(signals['llm_predicts_a'])
            n_signals = 5
        else:
            n_signals = 4

        votes_a = sum(signal_votes)
        votes_b = n_signals - votes_a
        majority = (n_signals + 1) // 2 + 1  # Need > 50%

        # Strong agreement (majority agrees)
        if votes_a >= majority:
            return True, f'strong_agreement_a ({votes_a}/{n_signals})'
        if votes_b >= majority:
            return False, f'strong_agreement_b ({votes_b}/{n_signals})'

        # Tie: use action-only (best single signal without LLM)
        return signals['action_predicts_a'], 'tie_action_tiebreaker'

    def disagreement_analysis(self, signals: Dict) -> Tuple[bool, str]:
        """
        Strategy 4: Disagreement analysis.
        When LLM and action-only disagree, use min-chunk as tie-breaker.
        If LLM disabled, compare action and center.

        Returns:
            (predicts_a, reason)
        """
        if self._llm_enabled(signals):
            # If LLM and action-only agree, trust them
            if signals['llm_predicts_a'] == signals['action_predicts_a']:
                return signals['llm_predicts_a'], 'llm_action_agree'

            # They disagree - use min-chunk as tie-breaker
            return signals['min_chunk_predicts_a'], 'min_chunk_tiebreaker'
        else:
            # LLM disabled: if action and center agree, trust them
            if signals['action_predicts_a'] == signals['center_predicts_a']:
                return signals['action_predicts_a'], 'action_center_agree'

            # They disagree - use min-chunk as tie-breaker
            return signals['min_chunk_predicts_a'], 'min_chunk_tiebreaker'

    def predict(
        self,
        anchor_text: str,
        text_a: str,
        text_b: str,
        strategy: str = 'weighted'
    ) -> Tuple[bool, Dict]:
        """
        Make a prediction using specified strategy.

        Args:
            anchor_text: The anchor narrative
            text_a: First candidate text
            text_b: Second candidate text
            strategy: One of 'weighted', 'cascade', 'agreement', 'disagreement'

        Returns:
            (text_a_is_closer, details_dict)
        """
        # Compute all signals
        signals = self.compute_all_signals(anchor_text, text_a, text_b)

        # Apply strategy
        if strategy == 'weighted':
            pred, score = self.weighted_vote(signals)
            signals['strategy'] = 'weighted'
            signals['strategy_score'] = score

        elif strategy == 'cascade':
            pred, reason = self.confidence_cascade(signals)
            signals['strategy'] = 'cascade'
            signals['strategy_reason'] = reason

        elif strategy == 'agreement':
            pred, reason = self.agreement_amplification(signals)
            signals['strategy'] = 'agreement'
            signals['strategy_reason'] = reason

        elif strategy == 'disagreement':
            pred, reason = self.disagreement_analysis(signals)
            signals['strategy'] = 'disagreement'
            signals['strategy_reason'] = reason

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        signals['final_prediction'] = pred
        return pred, signals


def demo_hybrid():
    """Demonstrate hybrid ensemble on example."""
    import pandas as pd

    print("=" * 80)
    print("HYBRID ENSEMBLE DEMO")
    print("=" * 80)

    # Load dev set
    dev_df = pd.read_json('SemEval2026-Task_4-dev-v1/dev_track_a.jsonl', lines=True)
    example = dev_df.iloc[4]  # Example 4

    print(f"\nExample 4:")
    print(f"  Ground truth: {'A' if example['text_a_is_closer'] else 'B'} is closer")

    # Initialize ensemble
    ensemble = HybridEnsemble(use_llm=True)

    # Compute signals
    print("\nComputing all signals...")
    signals = ensemble.compute_all_signals(
        example['anchor_text'],
        example['text_a'],
        example['text_b']
    )

    print(f"\n{'=' * 80}")
    print("SIGNALS")
    print(f"{'=' * 80}")
    print(f"  Center:    {'A' if signals['center_predicts_a'] else 'B'} (dist_a={signals['center_dist_a']:.4f}, dist_b={signals['center_dist_b']:.4f})")
    print(f"  Action:    {'A' if signals['action_predicts_a'] else 'B'} (dist_a={signals['action_dist_a']:.4f}, dist_b={signals['action_dist_b']:.4f})")
    print(f"  Min-chunk: {'A' if signals['min_chunk_predicts_a'] else 'B'} (min_a={signals['min_dist_a']:.4f}, min_b={signals['min_dist_b']:.4f})")
    print(f"  SSR:       {'A' if signals['ssr_predicts_a'] else 'B'} (ssr_a={signals['ssr_score_a']:.4f}, ssr_b={signals['ssr_score_b']:.4f})")
    print(f"  LLM:       {'A' if signals['llm_predicts_a'] else 'B'} (confidence={signals['llm_confidence']})")

    print(f"\n{'=' * 80}")
    print("STRATEGIES")
    print(f"{'=' * 80}")

    strategies = ['weighted', 'cascade', 'agreement', 'disagreement']
    for strategy in strategies:
        pred, details = ensemble.predict(
            example['anchor_text'],
            example['text_a'],
            example['text_b'],
            strategy=strategy
        )

        correct = pred == example['text_a_is_closer']
        print(f"  {strategy:15s}: {'A' if pred else 'B'} {'✓' if correct else '✗'}")

    print(f"\n{'=' * 80}")


if __name__ == '__main__':
    demo_hybrid()
