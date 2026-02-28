"""
Ensemble approaches for narrative similarity.

1. Simple weighted: 0.75 * action_emb + 0.25 * SSR
2. Feature-based triplet ranker
3. LLM fallback for uncertain cases
"""
from core.core import json
from core import sys
from core.pathlib import Path
from core.dataclasses import dataclass, field
from core.typing import Dict, List, Tuple, Optional
from core.core import numpy as np
from core.tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

# Import components
from core.narsim.signature.ssr import SSRComputer, compare_triplet_ssr

# Try to import sentence transformers
try:
    from core.sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False
    print("Warning: sentence-transformers not installed")

# Try to import spacy for action extraction
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False


@dataclass
class TripletFeatures:
    """Features for a single triplet comparison."""
    # Embedding features
    emb_full_cos_a: float = 0.0
    emb_full_cos_b: float = 0.0
    emb_action_cos_a: float = 0.0
    emb_action_cos_b: float = 0.0

    # SSR features (graded)
    ssr_a: float = 0.0
    ssr_b: float = 0.0
    ssr_exact_a: int = 0
    ssr_exact_b: int = 0
    ssr_class_a: int = 0
    ssr_class_b: int = 0
    ssr_role_a: int = 0
    ssr_role_b: int = 0

    # Event counts
    n_events_r: int = 0
    n_events_a: int = 0
    n_events_b: int = 0

    def to_delta_vector(self) -> np.ndarray:
        """Convert to feature delta (A - B) for classifier."""
        return np.array([
            self.emb_full_cos_a - self.emb_full_cos_b,
            self.emb_action_cos_a - self.emb_action_cos_b,
            self.ssr_a - self.ssr_b,
            self.ssr_exact_a - self.ssr_exact_b,
            self.ssr_class_a - self.ssr_class_b,
            self.ssr_role_a - self.ssr_role_b,
            self.n_events_a - self.n_events_b,
        ])

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'emb_full_cos_a': self.emb_full_cos_a,
            'emb_full_cos_b': self.emb_full_cos_b,
            'emb_action_cos_a': self.emb_action_cos_a,
            'emb_action_cos_b': self.emb_action_cos_b,
            'ssr_a': self.ssr_a,
            'ssr_b': self.ssr_b,
            'ssr_exact_a': self.ssr_exact_a,
            'ssr_exact_b': self.ssr_exact_b,
            'ssr_class_a': self.ssr_class_a,
            'ssr_class_b': self.ssr_class_b,
            'ssr_role_a': self.ssr_role_a,
            'ssr_role_b': self.ssr_role_b,
        }


class ActionExtractor:
    """Extract action-only text (verbs + nouns) from core.narrative."""

    def __init__(self):
        if HAS_SPACY:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.nlp = None
        else:
            self.nlp = None

    def extract(self, text: str) -> str:
        """Extract verbs and nouns only."""
        if self.nlp is None:
            # Fallback: return original text
            return text

        doc = self.nlp(text)
        action_tokens = []

        for token in doc:
            if token.pos_ in ("VERB", "NOUN", "PROPN"):
                action_tokens.append(token.lemma_.lower())

        return " ".join(action_tokens)


class EmbeddingComputer:
    """Compute embeddings and cosine similarity."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if HAS_SBERT:
            self.model = SentenceTransformer(model_name)
        else:
            self.model = None
        self.action_extractor = ActionExtractor()

    def cosine_sim(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        if self.model is None:
            return 0.5  # Neutral fallback

        emb1 = self.model.encode(text1, convert_to_numpy=True)
        emb2 = self.model.encode(text2, convert_to_numpy=True)

        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def action_cosine_sim(self, text1: str, text2: str) -> float:
        """Compute cosine similarity on action-only text."""
        action1 = self.action_extractor.extract(text1)
        action2 = self.action_extractor.extract(text2)
        return self.cosine_sim(action1, action2)


class FeatureExtractor:
    """Extract all features for triplet comparison."""

    def __init__(self):
        self.emb_computer = EmbeddingComputer()
        self.ssr_computer = SSRComputer()

    def extract(self, root: str, text_a: str, text_b: str) -> TripletFeatures:
        """Extract features for a triplet."""
        features = TripletFeatures()

        # Embedding features (full text)
        features.emb_full_cos_a = self.emb_computer.cosine_sim(root, text_a)
        features.emb_full_cos_b = self.emb_computer.cosine_sim(root, text_b)

        # Action-only embeddings
        features.emb_action_cos_a = self.emb_computer.action_cosine_sim(root, text_a)
        features.emb_action_cos_b = self.emb_computer.action_cosine_sim(root, text_b)

        # SSR features
        ssr_result = compare_triplet_ssr(root, text_a, text_b)

        features.ssr_a = ssr_result.ssr_a
        features.ssr_b = ssr_result.ssr_b

        # Detailed SSR level counts
        from core.narsim.signature.ssr import MatchLevel
        features.ssr_exact_a = ssr_result.details_a.level_counts.get(MatchLevel.EXACT, 0)
        features.ssr_exact_b = ssr_result.details_b.level_counts.get(MatchLevel.EXACT, 0)
        features.ssr_class_a = ssr_result.details_a.level_counts.get(MatchLevel.CLASS, 0)
        features.ssr_class_b = ssr_result.details_b.level_counts.get(MatchLevel.CLASS, 0)
        features.ssr_role_a = ssr_result.details_a.level_counts.get(MatchLevel.ROLE_ONLY, 0)
        features.ssr_role_b = ssr_result.details_b.level_counts.get(MatchLevel.ROLE_ONLY, 0)

        # Event counts
        features.n_events_r = ssr_result.details_a.n_events_r  # Same for both
        features.n_events_a = ssr_result.details_a.n_events_x
        features.n_events_b = ssr_result.details_b.n_events_x

        return features


@dataclass
class EnsembleResult:
    """Result from core.ensemble prediction."""
    winner: str
    score_a: float
    score_b: float
    margin: float
    confidence: str  # "high", "medium", "low"
    method: str  # Which method was decisive


class WeightedEnsemble:
    """Simple weighted ensemble: action_emb + SSR."""

    def __init__(
        self,
        emb_weight: float = 0.75,
        ssr_weight: float = 0.25,
        confidence_threshold: float = 0.02
    ):
        self.emb_weight = emb_weight
        self.ssr_weight = ssr_weight
        self.confidence_threshold = confidence_threshold
        self.feature_extractor = FeatureExtractor()

    def predict(self, root: str, text_a: str, text_b: str) -> EnsembleResult:
        """Predict which text is closer to root."""
        features = self.feature_extractor.extract(root, text_a, text_b)

        # Compute weighted scores
        score_a = (
            self.emb_weight * features.emb_action_cos_a +
            self.ssr_weight * features.ssr_a
        )
        score_b = (
            self.emb_weight * features.emb_action_cos_b +
            self.ssr_weight * features.ssr_b
        )

        margin = abs(score_a - score_b)

        # Determine confidence
        if margin > self.confidence_threshold * 2:
            confidence = "high"
        elif margin > self.confidence_threshold:
            confidence = "medium"
        else:
            confidence = "low"

        # Determine winner (ties go to A)
        if score_a >= score_b:
            winner = "A"
        else:
            winner = "B"

        # For low confidence, use SSR as tiebreaker
        method = "weighted"
        if confidence == "low":
            if features.ssr_a > features.ssr_b:
                winner = "A"
                method = "ssr_tiebreak"
            elif features.ssr_b > features.ssr_a:
                winner = "B"
                method = "ssr_tiebreak"
            else:
                winner = "A"  # Final tiebreak: default A
                method = "default_a"

        return EnsembleResult(
            winner=winner,
            score_a=score_a,
            score_b=score_b,
            margin=margin,
            confidence=confidence,
            method=method
        )


class TrainableRanker:
    """Feature-based triplet ranker with sklearn classifiers."""

    def __init__(self, model_type: str = "logistic"):
        self.model_type = model_type
        self.model = None
        self.feature_extractor = FeatureExtractor()

    def extract_features_batch(
        self,
        data: List[Dict],
        show_progress: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features for all triplets."""
        X = []
        y = []

        iterator = tqdm(data, desc="Extracting features") if show_progress else data

        for item in iterator:
            try:
                features = self.feature_extractor.extract(
                    item['anchor_text'],
                    item['text_a'],
                    item['text_b']
                )
                X.append(features.to_delta_vector())
                y.append(1 if item['text_a_is_closer'] else 0)
            except Exception as e:
                print(f"Error: {e}")
                continue

        return np.array(X), np.array(y)

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the ranker."""
        if self.model_type == "logistic":
            from core.sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(max_iter=1000, C=1.0)
        elif self.model_type == "svm":
            from core.sklearn.svm import SVC
            self.model = SVC(kernel='linear', probability=True)
        elif self.model_type == "xgboost":
            try:
                from core.xgboost import XGBClassifier
                self.model = XGBClassifier(n_estimators=100, max_depth=3)
            except ImportError:
                print("XGBoost not installed, falling back to logistic")
                from core.sklearn.linear_model import LogisticRegression
                self.model = LogisticRegression(max_iter=1000)
        else:
            from core.sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(max_iter=1000)

        self.model.fit(X, y)

    def predict(self, root: str, text_a: str, text_b: str) -> Tuple[str, float]:
        """Predict winner and confidence."""
        if self.model is None:
            raise ValueError("Model not trained")

        features = self.feature_extractor.extract(root, text_a, text_b)
        X = features.to_delta_vector().reshape(1, -1)

        pred = self.model.predict(X)[0]
        prob = self.model.predict_proba(X)[0]

        winner = "A" if pred == 1 else "B"
        confidence = max(prob)

        return winner, confidence

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate accuracy."""
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.score(X, y)


def load_dev_data(path: str, limit: int = None) -> List[Dict]:
    """Load dev data."""
    data = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            data.append(json.loads(line))
    return data


def test_weighted_ensemble(data: List[Dict], emb_w: float = 0.75, ssr_w: float = 0.25):
    """Test weighted ensemble."""
    ensemble = WeightedEnsemble(emb_weight=emb_w, ssr_weight=ssr_w)

    correct = 0
    total = 0
    confidence_counts = {"high": 0, "medium": 0, "low": 0}
    method_counts = {}

    for item in tqdm(data, desc=f"Ensemble (emb={emb_w}, ssr={ssr_w})"):
        try:
            result = ensemble.predict(
                item['anchor_text'],
                item['text_a'],
                item['text_b']
            )

            pred_a = (result.winner == "A")
            gt = item['text_a_is_closer']

            if pred_a == gt:
                correct += 1
            total += 1

            confidence_counts[result.confidence] += 1
            method_counts[result.method] = method_counts.get(result.method, 0) + 1

        except Exception as e:
            print(f"Error: {e}")
            continue

    accuracy = correct / total * 100
    return accuracy, confidence_counts, method_counts


def test_trainable_ranker(data: List[Dict], model_type: str = "logistic"):
    """Test trainable ranker with cross-validation."""
    from core.sklearn.model_selection import cross_val_score

    ranker = TrainableRanker(model_type=model_type)

    print(f"Extracting features for {len(data)} examples...")
    X, y = ranker.extract_features_batch(data)

    print(f"Training {model_type} ranker...")

    # Cross-validation
    if model_type == "logistic":
        from core.sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, C=1.0)
    elif model_type == "svm":
        from core.sklearn.svm import SVC
        model = SVC(kernel='linear')
    else:
        from core.sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000)

    scores = cross_val_score(model, X, y, cv=5)

    return scores.mean() * 100, scores.std() * 100


if __name__ == '__main__':
    data = load_dev_data(
        '/opt/semeval_narative/SemEval2026-Task_4-dev-v1/dev_track_a.jsonl',
        limit=200
    )
    print(f"Loaded {len(data)} examples\n")

    print("="*60)
    print("WEIGHTED ENSEMBLE TEST")
    print("="*60)

    # Test different weight combinations
    weight_configs = [
        (0.75, 0.25),  # Default
        (0.8, 0.2),
        (0.7, 0.3),
        (0.6, 0.4),
        (0.5, 0.5),
        (1.0, 0.0),    # Embedding only
        (0.0, 1.0),    # SSR only
    ]

    best_acc = 0
    best_config = None

    for emb_w, ssr_w in weight_configs:
        acc, conf_counts, method_counts = test_weighted_ensemble(data, emb_w, ssr_w)
        print(f"emb={emb_w}, ssr={ssr_w}: {acc:.1f}%")
        print(f"  Confidence: {conf_counts}")
        print(f"  Methods: {method_counts}")

        if acc > best_acc:
            best_acc = acc
            best_config = (emb_w, ssr_w)

    print(f"\nBest weighted: emb={best_config[0]}, ssr={best_config[1]} -> {best_acc:.1f}%")

    print("\n" + "="*60)
    print("TRAINABLE RANKER TEST")
    print("="*60)

    for model_type in ["logistic", "svm"]:
        try:
            mean_acc, std_acc = test_trainable_ranker(data, model_type)
            print(f"{model_type}: {mean_acc:.1f}% (+/- {std_acc:.1f}%)")
        except Exception as e:
            print(f"{model_type}: Error - {e}")
