# narrative_ranker.py
# -*- coding: utf-8 -*-
"""
Neuro-symbolic narrative similarity (anchor vs A/B) with semantic-attractor + sequentiality features.
Supports JSONL or CSV I/O (auto-detected by file extension).

Train:
  python narrative_ranker.py train --dev_path data/dev.jsonl --out_model artifacts/ranker.joblib
Predict:
  python narrative_ranker.py predict --input_path data/dev.jsonl --model artifacts/ranker.joblib --out_path out/dev_pred.jsonl
Eval (requires gold labels):
  python narrative_ranker.py eval --gold_path data/dev.jsonl --pred_path out/dev_pred.jsonl

Expected fields per record/row:
  id, anchor_text, text_a, text_b, text_a_is_closer (bool/0/1; optional on test)
"""

from core.core import argparse
import os
from core.core import warnings
warnings.filterwarnings("ignore")

from core.core import numpy as np
from core.core import pandas as pd
from core.core import regex as re
from core.sentence_transformers import SentenceTransformer, util
from core.sklearn.linear_model import LogisticRegression
from core.sklearn.preprocessing import StandardScaler
from core.sklearn.pipeline import Pipeline
from core.sklearn.model_selection import KFold
from core.sklearn.metrics import accuracy_score
from core.core import joblib


# -------------------- Utils: I/O --------------------

def _is_jsonl(path: str) -> bool:
    return path.lower().endswith(".jsonl") or path.lower().endswith(".ndjson")

def read_table(path: str) -> pd.DataFrame:
    if _is_jsonl(path):
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)

def write_table(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if _is_jsonl(path):
        df.to_json(path, orient="records", lines=True, force_ascii=False)
    else:
        df.to_csv(path, index=False)

def bool_from_any(x):
    if isinstance(x, (bool, np.bool_)): return bool(x)
    if isinstance(x, (int, np.integer)): return x != 0
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true","t","yes","y","1"}: return True
        if s in {"false","f","no","n","0"}: return False
    return None


# -------------------- Light NLP helpers --------------------

def sent_split(text: str):
    sents = re.split(r'(?<=[\.!?])\s+|\n+', (text or "").strip())
    sents = [s.strip() for s in sents if s.strip()]
    return sents if sents else [text.strip()]

def try_load_spacy():
    try:
        import spacy
        try:
            return spacy.load("en_core_web_trf")
        except Exception:
            return spacy.load("en_core_web_sm")
    except Exception:
        return None

_SPACY = try_load_spacy()

def named_entities(text: str):
    if _SPACY:
        doc = _SPACY(text or "")
        return {(e.text, e.label_) for e in doc.ents}
    toks = re.findall(r'\b[A-Z][a-z]{2,}\b', text or "")
    return {(t, "PNN") for t in set(toks)}

def verb_lemmas(text: str):
    if _SPACY:
        doc = _SPACY(text or "")
        out = []
        for sent in doc.sents:
            for tok in sent:
                if tok.pos_ == "VERB":
                    out.append(tok.lemma_.lower())
        return out
    toks = re.findall(r"\b[\w']+\b", (text or "").lower())
    return [t for t in toks if re.search(r'(ed|ing)$', t)]

def kendall_tau_like(seq_a, seq_b):
    def unique_in_order(xs):
        seen = set(); out=[]
        for x in xs:
            if x not in seen:
                seen.add(x); out.append(x)
        return out
    ua = unique_in_order(seq_a)
    ub = unique_in_order(seq_b)
    common = [x for x in ua if x in set(ub)]
    if len(common) < 2: return 0.5
    pos_a = {x:i for i,x in enumerate(ua)}
    pos_b = {x:i for i,x in enumerate(ub)}
    js = [pos_b[x] for x in common]
    inversions=0; total=0
    for i in range(len(js)):
        for j in range(i+1, len(js)):
            total += 1
            if js[i] > js[j]: inversions += 1
    if total == 0: return 1.0
    return 1.0 - (inversions/total)

def cosine(u, v):
    return float(util.cos_sim(u, v))

def dtw_distance(seq1, seq2):
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0: return 1e9
    D = np.full((n+1, m+1), np.inf, dtype=np.float32)
    D[0,0] = 0.0
    for i in range(1, n+1):
        ai = float(seq1[i-1])
        for j in range(1, m+1):
            bj = float(seq2[j-1])
            cost = abs(ai - bj)
            D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    return float(D[n,m] / (n + m))

def dtw_similarity(curve1, curve2):
    d = dtw_distance(curve1, curve2)
    return 1.0 / (1.0 + d)


# -------------------- Embeddings --------------------

class Encoder:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", device=None):
        self.model = SentenceTransformer(model_name, device=device)
    def embed_text(self, text: str):
        return self.model.encode(text or "", normalize_embeddings=True, convert_to_numpy=True)
    def embed_sents(self, sents):
        if not sents: sents = [""]
        return self.model.encode(sents, normalize_embeddings=True, convert_to_numpy=True)


# -------------------- Features --------------------

def attractor(emb_seq, weights=None):
    if len(emb_seq) == 0:
        return np.zeros((emb_seq.shape[1] if hasattr(emb_seq, "shape") and emb_seq.ndim==2 else 768,), dtype=np.float32)
    E = np.asarray(emb_seq)
    if weights is None:
        return E.mean(axis=0)
    w = np.asarray(weights).reshape(-1,1)
    w = w / (w.sum() + 1e-8)
    return (E * w).sum(axis=0)

def basin_energy(emb_seq, anchor_att):
    if len(emb_seq) == 0: return 0.0
    diffs = emb_seq - anchor_att[None,:]
    return float(np.mean(np.sum(diffs*diffs, axis=1)))

def distance_curve_to_attractor(emb_seq, anchor_att):
    if len(emb_seq) == 0: return [1.0]
    diffs = emb_seq - anchor_att[None,:]
    return list(np.sqrt(np.sum(diffs*diffs, axis=1)))

def ner_jaccard(a_text, b_text):
    A = set(named_entities(a_text))
    B = set(named_entities(b_text))
    if not A and not B: return 0.0
    return len(A & B) / float(max(1, len(A | B)))

def pair_features(enc: Encoder, anchor_text: str, other_text: str):
    a_doc = enc.embed_text(anchor_text)
    o_doc = enc.embed_text(other_text)

    a_sents = sent_split(anchor_text)
    o_sents = sent_split(other_text)
    Ea = enc.embed_sents(a_sents)
    Eo = enc.embed_sents(o_sents)

    att_a = attractor(Ea)
    att_o = attractor(Eo)

    cos_global = cosine(a_doc, o_doc)
    cos_att = cosine(att_a, att_o)
    neg_basin = -basin_energy(Eo, att_a)

    curve_a = distance_curve_to_attractor(Ea, att_a)
    curve_o = distance_curve_to_attractor(Eo, att_a)
    dtw_sim = dtw_similarity(curve_a, curve_o)

    va = verb_lemmas(anchor_text)
    vo = verb_lemmas(other_text)
    order_k = kendall_tau_like(va, vo)

    ner_overlap = ner_jaccard(anchor_text, other_text)

    return np.array([cos_global, cos_att, neg_basin, dtw_sim, order_k, ner_overlap], dtype=float)

def triplet_features(enc: Encoder, anchor_text: str, text_a: str, text_b: str):
    fa = pair_features(enc, anchor_text, text_a)
    fb = pair_features(enc, anchor_text, text_b)
    diff = fa - fb  # positive => A closer than B
    return fa, fb, diff


# -------------------- Train / Predict / Eval --------------------

def train(dev_path, out_model, encoder_name, folds=5, random_state=42):
    enc = Encoder(encoder_name)
    df = read_table(dev_path)

    if "text_a_is_closer" not in df.columns:
        raise RuntimeError("Training data must include 'text_a_is_closer'.")

    df["label_bin"] = df["text_a_is_closer"].apply(bool_from_any)
    if df["label_bin"].isna().any():
        raise RuntimeError("Could not parse some 'text_a_is_closer' values into booleans.")

    X, y = [], []
    for _, r in df.iterrows():
        fa, fb, diff = triplet_features(enc, r["anchor_text"], r["text_a"], r["text_b"])
        X.append(diff)
        y.append(1 if r["label_bin"] else 0)
    X = np.vstack(X); y = np.array(y)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced", random_state=random_state))
    ])

    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    accs = []
    for tr, va in kf.split(X):
        pipe.fit(X[tr], y[tr])
        p = pipe.predict(X[va])
        accs.append(accuracy_score(y[va], p))
    print(f"[CV] acc={np.mean(accs):.4f} ± {np.std(accs):.4f}  (n={len(y)})")

    pipe.fit(X, y)
    os.makedirs(os.path.dirname(out_model) or ".", exist_ok=True)
    joblib.dump({"pipe": pipe, "encoder_name": encoder_name}, out_model)
    print(f"[OK] saved {out_model}")

def predict(input_path, model_path, out_path):
    bundle = joblib.load(model_path)
    pipe = bundle["pipe"]
    enc = Encoder(bundle["encoder_name"])

    df = read_table(input_path)
    rows = []
    for _, r in df.iterrows():
        fa, fb, diff = triplet_features(enc, r["anchor_text"], r["text_a"], r["text_b"])
        prob_A = float(pipe.predict_proba(diff.reshape(1,-1))[0,1])  # P(A closer)
        pred_is_A = prob_A >= 0.5
        rows.append({
            "id": r.get("id", _),
            "pred_text_a_is_closer": bool(pred_is_A),
            "prob_text_a": prob_A,
            "prob_text_b": 1.0 - prob_A,
            "cos_global_AB": float(fa[0]),
            "cos_global_AC": float(fb[0]),
            "cos_att_AB": float(fa[1]),
            "cos_att_AC": float(fb[1]),
            "neg_basin_AB": float(fa[2]),
            "neg_basin_AC": float(fb[2]),
            "dtw_sim_AB": float(fa[3]),
            "dtw_sim_AC": float(fb[3]),
            "order_k_AB": float(fa[4]),
            "order_k_AC": float(fb[4]),
            "ner_overlap_AB": float(fa[5]),
            "ner_overlap_AC": float(fb[5]),
        })
    out = pd.DataFrame(rows)
    write_table(out, out_path)
    print(f"[OK] wrote {out_path}")

def eval_dev(gold_path, pred_path):
    gold = read_table(gold_path)
    pred = read_table(pred_path)

    if "id" not in gold.columns:
        # create deterministic ids if missing in gold
        gold = gold.copy()
        gold["id"] = list(range(len(gold)))

    m = pd.merge(gold, pred, on="id", how="inner")
    if "text_a_is_closer" not in m.columns:
        raise RuntimeError("gold data must have 'text_a_is_closer' column.")

    gold_y = m["text_a_is_closer"].apply(bool_from_any).astype(int).to_numpy()
    pred_y = m["pred_text_a_is_closer"].astype(bool).astype(int).to_numpy()
    acc = accuracy_score(gold_y, pred_y)
    print(f"Accuracy: {acc:.4f}  (N={len(m)})")


# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_tr = sub.add_parser("train")
    ap_tr.add_argument("--dev_path", required=True)
    ap_tr.add_argument("--out_model", default="artifacts/ranker.joblib")
    ap_tr.add_argument("--encoder_name", default="sentence-transformers/all-mpnet-base-v2")
    ap_tr.add_argument("--folds", type=int, default=5)

    ap_pr = sub.add_parser("predict")
    ap_pr.add_argument("--input_path", required=True)
    ap_pr.add_argument("--model", required=True)
    ap_pr.add_argument("--out_path", default="out/pred.jsonl")

    ap_ev = sub.add_parser("eval")
    ap_ev.add_argument("--gold_path", required=True)
    ap_ev.add_argument("--pred_path", required=True)

    args = ap.parse_args()
    if args.cmd == "train":
        train(args.dev_path, args.out_model, args.encoder_name, args.folds)
    elif args.cmd == "predict":
        predict(args.input_path, args.model, args.out_path)
    elif args.cmd == "eval":
        eval_dev(args.gold_path, args.pred_path)

if __name__ == "__main__":
    main()
