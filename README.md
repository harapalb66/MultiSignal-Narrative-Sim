# MultiSignal-Narrative-Sim

This repository contains the source code for our submission to **SemEval 2026 Task 4 (Track A): Narrative Story Similarity and Narrative Representation Learning**.

Our architecture utilizes a 4-Signal Neuro-Symbolic ensemble, combining:
1. **Action-Focused Neural Embeddings** (Semantic gravity)
2. **Structural Survival Ratio (SSR)** (Symbolic logic mapping via dependencies)
3. **LLM-Based Structural Comparison** (High-level narrative abstractions)
4. **Narrative Phase-Space Dynamics** (Continuous trajectory modeling, Lyapunov exponents, and bifurcation signatures)

This approach ensures narrative similarity is measured not just through static semantic overlap, but via the temporal shape and structural causality of the stories themselves. Our dynamic trajectory mapping allowed the system to achieve an accuracy of **68.25%** by serving as a sub-symbolic disambiguation filter during low-confidence genre overlaps.

## Project Structure

This repository is strictly curated. All exploratory scripts, initial experiments, and ablation tests have been removed to ensure the codebase remains completely focused on reproducing the **0.6825** accuracy, 4-signal pipeline described in our final paper.

The core Python logic is organized as follows:

- `main.py`: The unified entry point for running the pipeline (training, predicting, extracting).
- **`/core/`**: Houses the mathematical heartbeat of the ensemble.
  - `ensemble.py`, `hybrid_ensemble.py`, `simple_ensemble.py`: Logic for combining multiple signals.
  - `narrative_attractor_dynamics.py`, `narative_dynamic.py`: Implementation of Signal 4 (Complexity Science features, Attractor Basins, Lyapunov Exponents).
- **`/utils/`**: Helper modules for data extraction and feature engineering.
  - `narrative_structure_extractor.py`, `structure_comparator.py`: Abstraction and LLM-structural representations (Signal 3).
  - `pos_based_chunking.py`, `basin_features.py`: Utilities for chunk extraction and semantic trajectory mapping.
  - `analyze_dynamics.py`, `visualize_trajectories.py`: Tools for inspecting and graphing the semantic trajectories.
- **`/scripts/`**: Executable scripts for specific tasks.
  - `train_hybrid_model.py`, `predict_with_attractors.py`, `generate_submission*.py`: Scripts used to format results and run evaluations.

*Note: For security reasons, any OpenAI API keys used for Signal 3 have been redacted. You must set your own `os.environ['OPENAI_API_KEY']` prior to running the LLM structural endpoints.*

## Setup

First, initialize your environment and install the required dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

## Usage

You can run the pipeline from the root directory using the `main.py` entry point:

```bash
# Example syntax:
python main.py evaluate --input data/dev_track_a.jsonl --model models/hybrid_model.joblib
```

For direct evaluation of the complete hybrid model on the dev set, you can also target the scripts directly:
```bash
python scripts/train_hybrid_model.py eval \
  --input_path data/dev_track_a.jsonl \
  --model_path models/hybrid_model.joblib
```
