# INN Research Paper: Project Dashboard

## Title: "Intelligent Neural Networks for Language Modeling: An Analysis of Capabilities and Limitations"

### 1. Abstract
- **Goal**: Evaluate a biologically-inspired architecture (INNv2) where neurons are independent agents communicating via attention.
- **Key Results**: 
    - Strong convergence on Shakespeare (Char-level).
    - Validated learning on Penn TreeBank (Word-level, ~5M params).
    - Achieved Train PPL < 50 (Strong capacity), Valid PPL ~180-220 (Generalization gap).
- **Conclusion**: Architecture is viable and computationally efficient, but requires specific regularization to generalize on small datasets.

### 2. Introduction
- Concept: "Brains are colonies of independent agents" (Minsky).
- Current paradigm: Monolithic layers (Transformer/LSTM).
- Proposal: INN (Intelligent Neural Network) - Independent stateful neurons (Mamba) + sparse communication (Attention).

### 3. Architecture (INNv2)
- **Neuron Core**: Mamba Block (State Space Model) for temporal processing.
- **Communication**: Masked Multi-Head Attention (Spatial processing).
- **Topology**: Parallel layer-wise organization.
- *To Do*: Diagram of the "Neuron Cell" and the "Colony Topology".

### 4. Experiments
#### A. Main Results (PTB)
- **Config**: `d_model=256`, `n_neurons=16`, `n_layers=4`, `param_count=5.5M`.
- **Result**: Train PPL < 100, Valid PPL ~180-220 (without heavy reg).
- *To Do*: Run with Dropout=0.3 + Weight Decay=0.1 to fix overfitting.

#### B. Baselines (To Run)
We need to compare INNv2 (5.5M params) against:
1.  **LSTM Baseline**: 2-layer LSTM, same param count.
2.  **Transformer Baseline**: 4-layer Transformer, same param count.
*Goal*: Show INNv2 converges faster or reaches better PPL than LSTM, and rivals Transformer efficiency.

### 5. Analysis
- **Visualizations**:
    - *Neuron Specialization*: Heatmaps showing neurons activating on specific POS (Verbs, Nouns).
    - *Attention Patterns*: Who talks to whom?
- **Ablations (To Run)**:
    - *No-Communication*: Remove Attention layer -> Does it collapse?
    - *No-Memory*: Replace Mamba with FeedForward -> Does it fail on long seq?

### 6. Discussion
- **Strengths**: Parameter efficiency, interpretable "neuron" activity.
- **Weaknesses**: Tendency to overfit (high capacity), sensitivity to hyperparams.
- **Future**: Hierarchical INN (Modular Brain), Multi-modal agents.

---

## Repository Structure
- `code/`: Training scripts (INN, Baselines, Ablations).
- `experiments/`: Logs and checkpoints.
- `visualizations/`: Generated heatmaps and plots.
- `latex/`: Draft of the paper.

