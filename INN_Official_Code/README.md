# Intelligent Neural Networks (INN)

**Official Implementation** for the paper: *"Intelligent Neural Networks: Neural Graph Topology with Internal State and Communicative Attention"*.

![Architecture](https://via.placeholder.com/800x300?text=Concept+Architecture+Diagram)

## 🧠 Abstract

We propose the **Intelligent Neural Network (INN)**, a paradigm shift from layered transformations to a **graph of communicating neurons**.
In an INN:
1.  **Neurons are First-Class Entities:** Each neuron has internal memory (state) modeled by a Selective State Space Model (**Mamba**).
2.  **Communication is Dynamic:** Neurons exchange information via a learned attention mechanism (Multi-Head Attention) within a complete graph topology.

This architecture achieves competitive performance with LSTMs and Transformers on character-level language modeling tasks (Text8, WikiText-2) while using significantly fewer parameters.

## 📊 Key Results

| Dataset | Model | Params | Metric | Result |
| :--- | :--- | :--- | :--- | :--- |
| **Text8** | Transformer | 4.7M | BPC | 2.055 |
| | LSTM | 6.3M | BPC | 1.682 |
| | **INN (Ours)** | **4.2M** | **BPC** | **1.705** |
| **WikiText-2** | Transformer | 4.5M | PPL | 3.49 |
| | LSTM | 5.1M | PPL | 3.57 |
| | **INN (Ours)** | **4.5M** | **PPL** | **3.61** |

## 🚀 Installation

```bash
git clone https://github.com/yourusername/INN.git
cd INN
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- `mamba-ssm` (CUDA required)
- `causal-conv1d`
- `datasets`
- `matplotlib`, `seaborn`

## 🏃 Usage

### 1. Train on WikiText-2
To reproduce the results on WikiText-2 character-level modeling:

```bash
python benchmarks/run_wikitext.py --epochs 20 --batch_size 16
```

### 2. Visualize Neuron Communication
After training, visualize the attention heatmap (neuron-to-neuron communication):

```bash
python visualizations/plot_attention.py --model_path best_inn_wikitext.pth
```

### 3. Run Mamba Baseline
To verify that the graph topology adds value over a pure State-Space Model:

```bash
python benchmarks/run_mamba_baseline.py
```

## 📂 Structure

- `src/model.py`: Core INN architecture definition.
- `benchmarks/`: Scripts to run training and evaluation.
- `visualizations/`: Tools to generate paper figures.

## 📝 Citation

If you use this code, please cite:

```bibtex
@article{inn2024,
  title={Intelligent Neural Networks},
  author={Salomon, Antoine},
  journal={ArXiv Preprint},
  year={2024}
}
```

