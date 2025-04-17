# scnn-jax: Simplicial Complex Neural Network for Distribution System State-Estimation

This project implements a Simplicial Complex Neural Network (SCNN) for State Estimation in Transmission Systems using [JAX](https://github.com/google/jax). It leverages the rich topological structure of power grids, modeled as simplicial complexes, to improve the accuracy and robustness of state estimation models.

## Overview

Transmission systems can be naturally represented as graphs with higher-order interactions, such as node-edge and edge-triangle dependencies which are not adequately captured by traditional graph-based node-regression methods. This project extends beyond standard Graph Neural Networks (GNNs) by employing Simplicial Complexes, enabling the modeling of multi-node interactions via a Simplicial Complex Neural Network. 

Our SCNN learns to predict the states (voltage magnitude and phase angles of buses) of a power system based on measurements and pseudo-measurements as the input to the models. 

---

## 🔧 Setup

### Prerequisites

- Python 3.11
- [JAX](https://github.com/google/jax) with GPU/TPU support (optional but recommended)
- JAX libraries: `jax`, `jaxlib`
- Other dependencies:
  - `flax` (for neural network layers)
  - `optax` (for optimizers)
  - `numpy`
  - `networkx` (for topology handling)
  - `matplotlib` (optional, for visualization)
  - `jgraph` (using jax-graph NN methods)

### Installation

1. Clone this repository:

```bash
git clone https://github.com/prajapati-incontrol/scnn-jax.git
cd scnn-jax
```


2. Set up a virtual environment (recommended):


```bash
python -m venv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```
---

## 📁 Project Structure

```
scnn-jax/
├── data/               # Sample transmission system data and graphs
├── src/                # Source files 
       ├── dataset/     # Custom Dataset Object
       ├── model/       # SCNN models 
       ├── training/    # Trainer functions 
├── utils/              # Topology and data processing utilities
├── main.py             # Orchestrate everything
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## 🚀 Getting Started

To train the SCNN on your dataset:

```bash
python main.py --config configs/train_config.yaml
```

To evaluate a trained model:

```bash
python evaluate.py --model-checkpoint checkpoints/model.pt
```

---

## 📊 Results

The SCNN model demonstrates improved accuracy and generalization on benchmark transmission system datasets, outperforming traditional GNN-based and physics-based estimators under partial observability and noisy measurements.

Performance metrics:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Convergence Rate

(Include benchmark results and charts if available.)

---

## 📘 References

- Abur, A., & Exposito, A. G. *Power System State Estimation: Theory and Implementation*, CRC Press.

- Yang, M., Isufi, E. and Leus, G., 2022, May. Simplicial convolutional neural networks. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 8847-8851). IEEE.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, fork the repo, and create pull requests.

---

