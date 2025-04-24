# JAX implementation of Distribution System State-Estimation 

This project is open-source version of a private repository for Distribution Power Sytem State Estimation using [JAX](https://github.com/google/jax). It leverages the rich topological structure of power grids, modeled as simplicial complexes, to improve the accuracy and robustness of state estimation models. 

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
  - `torch-geometric` (for graph data object and dataloaders)

### Installation

1. Clone this repository:

```bash
git clone https://github.com/prajapati-incontrol/gd4ps-jax.git
cd gd4ps-jax
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
├── log/                    # log files
       ├── script_log.txt
├── data/                   # Sample transmission system data and graphs
├── src/                    # Source files 
       ├── dataset/         # Custom Dataset Object
       ├── model/           # SCNN models 
       ├── training/        # Trainer functions 
├── utils.py                # Topology and data processing utilities
├── main.py                 # Orchestrate everything
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---


## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, fork the repo, and create pull requests.

---

