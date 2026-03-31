# LLUMYS - README

![LLUMYS_v2.png](LLUMYS_v2.png)

# LLUMYS: Learning Loss informed Uncertainty Aware MLIP from Yonsei and Sydney

***LLUMYS*** is an E(3)-equivariant graph neural network interatomic potential that learns per-atom uncertainty based on the [learning loss](https://arxiv.org/abs/1905.03677) concept. It has been developed as a companion package to ***NARA*** (a Firefly algorithm based atomic structure optimization framework). The name ***LLUMYS*** originates from *lumen*, meaning “light”, symbolizing the glow of a firefly.

## Prerequisites

- Python ≥ 3.10
- ASE ≥ 3.26.0

> Earlier versions are compatible, but 3.26.0 or later is recommended because optimizer.irun now explicitly includes gradient evaluation/log
> 
- e3nn
- pytorch ≥ 2.4

> Please follow the installation guide on the official [PyTorch website](https://pytorch.org/get-started/locally/)
> 
- (optional) openequivariance, GCC≥9

> For additional GPU acceleration. Requires installation of the matching CUDA toolkit for your PyTorch and CUDA driver versions.
> 

## Installation

### Using PIP

You can install LLUMYS directly from PyPI:

```bash
pip install llumys
```

→ To use OpenEquivariance, install openequivariance package first, matching with your GPU environment ($pip install openequivariance)

## Usage

### (1) Import LLUMYS (choose one based on your case)

```python
## a. For import default GNN model
from llumys.gnn import EquivariantGNN
from llumys.train import main_GNN

## b. For import learning-loss model
from llumys.gnn_LL import EquivariantGNN_UC
from llumys.train import main_UC

## c. For import OpenEquivariance-implemented versions of default GNN model
from llumys.gnn_oeq import EquivariantGNN
from llumys.train_oeq import main_GNN

## d. For import OpenEquivariance-implemented versions of LL model
from llumys.gnn_LL_oeq import EquivariantGNN_UC
from llumys.train_oeq import main_UC
```

### (2) Train the model

```python
# Setting
device = "cuda" # or mps or cpu

### tag-out parameters will be set as default
gnn_config = dict(
    r_cut = 5.0, # radius cutoff
    irreps_hidden = "128x0e+128x1o", # low option. -> '256x0e' (invariant) or '128x0e+128x1o' or '32x0e+32x0o+32x1e+32x1o' are usually used for real cases
    dtype = "float64", # float64 is overally recommended for optimization, float32 could be enough for MD simulations    
    # irreps_out_hidden = "16x0e",
    # node_embedding_dim = 32,
    # edge_n_basis = 8,
    # n_invariant_neurons = 8, # number of hidden neurons in radial function, smaller is faster [from nequip documentatoin]
    # n_invariant_layers = 1, # number of radial layers, Usually 1-3 works best, smaller is faster [from nequip documentatoin]
    # n_message_passing = 3,
    # max_grad_norm = 10,
    )

# Training
## for a or c -> use main_GNN
main = main_GNN
## for b or d -> use main_UC
main = main_UC

main(
    xyz_filename = "[your_train_dataset_filename.xyz]",
    valid_xyz = "[your_valid_dataset_filename.xyz]", # or use train_ratio for splitting the xyz
    device = device,
    batch_size = 5, # when including forces, 1-5 will be more appropriate.
    valid_batch_size = 10, # higher the faster. set this below OOM
    max_epoch = 1000,
    patience = 50, # to not use early_stopping, set patience=None 
    best_model_filename = "[filename-for-best-model]",
    loss_filename = "[filename-for-losses]",
    gnn_config = gnn_config,
)
```

### (3) Usage: using LLUMYS as an ASE calculator

```python
# import GNN wrapper
## a. default GNN model
from llumys.ase_wrapper import GNNWrapper
## b. LL model
from llumys.ase_wrapper import GNNUCWrapper
## c. default GNN model - OEQ
from llumys.ase_wrapper_oeq import GNNWrapper
## d. LL model - OEQ
from llumys.ase_wrapper_oeq import GNNUCWrapper

# loading the best-model.pt
## for a or c
WRAPPER = GNNWrapper
## for b or d
WRAPPER = GNNUCWrapper

ase_calc = WRAPPER(gnn_model="[filename-for-best-model.pt]", device=device)
```

→ For more detailed information, please see Tutorials

## Acknowledgement

*TBD*

## License Information

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Troubleshooting

1. CUDA error: invalid device ordinal

```
RuntimeError: CUDA error: invalid device ordinal
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

→ This error occurs when multiple GPUs exist on a node but the user explicitly selects a non-allocated GPU (ex.  `device=torch.device("cuda:2")`). In such cases, specify the target GPU when running the Python script: 

`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python [your-script-name.py]`

1. ImportError when `import openequivariance`

```
ImportError: Unable to load OpenEquivariance extension library: CUDA_HOME environment variable is not set. Please set it to your CUDA install root.
```

Check if `nvcc --version` runs properly.

→ If it doesn’t work, you may need to reinstall CUDA toolkit.

→ If it does work, add the following line to your .bashrc or job script to make sure OpenEquivariance recognizes CUDA:

 `export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"`

1. GCC version error: Please use GCC 9 or later

```
#error "You're trying to build PyTorch with a too old version of GCC. We need GCC 9 or later."
```

Check the gcc versions with `gcc -v`

1. Import hangs with no error message

If a Python script hangs during import without showing any error message, first check whether the issue is related to the OpenEquivariance JIT extension by running:

```bash
$ python -c "import openequivariance"
```

If this also hangs without any error message, the problem may be caused by the PyTorch extensions cache. In this case, set `TORCH_EXTENSIONS_DIR` to a job-local or user-defined directory before running the script.

For example:

```bash
export TORCH_EXTENSIONS_DIR=$PBS_JOBFS/torch_extensions

mkdir -p "$TORCH_EXTENSIONS_DIR"
```

This forces PyTorch to build extensions in the specified directory instead of reusing the default cache path, which can help avoid stale lock or shared-cache issues on HPC systems.
