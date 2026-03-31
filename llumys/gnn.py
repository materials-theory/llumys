# coding: utf-8

# This file includes portions of code adapted from NequIP (Message Passing class)
# (https://github.com/mir-group/nequip)
# Copyright (c) 2021 NequIP authors
# Licensed under the MIT License

__name__ = "LLUMYS"
__author__ = "Giyeok Lee"
__email__ = "giyeok.lee@sydney.edu.au"
__date__ = "Oct 23, 2025"
__maintainer__ = "Giyeok Lee"
__version__ = "1.0.0"
# __copyright__ = "Copyright (c) Materials Theory Group @ Yonsei University (2025)"


### Import and warnings settings
from llumys.distance import *
import os, time, pickle, warnings, re
import numpy as np
from typing import List, Optional, Any, Dict

from ase import Atoms
from ase.io.extxyz import read_xyz
from ase.data import atomic_numbers
# from ase.geometry import get_distances # 이거 말고 get_distances_torch 사용할 거임

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
# from torch_scatter import scatter

### If kernel keeps dying, you can try this
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

### 1. From e3nn.utils.jit, to remove the TorchScript warnings
warnings.filterwarnings(
    "ignore",
    message=re.escape(
        "The TorchScript type system doesn't support instance-level annotations on empty non-base types "
        "in `__init__`. Instead, either 1) use a type annotation in the class body, or 2) wrap the type "
        "in `torch.jit.Attribute`."
    ),
    category=UserWarning,
    module="torch",
)

### 2. It seems.. e3nn.o3._wigner using load with wegihts_only=False
# This is also shown in MACE, NequIP, and so on
warnings.filterwarnings(
    "ignore",
    message=re.escape(
        "You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. "
        "It is possible to construct malicious pickle data which will execute arbitrary code during unpickling "
        "(See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, "
        "the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during "
        "unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by "
        "the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where "
        "you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature."
    ),
    category=FutureWarning,
    # module="torch",
)

from e3nn.o3 import spherical_harmonics, Irrep, Irreps, Linear, TensorProduct, FullyConnectedTensorProduct
from e3nn.nn import FullyConnectedNet, Gate

### Settings done

_eps = 1e-8
act_for_eq = {1:F.silu, -1:torch.tanh} # F.tanh will be deprecated

def print_memory_usage():
    import psutil
    # Check CPU memory usage
    process = psutil.Process(os.getpid())  # bring current process ID현
    cpu_mem = process.memory_info().rss  # Resident Set Size: Actual physical memory usage
    
    print(f"CPU Memory Used: {cpu_mem/1024**2:.2f}MB")
    print(f"CPU Memory Percentage: {process.memory_percent():.1f}%")
    
    # Check GPU memory usage
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved()/1024**2:.2f}MB")

def check_tp_path(irreps_node, irreps_edge, ir_out):
    '''
    check path for tensor product


    Adapted from NequIP, which is available under MIT license:
    The MIT License (MIT) Copyright (c)
    2021 The President and Fellows of Harvard College
    '''
    irreps_node = Irreps(irreps_node).simplify()
    irreps_edge = Irreps(irreps_edge).simplify()
    ir_out      = Irrep(ir_out) # Caution. just Irrep, not Irreps which contains the n_channel

    for _, ir_n in irreps_node:
        for _, ir_e in irreps_edge:
            if ir_out in ir_n*ir_e:
                return True
    return False


class RadialFunction(nn.Module):
    r_cut: float
    p: int
    b_pi: nn.Parameter
    n_basis: int
    def __init__(self, r_cut, n_basis=8, p=6):
        super().__init__()
        self.r_cut = r_cut
        self.p = p
        self.b_pi = nn.Parameter(torch.linspace(1, n_basis, n_basis)*torch.pi)
        self.n_basis = n_basis

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        r: [E], edge distance
        return: [E, n_basis], R(r_ij)에 해당하는 raw radial basis.
        """
        # sin_term = sin(b*pi*(r/r_cut))/r shape
        sin_term = torch.sin(self.b_pi.view(1, -1)*r.unsqueeze(-1)/self.r_cut) / r.unsqueeze(-1)

        # normalized distance
        rij = r / self.r_cut
        p = self.p

        # polynomial cutoff u(r)
        u_rij = 1.0 - (p+1.0)*(p+2.0)/2.0 * torch.pow(rij, p) \
                + p*(p+2.0)*torch.pow(rij, p+1.0) \
                - p*(p+1.0)/2.0*torch.pow(rij, p+2.0)
        # applying cutoff: 0 if r > r_cut
        u_rij *= (rij < 1.0)

        # basis_values: [E, n_basis]
        # (2.0/self.r_cut) factor: plz refer NequIP's method
        basis_values = (2.0/self.r_cut) * sin_term * u_rij.unsqueeze(-1)

        # 0 if r=0
        basis_values = torch.where(r.unsqueeze(-1)>0, basis_values, torch.zeros_like(basis_values))

        return basis_values  # shape: (E, n_basis)

def compute_Ylm(vec_ij, max_l=1):
    '''
    Compute spherical harnomics Ylm(unit vector of r_ij) for a given vector, vec_ij.

   :param vec_ij (torch.Tensor): [N_edges, 3], vectors.
   :param max_l (int): Maximum degree of spherical harmonics

    [Return]
    torch.Tensor: [N_edges, irreps_Ylm], irreps representing Ylm
    '''
    norm = vec_ij.norm(dim=-1, keepdim=True)
    # Should we add the filter process to avoid division by zero? I never experienced this.
    # norm[norm < _eps] = _eps 혹은, 아래 방법이 나을지도
    # norm = torch.clamp(norm, min=_eps)
    unit_vec = vec_ij/norm
    Y = spherical_harmonics([l for l in range(max_l+1)], unit_vec, normalize=True) # TODO: we may can turn off the normalize here. check again and tag off the normalization here
    return Y

class ASE_Dataset_EF(Dataset):
    data: List[Dict[str, Any]]
    dt: torch.dtype
    type_map: Dict[int, int]
    atom_energies: torch.Tensor
    apply_constraint: bool
    avg_num_neighbors: float

    def __init__(
        self,
        atoms_list:List[Atoms],
        r_cut:float,
        dtype:str = "float64",
        type_map:Dict[int, int] = None,
        input_atom_energies:Dict[str, float] = None, # If not provided, linear regression is performed
        return_atom_energies: bool = False,
        apply_constraint:bool = False # When constraints are applied to ase Atoms like FixedAtoms, certain atoms' force can be (0,0,0) when use atoms.get_forces method
        ):

        self.data = []
        if isinstance(dtype, torch.dtype):
            self.dt = dtype
        elif isinstance(dtype, str):
            self.dt = torch.float64 if dtype.lower()=="float64" else torch.float32
        else:
            raise IOError(f"What is this dtype?: {dtype}")

        self.apply_constraint = apply_constraint

        if type_map is None:
            type_map = {}

        if input_atom_energies is None:
            input_atom_energies = {}
        else:
            assert isinstance(input_atom_energies, dict), "input_atom_energies should be a dictionary"
            
        if len(type_map)==0:
            unique_numbers = set()
            for atoms in atoms_list:
                unique_numbers.update(atoms.get_atomic_numbers())
            unique_numbers = sorted(unique_numbers)
            type_map = {z: i for i, z in enumerate(unique_numbers)}
        else:
            assert isinstance(type_map, dict), "type_map should be a dictionary"
            type_map = dict(sorted(type_map.items())) # sorting keys

        self.type_map = type_map

        total_edges = 0
        total_atoms = 0

        # for linear regression
        n_types = max(type_map.values())+1
        x_counts = []
        y_energies = []

        for atoms in atoms_list:
            if "energy" in atoms.info:
                energy = atoms.info["energy"]
            else:
                energy = atoms.get_potential_energy()

            if "forces" in atoms.arrays:
                forces = atoms.arrays["forces"]
            else:
                forces = atoms.get_forces(apply_constraint=self.apply_constraint)

            model = atoms.copy()
            model.wrap() # This is better for retrieving COM consistently
            pos = torch.tensor(model.get_positions(), dtype=self.dt)
            atom_types = torch.tensor([type_map[an] for an in model.get_atomic_numbers()], dtype=torch.long)

            # for linear regression
            x_counts.append(torch.bincount(atom_types, minlength=n_types).to(dtype=self.dt))
            y_energies.append(energy)

            cell = torch.tensor(model.cell.array, dtype=self.dt)
            pbc = torch.tensor(model.pbc, dtype=torch.bool)

            edge_i, edge_j, Vecs, Ds, edge_cell_shift = get_distances_torch(
                pos,
                cell=cell,
                return_all_neighborlist=True,
                r_cut_for_neighbor = r_cut,
                return_ALL = False
                )

            total_edges += len(edge_i)
            total_atoms += len(pos)

            self.data.append(dict(
                pos = pos,
                atom_types = atom_types,
                cell = cell,
                edge_cell_shift = edge_cell_shift,
                edge_i = edge_i,
                edge_j = edge_j,
                # "edge_vecs": vec_ij, # calculate this later. if we calculate now, autograd's gradient flow is broken
                energy = energy, # this is not torch.tensor yet
                forces = torch.tensor(forces, dtype=self.dt),
            ))
        self.avg_num_neighbors = total_edges/total_atoms if total_atoms > 0 else None

        self.atom_energies = torch.zeros(n_types, dtype=self.dt)
        if return_atom_energies:
            if len(input_atom_energies) != 0:
                if isinstance(list(input_atom_energies)[0], int):
                    if set(type_map.keys())==set(input_atom_energies):
                        # key of input_atom_energies is atomic_numbers
                        for k, v in input_atom_energies.items():
                            self.atom_energies[self.type_map[k]]=v

                    elif set(type_map.values())==set(input_atom_energies):
                        # key of input_atom_energies is atom_type (index by type_map)
                        for k, v in input_atom_energies.items():
                            self.atom_energies[k] = v

                    else:
                        raise IOError(f"wrong input of input_atom_energies: {input_atom_energies}")

                elif isinstance(list(input_atom_energies)[0], str):
                    _elements = list(input_atom_energies)
                    for _ele in _elements:
                        if isinstance(_ele, str):
                            _num = atomic_numbers[_ele]
                        self.atom_energies[self.type_map[_num]] = input_atom_energies[_ele]

                else:
                    raise IOError(f"Unknown type of atom_energies: {type(list(input_atom_energies)[0])}")

            else:
                # perform linear regression
                _X = torch.stack(x_counts)
                _Y = torch.tensor(y_energies, dtype=self.dt)

                try:
                    _res = torch.linalg.lstsq(_X, _Y, driver="gels") # gels: only valid driver in CUDA, use QR decomposition
                    self.atom_energies = _res.solution
                except Exception as e:
                    print(f"Failed to compute atom energies from Linear regression: {e}")
                    print(f"Using Zero energies instead")
                    self.atom_energies = torch.zeros(n_types, dtype=self.dt)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iadd__(self, other):
        # in-place add
        self.data += other.data
        return self
    
def collate_fn(batch, single_input=False, include_EF=True):
    if single_input:
        data = batch[0].copy()
        if ("energy" in data) and include_EF:
            data["energy"] = torch.tensor([data["energy"]], dtype=data["pos"].dtype)
        data["cell"] = data["cell"].unsqueeze(0)
        N_nodes = len(data["pos"])
        N_edges = len(data["edge_cell_shift"])
        data["batch_nodes"] = torch.full((N_nodes,), 0, dtype=torch.long)
        data["batch_edges"] = torch.full((N_edges,), 0, dtype=torch.long)
        data["n_atoms"] = torch.tensor([N_nodes], dtype=torch.long)
        return data
    
    pos_list, types_list = [], []
    i_list, j_list = [], []
    cell_list, edge_cell_shift_list = [], []
    energy_list, forces_list = [], []
    batch_nodes_list, batch_edges_list = [], []
    n_atoms_list = []
    
    _offset = 0
    for bi, sample in enumerate(batch):
        N_nodes = len(sample["atom_types"])
        pos_list.append(sample["pos"])
        cell_list.append(sample["cell"].unsqueeze(0)) # Shape: (1,3,3) -> it will be (batch_size,3,3) after concat
        edge_cell_shift_list.append(sample["edge_cell_shift"])
        N_edges = len(sample["edge_cell_shift"])
        types_list.append(sample["atom_types"])
        if include_EF:
            energy_list.append(sample["energy"])
            forces_list.append(sample["forces"])
        n_atoms_list.append(N_nodes)

        ei = sample["edge_i"] + _offset # edge is based on atom index
        ej = sample["edge_j"] + _offset


        i_list.append(ei)
        j_list.append(ej)
        batch_nodes_list.append(torch.full((N_nodes,), bi, dtype=torch.long))
        batch_edges_list.append(torch.full((N_edges,), bi, dtype=torch.long))

        _offset += N_nodes
    
    return_dict = dict(
        pos = torch.cat(pos_list, dim=0),
        atom_types = torch.cat(types_list, dim=0),
        cell = torch.cat(cell_list, dim=0),
        edge_cell_shift = torch.cat(edge_cell_shift_list, dim=0),
        edge_i = torch.cat(i_list, dim=0),
        edge_j = torch.cat(j_list, dim=0),
        batch_nodes = torch.cat(batch_nodes_list, dim=0),
        batch_edges = torch.cat(batch_edges_list, dim=0),
        n_atoms = torch.tensor(n_atoms_list, dtype=torch.long)
    )

    if include_EF:
        return_dict["energy"] = torch.tensor(energy_list, dtype=pos_list[0].dtype)
        return_dict["forces"] = torch.cat(forces_list, dim=0)

    return return_dict


###############################################################
# Aggregation (Interaction Block > Gate)
###############################################################
class MessagePassing(nn.Module):
    '''
    Message Passing

    Adapted from NequIP, which is available under MIT license:
    The MIT License (MIT) Copyright (c)
    2021 The President and Fellows of Harvard College
    '''    
    avg_num_neighbors: Optional[float]
    use_sc: bool
    linear_1: Linear
    tp: TensorProduct
    fc: FullyConnectedNet
    linear_2: Linear
    sc: Optional[FullyConnectedTensorProduct]

    def __init__(
        self,
        irreps_node_input: str,
        irreps_node_attr: str,
        irreps_edge_attr: str,
        irreps_node_output: str,
        invariant_layers: int  = 1,
        invariant_neurons:int  = 8,
        edge_n_basis: int = 8,
        avg_num_neighbors=None,
        use_sc=True
    ):
        super().__init__()
        self.avg_num_neighbors = avg_num_neighbors
        self.use_sc = use_sc

        a = Irreps(irreps_node_input)
        b = Irreps(irreps_edge_attr)
        c = Irreps(irreps_node_output) # irreps_hidden

        irreps_scalars = Irreps([(mul, ir) for mul, ir in c if ir.l==0 and check_tp_path(a, b, ir)])
        irreps_gated = Irreps([(mul, ir) for mul, ir in c if ir.l>0 and check_tp_path(a, b, ir)])
        irreps_layer_out = (irreps_scalars + irreps_gated).simplify()

        ir = ("0e" if check_tp_path(a, b, "0e") else "0o")
        irreps_gates = Irreps([(mul, ir) for mul, _ in irreps_gated])

        self.gate = Gate(
            irreps_scalars=irreps_scalars,
            act_scalars = [act_for_eq[ir.p] for _, ir in irreps_scalars],
            irreps_gates = irreps_gates,
            act_gates = [act_for_eq[ir.p] for _, ir in irreps_gates],
            irreps_gated = irreps_gated,
            )

        node_in = a
        # ib_out = Irreps(irreps_node_output)
        node_out_ib = self.gate.irreps_in.simplify() # node out of Interaction Block. not node out of MessagePassing
        node_out_mp = self.gate.irreps_out.simplify() # node out of MessagePassing Block
        edge_attr_ir = Irreps(irreps_edge_attr)
        node_attr_ir = Irreps(irreps_node_attr)

        self.linear_1 = Linear(node_in, node_in, internal_weights = True, shared_weights = True)

        irreps_mid = []
        instructions = []
        
        for i_in, (mul_in, ir_in) in enumerate(node_in):
            for j_edge, (mul_edge, ir_edge) in enumerate(edge_attr_ir):
                for ir_out in ir_in * ir_edge:
                    k = len(irreps_mid) # TODO-STUDY: k는 단지 순서인가?
                    irreps_mid.append((mul_in, ir_out))
                    instructions.append((i_in, j_edge, k, "uvu", True))
                    
        irreps_mid = Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [(i_in1, i_in2, p[i_out], mode, train) for i_in1, i_in2, i_out, mode, train in instructions]

        # Fig. 1d / tensor product of Ylm and RadialFunction
        self.tp = TensorProduct(
            node_in,
            edge_attr_ir, # Yml
            irreps_mid,
            instructions = instructions,
            internal_weights=False,
            shared_weights=False
        )
        
        # FC which generates weight from R(r_ij)
        layer_size = [edge_n_basis] + invariant_layers*[invariant_neurons] + [self.tp.weight_numel]

        # fully connected mlp: edge_n_basis > invariant_neurons > invariant_neurons > tp.weight_numel
        self.fc = FullyConnectedNet(layer_size, act=F.silu)
        # Here, activation function (\phi) will be automatically normalized by a scaling factor such that
        # \int_{-\infty}^{\infty} \phi(z)^2 \frac{e^{-z^2 / 2}}{\sqrt{2 \pi}} d z=1

        self.linear_2 = Linear(
            irreps_in = irreps_mid.simplify(),
            irreps_out = node_out_ib,
            internal_weights = True,
            shared_weights = True,
        )

        # self-connection
        if self.use_sc:
            self.sc = FullyConnectedTensorProduct(node_in, node_attr_ir, node_out_ib)
        else:
            self.sc = None

    def forward(self, data: dict) -> dict:
        x = data["node_features"]
        node_attrs = data["node_attrs"]
        edge_index = data["edge_index"]
        edge_attrs = data["edge_attrs"]
        edge_embed = data["edge_embedding"]
        edge_i = edge_index[0]
        edge_j = edge_index[1]

        orig_x = x

        if self.sc is None:
            sc = 0
        else:
            sc = self.sc(x, node_attrs)

        x = self.linear_1(x)
        w = self.fc(edge_embed)
        x_i = x[edge_i]
        Y_ij = edge_attrs
        msgs = self.tp(x_i, Y_ij, w)

        # if self.avg_num_neighbors is not None:
        #     msgs = msgs.div(self.avg_num_neighbors**0.5)

        ### TODO-CRITICAL: 이 부분 avg_num_neighbors 전부 안 사용하게 되는건데 이거 맞는거임??
        ### 아 이거 좀 고민됨. 나누지 않는게 더 성능이 좋게 나옴. 왜 그러지??
        ### 그럼 어느 경우에 이거 나누지 않는게 문제가 될 때가 있을까??


        x_out = torch.zeros(x.size(0), msgs.size(1), device=x.device, dtype=x.dtype)
        x_out.index_add_(0, edge_j, msgs)

        # if self.avg_num_neighbors is not None:
        #     x_out = x_out.div(self.avg_num_neighbors**0.5)
        ### TODO-CRITICAL: 아니 그럼 이렇게 넣으면 다름???

        x_out = self.linear_2(x_out) + sc

        x_out = self.gate(x_out)

        if x_out.shape == orig_x.shape:
            # resnet style
            data["node_features"] = x_out + orig_x
        else:
            data["node_features"] = x_out
        return data

###############################################################
# EquivariantGNN
###############################################################
class EquivariantGNN(nn.Module):
    energy_loss_fn: nn.MSELoss

    gnn_config: Dict[str, Any]
    r_cut: float
    dt: torch.dtype
    type_map: Dict[int, int]
    atom_energies: torch.Tensor
    avg_num_neighbors: float
    node_embedding_layer: nn.Embedding
    radial: RadialFunction
    blocks: nn.ModuleList
    energy_out: Linear

    def __init__(
        self,
        type_map:dict   = None,
        gnn_config:dict = None,
        atom_energies:torch.Tensor = None,
        avg_num_neighbors:float = None,
    ):
        '''
        [Arguments in gnn_config]
        |___ irreps_hidden: str = "32x0e+32x0o+32x1e+32x1o",
        |___ node_embedding_dim: int = 32,
        |___ r_cut: float = 3.0,
        |___ edge_n_basis: int = 8,
        |___ n_invariant_neurons: int = 64, # number of hidden neurons in radial function, smaller is faster [from nequip]
        |___ n_invariant_layers: int = 2, # number of radial layers, Usually 1-3 works best, smaller is faster [from nequip]
        |___ n_message_passing: int = 3,
        |___ dtype: str = "float64",
        '''        
        super().__init__()
        self.energy_loss_fn = nn.MSELoss()

        if type_map is None:
            type_map = {}
        if gnn_config is None:
            gnn_config = {}
        
        gnn_default_config = dict(
            irreps_hidden = "32x0e+32x0o+32x1e+32x1o",
            irreps_out_hidden = "16x0e",
            node_embedding_dim = 32,
            r_cut = 5.0,        
            edge_n_basis = 8,
            n_invariant_neurons = 64,
            n_invariant_layers = 2, 
            n_message_passing = 3,
            dtype = "float64",
            max_grad_norm = None,
        )
        
        if len(gnn_config)>0:
            gnn_default_config.update(gnn_config)
        gnn_config = gnn_default_config

        self.gnn_config = gnn_config

        irreps_hidden = gnn_config["irreps_hidden"]
        irreps_out_hidden = gnn_config["irreps_out_hidden"]
        node_embedding_dim = gnn_config["node_embedding_dim"]
        r_cut = gnn_config["r_cut"]
        edge_n_basis = gnn_config["edge_n_basis"]
        n_invariant_neurons = gnn_config["n_invariant_neurons"]
        n_invariant_layers = gnn_config["n_invariant_layers"]
        n_message_passing = gnn_config["n_message_passing"]
        dtype = gnn_config["dtype"]
        if isinstance(dtype, torch.dtype):
            dtype = str(dtype).split(".")[-1]
            self.gnn_config["dtype"] = dtype
        
        self.r_cut = r_cut
        self.dt = torch.float64 if dtype=="float64" else torch.float32
        
        if len(type_map)==0:
            raise IOError("Type_map should be a dict, {element_type : atom_type}")
        
        self.type_map = type_map
        unique_elements = sorted(list(type_map))
        n_elements = len(unique_elements)
        self.n_elements = n_elements

        if atom_energies is None:
            atom_energies = torch.zeros(max(type_map.values())+1, dtype=self.dt)
        self.register_buffer("atom_energies", atom_energies)
        # |___ instead of self.atom_energies = atom_energies, this enables automatic transfer with other model parameters

        self.node_embedding_layer = nn.Embedding(n_elements, node_embedding_dim)
        with torch.no_grad():
            nn.init.normal_(self.node_embedding_layer.weight, mean=0.0, std=0.1)

        # RadialFunction에서 b_pi를 learnable하게 설정
        self.radial = RadialFunction(
            r_cut=self.r_cut,
            n_basis=edge_n_basis,
            p=6
        )

        self.avg_num_neighbors = avg_num_neighbors

        irreps_node_in = f"{node_embedding_dim}x0e"
        irreps_node_attr = f"{n_elements}x0e"
        irreps_edge_attr = "0e+1o"
        irreps_node_out = irreps_hidden

        blocks = []
        _node_in = irreps_node_in
        for _ in range(n_message_passing):
            block = MessagePassing(
                irreps_node_input  = _node_in,
                irreps_node_attr   = irreps_node_attr,
                irreps_edge_attr   = irreps_edge_attr,
                irreps_node_output = irreps_node_out,
                invariant_layers   = n_invariant_layers,
                invariant_neurons  = n_invariant_neurons,
                edge_n_basis = edge_n_basis,
                avg_num_neighbors  = self.avg_num_neighbors,
                use_sc = True
            )
            blocks.append(block)
            # _node_in = irreps_node_out # No. gate sometimes does not return irreps_node_out
            _node_in = block.gate.irreps_out.simplify()

        if Irreps(_node_in).sort()[0].simplify() != Irreps(irreps_node_out).sort()[0].simplify():
            raise IOError("You may need to increase the n_blocks or reduce the complexity of irreps_hidden")

        self.blocks = nn.ModuleList(blocks)

        self.energy_out = nn.Sequential(
            Linear(irreps_node_out, Irreps(irreps_out_hidden)),
            Linear(Irreps(irreps_out_hidden), Irreps("1x0e"))
            )

        if dtype.lower()=="float64":
            self.double()

    def forward(self, batch: dict, compute_forces=True, return_desc=False, return_mid_features = False) -> dict:
        pos = batch["pos"]

        if compute_forces and not return_desc:
            pos.requires_grad_(True)

        atom_types = batch["atom_types"]
        edge_i, edge_j = batch["edge_i"], batch["edge_j"]
        # vec_ij = batch["edge_vecs"] # <-- wrong.. autograd disconnected in this way
        
        vec_ij_non_mic = pos[edge_j] - pos[edge_i]
        # cell_shift = batch["edge_cell_shift"] @ batch["cell"] # <-- wrong: impossible if batch came in. cell cannot be just concatenated
        edge_cell_shift = batch["edge_cell_shift"]

        ### a) increase dimension, use batch matrix-matrix multiplication
        _b_cell_shift = edge_cell_shift.unsqueeze(1).to(pos.dtype) # shape: (N_edges,1,3)
        _b_cell = batch["cell"][batch["batch_edges"]] # shape: (N_edges, 3, 3)
        cell_shift = torch.bmm(_b_cell_shift, _b_cell).squeeze(1) # Batch matrix-matrix product

        ### b) batch-wise calculation -- less memory, but 4 times slower than a
        # batch_edge = batch["batch_edges"]
        # cell = batch["cell"]
        # cell_shift = torch.zeros(
        #     edge_cell_shift.size(0),
        #     3,
        #     dtype  = pos.dtype,
        #     device = pos.device
        #     )
        # batch_size = cell.size(0)
        # for bi in range(batch_size):
        #     mask = (batch_edge == bi)
        #     if mask.any():
        #         ecs_bi = edge_cell_shift[mask].to(pos.dtype)
        #         c_bi = cell[bi]
        #         cell_shift[mask] = ecs_bi@c_bi

        vec_ij = vec_ij_non_mic + cell_shift

        r_ij = vec_ij.norm(dim=-1)
        node_feats = self.node_embedding_layer(atom_types)
        node_attrs = F.one_hot(atom_types, num_classes = self.n_elements).to(device=node_feats.device, dtype=node_feats.dtype)

        R_ij_raw = self.radial(r_ij) # R(r_ij) (raw basis)

        Y_ij = compute_Ylm(vec_ij, max_l = 1)

        data = {
            "pos": pos,
            "node_attrs": node_attrs,
            "node_features": node_feats,
            "edge_index": torch.stack([edge_i, edge_j], dim=0),
            "edge_attrs": Y_ij,
            "edge_embedding": R_ij_raw
        }

        if return_mid_features:
            mid_features = []

        for block in self.blocks:
            data = block(data)
            if return_mid_features:
                mid_features.append(data["node_features"])

        node_feats = data["node_features"]

        if return_desc:
            return node_feats
        
        energy_per_atom = self.energy_out(node_feats) + self.atom_energies[atom_types].unsqueeze(-1)
        data["energy_pred_per_atom"] = energy_per_atom
        batch_node = batch["batch_nodes"]

        energy_pred = torch.zeros(batch["cell"].size(0), 1, device=node_feats.device, dtype=node_feats.dtype)
        energy_pred = energy_pred.index_add_(0, batch_node, energy_per_atom).squeeze(-1)
        data["energy_pred"] = energy_pred
        
        if compute_forces:
            forces = -torch.autograd.grad(energy_per_atom.sum(), pos, create_graph=self.training)[0]
            data["forces_pred"] = forces
            pos.requires_grad_(False)

        return data if not return_mid_features else (mid_features, data)
    

    def atoms2dict(self, atoms:Atoms, device=None, include_D_matrix=False):
        if device is None:
            device = torch.device("cpu")
        pos = torch.tensor(atoms.get_positions(), dtype=self.dt, device=device)
        atom_types = torch.tensor([self.type_map[an] for an in atoms.get_atomic_numbers()], dtype=torch.long, device=device)
        cell = torch.tensor(atoms.cell.array, dtype=self.dt, device=device)
        pbc = torch.tensor(atoms.pbc, dtype=torch.bool, device=device)
        edge_i, edge_j, Vecs, Ds, edge_cell_shift = get_distances_torch(
            pos,
            cell=cell,
            return_all_neighborlist = True,
            r_cut_for_neighbor=self.r_cut,
            )

        return_dict = dict(
            pos = pos,
            atom_types = atom_types,
            cell = cell,
            edge_cell_shift = edge_cell_shift,
            edge_i = edge_i,
            edge_j = edge_j)

        if include_D_matrix:
            return_dict["Vecs"] = Vecs
            return_dict["Ds"] = Ds

        return return_dict
    
    def predict(self, atoms, compute_forces = True, return_desc=False, device=None):
        if device is None:
            device = torch.device("cpu")

        if isinstance(atoms, Atoms):
            _dict = self.atoms2dict(atoms, device=device)
            # atomsdict = collate_fn([_dict], single_input=True) # 'device' not working well
        elif isinstance(atoms, dict):
            _dict = atoms
        else:
            raise IOError(f"input type error: {type(atoms)} not supported")

        atomsdict = {k:v.to(device) for k, v in collate_fn([_dict], single_input=True).items()}
        if not compute_forces:
            with torch.no_grad():
                res_dict = self.forward(atomsdict, compute_forces=False, return_desc=return_desc)
        else:
            res_dict = self.forward(atomsdict, compute_forces=compute_forces, return_desc=return_desc)
        return res_dict

    
    def save(self, filename = "best_model.pt"):
        save_dict = {
        "model_state_dict": self.state_dict(), # atom_energies will be saved in here
        "gnn_config": self.gnn_config.copy(),
        "type_map": self.type_map.copy(),
        "avg_num_neighbors": self.avg_num_neighbors,
        }
        torch.save(save_dict, filename)

    @classmethod
    def load(cls, filename = "best_model.pt", device:str = None, dtype:str = None):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        try:
            loaded_data = torch.load(filename, map_location = device, weights_only = False)
        except TypeError:
            # If the saved model trained in float64, but if you want to load this to gpu machine that only supports float32,
            # load it to cpu first
            loaded_data = torch.load(filename, map_location = torch.device("cpu"), weights_only = False)

        gnn_config = loaded_data["gnn_config"]
        if dtype is not None:
            gnn_config["dtype"] = dtype
        type_map = loaded_data["type_map"]

        _ann = loaded_data.get("avg_num_neighbors", None)
        if _ann is None:
            raise IOError("You are trying to load the old GNN model, which doesn't have avg_num_neighbors. Please re-train it")

        loaded_model = cls(
            type_map = type_map,
            atom_energies = None,
            gnn_config = gnn_config,
            avg_num_neighbors = float(_ann)
            )
        loaded_model.load_state_dict(loaded_data["model_state_dict"]) # atom_energies will be loaded here
        return loaded_model.to(device=device)
    
    def per_atom_mse_loss_for_forces(self, forces_pred, forces_real):
        diff = forces_pred - forces_real
        per_atom_error = (diff ** 2).sum(dim=-1)  # [N_atoms] # NequIP uses mean instead of sum here
        force_loss = per_atom_error.mean()
        return force_loss
    
    def train_epoch(self, dataloader:DataLoader, optimizer, device, ef_ratio=[1.0, 1.0]):
        '''
        Training current model based on the dataloader

        :param ef_ratio (list): Energy and force ratio for training

        [Return]
        avg_loss, avg_energy_loss, avg_force_loss
        '''
        self.train()
        total_loss = 0.0
        total_energy_loss = 0.0
        total_force_loss = 0.0

        for _, batch in enumerate(dataloader):
            batch = {k:v.to(device) for k, v in batch.items()}

            ## just for checking whether GPU is using / How is the memory consumption
            # print(f"\n{_+1:d}th batch:")
            # print_memory_usage()

            optimizer.zero_grad()
            predictions = self.forward(batch, compute_forces = True)
            
            # Real
            energy_real = batch["energy"] # [N_batch]
            forces_real = batch["forces"] # [sum(N_atoms), 3]
            
            # Prediction
            energy_pred = predictions["energy_pred"]
            forces_pred = predictions["forces_pred"]
            
            # Loss
            n_atoms = batch["n_atoms"]
            # energy_loss = self.energy_loss_fn(energy_pred, energy_real) # MSELoss
            energy_loss = self.energy_loss_fn(energy_pred/n_atoms, energy_real/n_atoms) # MSELoss on energy (per atom)
            force_loss = self.per_atom_mse_loss_for_forces(forces_pred, forces_real)
            total_batch_loss = ef_ratio[0] * energy_loss + ef_ratio[1] * force_loss
            total_batch_loss.backward()

            ## For completeness, many experts recommend clipping the gradient's max_norm into certain threshold
            ## But what should be the appropriate values for max_norm? One way is, obtaining from avg_norm*0.5~10
            ## In order to this, we need to save the norm values, until a sufficiently large number is obtained
            ## https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch/56069467#56069467

            # params_grad = torch.tensor([p.grad for p in self.parameters() if p.grad is not None], dtype=forces_pred.dtype)
            # params_grad = torch.tensor(params_grad, dtype=torch.float64)
            # pg_norm = torch.linalg.vector_norm(params_grad) # default orb=2, save this into somewhere.
            # self.v.append(pg_norm)
            # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=torch.mean(self.v)*0.5~10)
            _max_grad_norm = self.gnn_config["max_grad_norm"]
            if _max_grad_norm is None:
                _skip_clipping = True

            elif isinstance(_max_grad_norm, str):
                if _max_grad_norm.lower() in ["inf", "null", "none"]:
                    _skip_clipping = True
                else:
                    try:
                        self.gnn_config["max_grad_norm"] = float(_max_grad_norm)
                        _skip_clipping = False
                    except:
                        raise IOError(f"Unknown input of max_grad_norm: {_max_grad_norm}")
            else:
                try:
                    self.gnn_config["max_grad_norm"] = float(_max_grad_norm)
                    _skip_clipping = False
                except:
                    raise IOError(f"Unknown input of max_grad_norm: {_max_grad_norm}")

            if not _skip_clipping:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.gnn_config["max_grad_norm"])

            optimizer.step()

            total_loss += total_batch_loss.item()
            total_energy_loss += energy_loss.item()
            total_force_loss += force_loss.item()

        avg_loss = total_loss / len(dataloader)
        avg_energy_loss = total_energy_loss / len(dataloader)
        avg_force_loss = total_force_loss / len(dataloader)

        return avg_loss, avg_energy_loss, avg_force_loss

    def validate_epoch(self, dataloader:DataLoader, device, ef_ratio=[1.0, 1.0]):
        self.eval()
        total_loss = 0.0
        total_energy_loss = 0.0
        total_force_loss = 0.0

        # with torch.no_grad():
        for batch in dataloader:
            batch = {k:v.to(device) for k, v in batch.items()}

            predictions = self.forward(batch, compute_forces = True)
            energy_real = batch["energy"]
            forces_real = batch["forces"]
            energy_pred = predictions["energy_pred"]
            forces_pred = predictions["forces_pred"]

            n_atoms = batch["n_atoms"]
            energy_loss = self.energy_loss_fn(energy_pred/n_atoms, energy_real/n_atoms) # MSELoss on energy (per atom)            
            force_loss = self.per_atom_mse_loss_for_forces(forces_pred, forces_real)

            total_batch_loss = ef_ratio[0] * energy_loss + ef_ratio[1] * force_loss

            total_loss += total_batch_loss.item()
            total_energy_loss += energy_loss.item()
            total_force_loss += force_loss.item()

        avg_loss = total_loss / len(dataloader)
        avg_energy_loss = total_energy_loss / len(dataloader)
        avg_force_loss = total_force_loss / len(dataloader)

        return avg_loss, avg_energy_loss, avg_force_loss

    def train_epoch_E_only(self, dataloader:DataLoader, optimizer, device):
        self.train()
        total_energy_loss = 0.0

        for _, batch in enumerate(dataloader):
            batch = {k:v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            predictions = self.forward(batch, compute_forces = False)
            
            energy_real = batch["energy"] # [N_batch]
            energy_pred = predictions["energy_pred"]
            
            n_atoms = batch["n_atoms"]
            # energy_loss = self.energy_loss_fn(energy_pred, energy_real) # MSELoss
            energy_loss = self.energy_loss_fn(energy_pred/n_atoms, energy_real/n_atoms) # MSELoss on energy (per atom)
            energy_loss.backward()

            _max_grad_norm = self.gnn_config["max_grad_norm"]
            if _max_grad_norm is None:
                _skip_clipping = True

            elif isinstance(_max_grad_norm, str):
                if _max_grad_norm.lower() in ["inf", "null", "none"]:
                    _skip_clipping = True
                else:
                    try:
                        self.gnn_config["max_grad_norm"] = float(_max_grad_norm)
                        _skip_clipping = False
                    except:
                        raise IOError(f"Unknown input of max_grad_norm: {_max_grad_norm}")
            else:
                try:
                    self.gnn_config["max_grad_norm"] = float(_max_grad_norm)
                    _skip_clipping = False
                except:
                    raise IOError(f"Unknown input of max_grad_norm: {_max_grad_norm}")

            if not _skip_clipping:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.gnn_config["max_grad_norm"])

            optimizer.step()

            total_energy_loss += energy_loss.item()
        avg_energy_loss = total_energy_loss / len(dataloader)
        return avg_energy_loss

    def validate_epoch_E_only(self, dataloader:DataLoader, device):
        self.eval()
        total_energy_loss = 0.0
        for batch in dataloader:
            batch = {k:v.to(device) for k, v in batch.items()}
            predictions = self.forward(batch, compute_forces = True)
            energy_real = batch["energy"]
            energy_pred = predictions["energy_pred"]
            n_atoms = batch["n_atoms"]
            energy_loss = self.energy_loss_fn(energy_pred/n_atoms, energy_real/n_atoms) # MSELoss on energy (per atom)            
            total_energy_loss += energy_loss.item()
        avg_energy_loss = total_energy_loss / len(dataloader)
        return avg_energy_loss


