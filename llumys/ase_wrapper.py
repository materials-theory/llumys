from ase.calculators.calculator import Calculator, all_changes
from llumys.gnn import EquivariantGNN
from llumys.gnn_LL import EquivariantGNN_UC
import torch, copy
from typing import List
from e3nn.o3 import Irreps

def eq2inv(eq_desc, irreps_eq:str = None, idx_slices:List = None):
    if idx_slices is not None:
        assert irreps_eq is not None, "Either irreps_eq or idx_slices should be typed in"      
    else:
        irreps_eq = Irreps(irreps_eq)
        irreps_inv = []
        
        idx_slices = []
        start_idx = 0
        total_ginv_dim = 0
        for mul, irrep in irreps_eq:
            ginv_dim = mul * irrep.dim
            if irrep.l==0 and irrep.p==1:
                idx_slices.append(slice(start_idx, start_idx+ginv_dim))
                irreps_inv.append((mul, irrep))
                total_ginv_dim += ginv_dim
            start_idx += ginv_dim        
    assert isinstance(idx_slices, list)

    parts = [eq_desc[:,sl] for sl in idx_slices]
    return torch.cat(parts, dim=-1)

class GNNWrapper(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, gnn_model, device=None, dtype=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cpu") if device is None else device
        self.dtype = torch.float64 if dtype is None else dtype

        if isinstance(gnn_model, str):
            gnn_model = EquivariantGNN.load(gnn_model, dtype=self.dtype, device=self.device)
        assert isinstance(gnn_model, EquivariantGNN)
        self.gnn_model = gnn_model
        self._count = 0

    def calculate(self, atoms, properties=["energy"], system_changes=all_changes):
        self.gnn_model.eval()
        compute_forces = "forces" in properties
        res_dict = self.gnn_model.predict(atoms, return_desc=False, compute_forces=compute_forces, device=self.device)
        desc_eq = res_dict["node_features"].detach()
        atoms.arrays["node_features"] = desc_eq.cpu().numpy()
        atoms.arrays["node_features_inv"] = eq2inv(desc_eq,
                                                   irreps_eq = self.gnn_model.blocks[-1].gate.irreps_out).cpu().numpy()
        atoms.arrays["energy_pred_per_atom"] = res_dict["energy_pred_per_atom"].detach().cpu().numpy().flatten()
        _E = res_dict["energy_pred"].item()
        self.results["energy"] = _E
        atoms.info["energy"] = _E
        if compute_forces:
            _F = res_dict["forces_pred"].detach().cpu().numpy()
            self.results["forces"] = _F
            atoms.arrays["forces"] = _F
        self._count += 1

    def __copy__(self):
        cls = self.__class__
        new = cls.__new__(cls)
        for k, v in self.__dict__.items():
            setattr(new, k, v if k in {"gnn_model", "device", "dtype"} else copy.copy(v))
        return new

    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        shared = {"gnn_model", "device", "dtype"}
        reinit_empty = {"_cache", "_graph_cache", "_hooks"}
        for k, v in self.__dict__.items():
            if k in shared:
                setattr(new, k, v)
            elif k in reinit_empty:
                setattr(new, k, {} if isinstance(v, dict) else [])
            else:
                setattr(new, k, copy.deepcopy(v, memo))
        return new


class GNNUCWrapper(Calculator):
    implemented_properties = ['energy', 'forces']
    def __init__(self, gnn_model, device=None, dtype=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cpu") if device is None else device
        self.dtype = torch.float64 if dtype is None else dtype

        if isinstance(gnn_model, str):
            gnn_model = EquivariantGNN_UC.load(gnn_model, dtype=self.dtype, device=self.device)
        assert isinstance(gnn_model, EquivariantGNN_UC)
        self.gnn_model = gnn_model
        self._count = 0

    def calculate(self, atoms, properties=["energy"], system_changes=all_changes):
        self.gnn_model.eval()
        compute_forces = "forces" in properties
        res_dict = self.gnn_model.predict(atoms, return_desc=False, compute_forces=compute_forces, device=self.device)
        desc_eq = res_dict["node_features"].detach()
        atoms.arrays["node_features"] = desc_eq.cpu().numpy()
        atoms.arrays["node_features_inv"] = eq2inv(desc_eq,
                                                   irreps_eq = self.gnn_model.blocks[-1].gate.irreps_out).cpu().numpy()
        _E = res_dict["energy_pred"].item()
        self.results["energy"] = _E
        atoms.info["energy"] = _E
        atoms.arrays["lhat_per_atom"] = res_dict["Floss_hat_per_node"].detach().cpu().numpy().flatten()
        atoms.arrays["energy_pred_per_atom"] = res_dict["energy_pred_per_atom"].detach().cpu().numpy().flatten()
        atoms.info["lhat"] = atoms.arrays["lhat_per_atom"].sum()
        
        if compute_forces:
            _F = res_dict["forces_pred"].detach().cpu().numpy()
            self.results["forces"] = _F
            atoms.arrays["forces"] = _F

        self._count += 1

        # Note: some keys are overlapped between calc.results and atoms.arrays. -> may cause an error when using 'write_xyz'
        # In this case, use 'write_xyz' after setting 'atoms.calc = None'
    def __copy__(self):
        cls = self.__class__
        new = cls.__new__(cls)
        for k, v in self.__dict__.items():
            setattr(new, k, v if k in {"gnn_model", "device", "dtype"} else copy.copy(v))
        return new

    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        shared = {"gnn_model", "device", "dtype"}
        reinit_empty = {"_cache", "_graph_cache", "_hooks"}
        for k, v in self.__dict__.items():
            if k in shared:
                setattr(new, k, v)
            elif k in reinit_empty:
                setattr(new, k, {} if isinstance(v, dict) else [])
            else:
                setattr(new, k, copy.deepcopy(v, memo))
        return new