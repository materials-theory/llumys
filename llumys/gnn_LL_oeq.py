### Import and warnings settings
from llumys.gnn_oeq import *
# import torch

class EquivariantGNN_UC(EquivariantGNN):
    # lhat_scaling: List[float] = [1.0, 0.0]
    def __init__(
        self,
        type_map:dict = None,
        gnn_config:dict = None,
        atom_energies:dict = None,
        avg_num_neighbors:float = None,
        mid_fc_dim   = 8,
        final_fc_dim = 8,
        margin = 1.0
        ):
        super().__init__(
            type_map          = type_map,
            gnn_config        = gnn_config,
            atom_energies     = atom_energies,
            avg_num_neighbors = avg_num_neighbors)
        self.n_blocks = len(self.blocks)
        self.margin = margin

        _mid_fc_list = []
        for block in self.blocks:
            _mid_fc_list.append(Linear(block.gate.irreps_out, f"{mid_fc_dim}x0e"))
        self.mid_fc_list = nn.ModuleList(_mid_fc_list)

        self.subhead = nn.Sequential(
            nn.Linear(self.n_blocks * mid_fc_dim, final_fc_dim),
            nn.ReLU(),
            nn.Linear(final_fc_dim, 1),
            nn.Softplus()
            )

        self.ranking_loss_fn = nn.MarginRankingLoss(margin=self.margin)
        if self.gnn_config["dtype"].lower()=="float64":
            self.double()

        self.learning_loss_config = dict(
            mid_fc_dim = mid_fc_dim,
            final_fc_dim = final_fc_dim,
            margin = margin)


    def save(self, filename = "best_model.pt"):
        save_dict = {
        "model_state_dict": self.state_dict(),
        "gnn_config": self.gnn_config.copy(),
        "avg_num_neighbors": self.avg_num_neighbors,
        "type_map": self.type_map.copy(),
        "learning_loss_config" : self.learning_loss_config.copy()
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
        learning_loss_config = loaded_data["learning_loss_config"]

        _ann = loaded_data.get("avg_num_neighbors", None)
        if _ann is None:
            raise IOError("You are trying to load the old GNN model, which doesn't have avg_num_neighbors. Please re-train it")

        loaded_model = cls(
            type_map = type_map,
            gnn_config = gnn_config,
            avg_num_neighbors=_ann,
            **learning_loss_config)
        loaded_model.load_state_dict(loaded_data["model_state_dict"])
        return loaded_model.to(device=device)


    def forward(self, batch: dict, compute_forces=True, return_desc=False, detach_subhead=False) -> dict:
        if return_desc:
            return super().forward(batch, compute_forces = compute_forces, return_desc = True)

        mid_feats, out_data = super().forward(
            batch = batch,
            compute_forces = compute_forces,
            return_desc = False,
            return_mid_features = True)

        energy_pred = out_data["energy_pred"]
        device, dtype = energy_pred.device, energy_pred.dtype

        batch_nodes = batch["batch_nodes"]
        n_batch = batch_nodes.max().item()+1

        mid_pooled_list = []
        for i in range(self.n_blocks):
            _ = mid_feats[i].detach() if detach_subhead else mid_feats[i]
            mid_feat = self.mid_fc_list[i](_) # (N_atoms, n_feats-scalar > mid_fc_dim)
            mid_pooled_list.append(mid_feat)
        cat_mid_feat = torch.cat(mid_pooled_list, dim=-1) # (n_batch*N_atoms, n_blocks*mid_fc_dim)

        predicted_loss_per_node = self.subhead(cat_mid_feat)
        out_data["Floss_hat_per_node"] = predicted_loss_per_node
        return out_data

    def train_epoch(self, dataloader:DataLoader, optimizer, device, ef_ratio=[1.0, 1.0], l_lambda=1.0, detach_subhead=False):
        self.train()
        epoch_total_loss = 0.0
        epoch_target_loss = 0.0
        epoch_energy_loss = 0.0
        epoch_force_loss = 0.0
        epoch_rank_loss = 0.0

        for _, batch in enumerate(dataloader):
            batch = {k:v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            predictions = self.forward(batch, compute_forces = True, detach_subhead = detach_subhead)
            
            # Real
            energy_real = batch["energy"] # [N_batch]
            forces_real = batch["forces"] # [sum(N_atoms), 3]
            
            # Prediction
            energy_pred = predictions["energy_pred"]
            forces_pred = predictions["forces_pred"]
            Floss_hat_per_node = predictions["Floss_hat_per_node"].squeeze(-1) # [sum(N_atoms)]
            
            # Loss
            n_atoms = batch["n_atoms"]
            energy_loss = self.energy_loss_fn(energy_pred/n_atoms, energy_real/n_atoms) # MSELoss on energy (per atom)
            force_loss = self.per_atom_mse_loss_for_forces(forces_pred, forces_real)
            target_loss = ef_ratio[0] * energy_loss + ef_ratio[1] * force_loss

            real_force_err_per_node = torch.norm(forces_real - forces_pred, p=2, dim=-1)
            if detach_subhead:
                real_force_err_per_node = real_force_err_per_node.detach()

            num_nodes = real_force_err_per_node.shape[0]
            if num_nodes > 1:
                indices = torch.randperm(num_nodes, device=device)
                indices_i = indices[:num_nodes//2]
                indices_j = indices[num_nodes//2:2*(num_nodes//2)]
                real_err_i = real_force_err_per_node[indices_i]
                real_err_j = real_force_err_per_node[indices_j]
                pred_err_i = Floss_hat_per_node[indices_i]
                pred_err_j = Floss_hat_per_node[indices_j]

                y = torch.sign(real_err_i - real_err_j)
                y[y==0] = -1
                rank_loss = self.ranking_loss_fn(pred_err_i, pred_err_j, y)
            else:
                rank_loss = torch.tensor(0.0, device=device, dtype=energy_pred.dtype)

            total_loss = target_loss + l_lambda * rank_loss
            total_loss.backward()

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
            epoch_total_loss += total_loss.item()
            epoch_target_loss += target_loss.item()
            epoch_energy_loss += energy_loss.item()
            epoch_force_loss += force_loss.item()
            epoch_rank_loss += rank_loss.item()
        avg_total_loss = epoch_total_loss / len(dataloader)
        avg_target_loss = epoch_target_loss / len(dataloader)
        avg_energy_loss = epoch_energy_loss / len(dataloader)
        avg_force_loss = epoch_force_loss / len(dataloader)
        avg_rank_loss = epoch_rank_loss / len(dataloader)

        return avg_total_loss, avg_target_loss, avg_energy_loss, avg_force_loss



