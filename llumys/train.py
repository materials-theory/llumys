from llumys.gnn import *
from llumys.gnn_LL import EquivariantGNN_UC
from ase import Atoms

def main_GNN(
    xyz_filename:str,
    valid_xyz:str = None, # optional: if exists, train_ratio ignored
    train_ratio:float = 0.7, # default: 70% of data from xyz_filename used for trainset
    device:str = None,
    batch_size:int = 5,
    valid_batch_size:int = 1,
    max_epoch:int = 1000,
    patience:int = 50, # for not using early_stopping, set patience=None
    ef_ratio = [1, 1], # loss weight for energy : force
    gnn_config = None,
    input_atom_energies:Dict[str, float] = None,
    best_model_filename:str = "best_model.pt",
    loss_filename:str = "Losses.pickle",
    restart:str = "best_model.pt",
    ):
    '''
    [Note] currently, restart just performs the loading and retraining, not inheriting N_epoch, patience, lr and so on.
    # TODO: Maybe this is required for huge training..
    '''

    if gnn_config is None:
        gnn_config = {}

    if input_atom_energies is None:
        input_atom_energies = {}

    r_cut = gnn_config.get("r_cut", 5)
    dtype = gnn_config.get("dtype", "float64")

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        if isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, torch.device):
            device = device
        else:
            raise IOError("device arguments should be either str or torch.device instance")


    # database
    if isinstance(xyz_filename, list):
        if isinstance(xyz_filename[0], Atoms):
            all_dt = xyz_filename
    elif isinstance(xyz_filename, str):
        with open(xyz_filename, 'r') as fi:
            all_dt = list(read_xyz(fi, index=slice(None)))
    else:
        raise IOError(type(xyz_filename), "not supported for xyz_filename")

    # manual split of train:valid
    if valid_xyz is not None:
        if isinstance(valid_xyz, list):
            if isinstance(valid_xyz[0], Atoms):
                valid_atoms = valid_xyz
        elif isinstance(valid_xyz, str):
            with open(valid_xyz, 'r') as fi:
                valid_atoms = list(read_xyz(fi, index=slice(None)))
        else:
            raise IOError(type(valid_xyz), "not supported for valid_xyz")
        train_atoms = all_dt
    else:
        train_size = int(train_ratio * len(all_dt))
        val_size = len(all_dt) - train_size
        train_atoms, valid_atoms = random_split(all_dt, [train_size, val_size])

    if len(valid_atoms)==0:
        raise IOError("Training set-only mode is not implemented.")

    # test_atoms = []

    train_dataset = ASE_Dataset_EF(
        train_atoms,
        r_cut = r_cut,
        dtype = dtype,
        input_atom_energies  = input_atom_energies,
        return_atom_energies = True
        )

    type_map = train_dataset.type_map
    valid_dataset = ASE_Dataset_EF(valid_atoms, r_cut = r_cut, type_map = type_map, dtype = dtype)
    # test_dataset = ASE_Dataset_EF(test_atoms, r_cut = r_cut, type_map = type_map, dtype=dtype)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn = collate_fn) # shuffle: 매 epoch마다 데이터 순서 섞기!
    valid_loader = DataLoader(valid_dataset, batch_size = valid_batch_size, shuffle=False, collate_fn = collate_fn)

    if restart is None or not os.path.exists(restart):
        model = EquivariantGNN(
            type_map = type_map,
            gnn_config = gnn_config,
            atom_energies = train_dataset.atom_energies,
            avg_num_neighbors = train_dataset.avg_num_neighbors,
        ).to(device)
    else:
        model = EquivariantGNN.load(restart, device=device, dtype=dtype)

    optimizer = optim.Adam(model.parameters())
    model.train()

    ### Main Training -- both EF
    best_valid_loss = None
    epochs_no_improve = 0
    early_stop = False    
    L_train_avg_energy_loss = [] # MSELoss
    L_valid_avg_energy_loss = [] # MSELoss

    E_only = ef_ratio[1]==0
    if not E_only:
        L_train_avg_total_loss = []
        L_valid_avg_total_loss = []        
        L_train_avg_force_loss = [] # PerAtomMSELoss
        L_valid_avg_force_loss = [] # PerAtomMSELoss

    for epoch in range(max_epoch):
        st = time.time()

        if E_only:
            train_avg_energy_loss = model.train_epoch_E_only(train_loader, optimizer, device)
            valid_avg_energy_loss = model.validate_epoch_E_only(valid_loader, device)
        else:
            train_avg_total_loss, train_avg_energy_loss, train_avg_force_loss = model.train_epoch(train_loader, optimizer, device, ef_ratio=ef_ratio)
            valid_avg_total_loss, valid_avg_energy_loss, valid_avg_force_loss = model.validate_epoch(valid_loader, device, ef_ratio=ef_ratio)
        
        L_train_avg_energy_loss.append(train_avg_energy_loss)
        L_valid_avg_energy_loss.append(valid_avg_energy_loss)

        if not E_only:
            L_train_avg_total_loss.append(train_avg_total_loss)
            L_valid_avg_total_loss.append(valid_avg_total_loss)
            L_train_avg_force_loss.append(train_avg_force_loss)
            L_valid_avg_force_loss.append(valid_avg_force_loss)
        else:
            train_avg_total_loss = train_avg_energy_loss
            valid_avg_total_loss = valid_avg_energy_loss
        
        _improved = best_valid_loss is None or valid_avg_total_loss < best_valid_loss
        if patience is None:
            print(f'Epoch [{epoch+1}/{max_epoch}], Train Loss: {train_avg_total_loss:.4f}, Val Loss: {valid_avg_total_loss:.4f}, Walltime: {time.time()-st:.2f} seconds')
        else:
            print(f'Epoch [{epochs_no_improve+1}/{patience}] | from total [{epoch+1}/{max_epoch}], Train Loss: {train_avg_total_loss:.4f}, Val Loss: {valid_avg_total_loss:.4f}, Walltime: {time.time()-st:.2f} seconds')
        if _improved:
            best_valid_loss = valid_avg_total_loss
            epochs_no_improve = 0
            model.save(filename = f"{best_model_filename:s}")
        else:
            epochs_no_improve += 1
            if patience is not None:
                if epochs_no_improve >= patience:
                    print("Early Stopping!")
                    early_stop = True
                    break

    if loss_filename is not None:
        with open(f"{loss_filename:s}", 'wb') as fo:
            if E_only:
                pickle.dump([
                    L_train_avg_energy_loss,
                    L_valid_avg_energy_loss,
                ], fo)
            else:
                pickle.dump([
                    L_train_avg_total_loss,
                    L_train_avg_energy_loss,
                    L_train_avg_force_loss,
                    L_valid_avg_total_loss,
                    L_valid_avg_energy_loss,
                    L_valid_avg_force_loss
                ], fo)
    return epoch+1

def main_UC(
    xyz_filename:str,
    valid_xyz:str = None, # optional: if exists, train_ratio ignored
    train_ratio:float = 0.7, # default: 70% of data from xyz_filename used for trainset
    device:str = None,
    batch_size:int = 5,
    valid_batch_size:int = 1,
    max_epoch:int = 2000,
    nonLL_epoch:int = 10,
    patience:int = 75, # for not using early_stopping, set patience=None
    ef_ratio = [1, 1], # loss weight for energy : force
    gnn_config = None,
    input_atom_energies:Dict[str, float] = None,
    best_model_filename:str = "best_model_UC.pt",
    loss_filename:str = "Losses_UC.pickle",
    restart:str = "best_model_UC.pt"
    ## l_lambda = 1 # TODO - check first and then put l_lambda as argument
    ):
    '''
    [Note] currently, restart just performs the loading and retraining, not inheriting N_epoch, patience, lr and so on.
    # TODO: Maybe this is required for huge training..
    '''

    if gnn_config is None:
        gnn_config = {}

    if input_atom_energies is None:
        input_atom_energies = {}

    if ef_ratio[1]==0:
        ### E only mode not supported
        raise IOError("E-only mode not supported for LL, since it uses forces for ranking uncertainty")

    r_cut = gnn_config.get("r_cut", 5)
    dtype = gnn_config.get("dtype", "float64")

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        if isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, torch.device):
            device = device
        else:
            raise IOError("device arguments should be either str or torch.device instance")

    # database
    if isinstance(xyz_filename, list):
        if isinstance(xyz_filename[0], Atoms):
            all_dt = xyz_filename
    elif isinstance(xyz_filename, str):
        with open(xyz_filename, 'r') as fi:
            all_dt = list(read_xyz(fi, index=slice(None)))
    else:
        raise IOError(type(xyz_filename), "not supported for xyz_filename")

    # manual split of train:valid
    if valid_xyz is not None:
        if isinstance(valid_xyz, list):
            if isinstance(valid_xyz[0], Atoms):
                valid_atoms = valid_xyz
        elif isinstance(valid_xyz, str):
            with open(valid_xyz, 'r') as fi:
                valid_atoms = list(read_xyz(fi, index=slice(None)))
        else:
            raise IOError(type(valid_xyz), "not supported for valid_xyz")
        train_atoms = all_dt
    else:
        train_size = int(train_ratio * len(all_dt))
        val_size = len(all_dt) - train_size
        train_atoms, valid_atoms = random_split(all_dt, [train_size, val_size])

    # test_atoms = []

    train_dataset = ASE_Dataset_EF(
        train_atoms,
        r_cut = r_cut,
        dtype = dtype,
        input_atom_energies  = input_atom_energies,
        return_atom_energies = True
        )
    type_map = train_dataset.type_map
    valid_dataset = ASE_Dataset_EF(valid_atoms, r_cut = r_cut, type_map = type_map, dtype=dtype)
    # test_dataset = ASE_Dataset_EF(test_atoms, r_cut = r_cut, type_map = type_map, dtype=dtype)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn = collate_fn) # shuffle: 매 epoch마다 데이터 순서 섞기!
    valid_loader = DataLoader(valid_dataset, batch_size = valid_batch_size, shuffle=False, collate_fn = collate_fn)


    L_train_avg_total_loss = []
    L_train_avg_target_loss = []
    L_train_avg_energy_loss = [] # MSELoss
    L_train_avg_force_loss = [] # PerAtomMSELoss

    L_valid_avg_target_loss = []
    L_valid_avg_energy_loss = [] # MSELoss
    L_valid_avg_force_loss = [] # PerAtomMSELoss

    if restart is None or not os.path.exists(restart):
        # nonLL_epoch
        model_nonLL = EquivariantGNN(
            type_map = type_map,
            gnn_config = gnn_config,
            atom_energies = train_dataset.atom_energies,
            avg_num_neighbors = train_dataset.avg_num_neighbors,
        ).to(device)

        optimizer = optim.Adam(model_nonLL.parameters())
        model_nonLL.train()

        for epoch in range(nonLL_epoch):
            st = time.time()
            train_avg_target_loss, train_avg_energy_loss, train_avg_force_loss = model_nonLL.train_epoch(train_loader, optimizer, device, ef_ratio = ef_ratio)
            valid_avg_target_loss, valid_avg_energy_loss, valid_avg_force_loss = model_nonLL.validate_epoch(valid_loader, device, ef_ratio=ef_ratio)
            
            # L_train_avg_total_loss.append(train_avg_total_loss) # 여긴 없지 ㅋㅋㅋ
            L_train_avg_target_loss.append(train_avg_target_loss)
            L_valid_avg_target_loss.append(valid_avg_target_loss)

            L_train_avg_energy_loss.append(train_avg_energy_loss)
            L_valid_avg_energy_loss.append(valid_avg_energy_loss)
            L_train_avg_force_loss.append(train_avg_force_loss)
            L_valid_avg_force_loss.append(valid_avg_force_loss)

            print(f'Epoch [{epoch+1}/{nonLL_epoch}], Train Target Loss: {train_avg_target_loss:.4f}, Val Target Loss: {valid_avg_target_loss:.4f}, Walltime: {time.time()-st:.2f} seconds')

        model = EquivariantGNN_UC(
            type_map = type_map,
            gnn_config = gnn_config,
            atom_energies = train_dataset.atom_energies,
            avg_num_neighbors = train_dataset.avg_num_neighbors,
            mid_fc_dim   = 8,
            final_fc_dim = 8,
            margin = 1.0
        ).to(device)

        pretrained_dict = model_nonLL.state_dict()
        current_state_dict = model.state_dict()
        to_update = {k: v for k, v in pretrained_dict.items() if k in current_state_dict and v.size() == current_state_dict[k].size()}
        current_state_dict.update(to_update)
        model.load_state_dict(current_state_dict)
        # or model.load_state_dict(model_nonLL.state_dict(), strict=False)
    else:
        model = EquivariantGNN_UC.load(restart, device=device, dtype=dtype)

    optimizer = optim.Adam(model.parameters())

    ### 2. Main training --- detach=True
    model.train()
    best_valid_target_loss = None
    epochs_no_improve = 0
    early_stop = False

    detach_subhead_crit = True

    for epoch in range(max_epoch):
        st = time.time()
        train_avg_total_loss, train_avg_target_loss, train_avg_energy_loss, train_avg_force_loss = model.train_epoch(train_loader, optimizer, device, ef_ratio=ef_ratio, l_lambda=1, detach_subhead = detach_subhead_crit)
        valid_avg_target_loss, valid_avg_energy_loss, valid_avg_force_loss = model.validate_epoch(valid_loader, device, ef_ratio=ef_ratio)
        
        L_train_avg_total_loss.append(train_avg_total_loss)
        L_train_avg_target_loss.append(train_avg_target_loss)
        L_valid_avg_target_loss.append(valid_avg_target_loss)

        L_train_avg_energy_loss.append(train_avg_energy_loss)
        L_valid_avg_energy_loss.append(valid_avg_energy_loss)
        L_train_avg_force_loss.append(train_avg_force_loss)
        L_valid_avg_force_loss.append(valid_avg_force_loss)
        
        _improved = best_valid_target_loss is None or valid_avg_target_loss < best_valid_target_loss
        if patience is None:
            print(f'Epoch [{epoch+1}/{max_epoch}], Train Total Loss: {train_avg_total_loss:.4f}, Train Target Loss: {train_avg_target_loss:.4f}, Val Target Loss: {valid_avg_target_loss:.4f}, Walltime: {time.time()-st:.2f} seconds')
        else:
            print(f'Epoch [{epochs_no_improve+1}/{patience}] | from total [{epoch+1}/{max_epoch}], Train Total Loss: {train_avg_total_loss:.4f}, Train Target Loss: {train_avg_target_loss:.4f}, Val Target Loss: {valid_avg_target_loss:.4f}, Walltime: {time.time()-st:.2f} seconds')        
        if _improved:
            best_valid_target_loss = valid_avg_target_loss
            epochs_no_improve = 0
            model.save(filename = f"{best_model_filename:s}")
        else:
            epochs_no_improve += 1
            if patience is not None:
                if epochs_no_improve >= patience:
                    print("Early Stopping!")
                    early_stop = True
                    break

    if loss_filename is not None:
        with open(f"{loss_filename:s}", 'wb') as fo:
            pickle.dump([
                L_train_avg_total_loss,
                L_train_avg_target_loss,
                L_train_avg_energy_loss,
                L_train_avg_force_loss,
                L_valid_avg_target_loss,
                L_valid_avg_energy_loss,
                L_valid_avg_force_loss
            ], fo)

    return epoch+1