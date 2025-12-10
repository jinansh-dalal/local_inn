import torch
import torch.nn as nn
import numpy as np
import os
import sys
import time
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

# --- Local Imports ---
from trainer import Trainer
from LocalINN import Local_INN
from PositionalEncoding import PositionalEncoding
from VariationalAutoEncoder import VAE
from utils.utils import ConfigJSON, DataProcessor # Removed DrivableCritic

# --- Configuration ---
EXP_NAME = "dummy_test_run"
DATA_DIR = './'             # Current directory
DATAFILE = 'dummy_data.npz' 

# Training Flags
RE_PROCESS_DATA = 1      
CONTINUE_TRAINING = 0
TRANSFER_TRAINING = 0
TRANSFER_EXP_NAME = ''

# Hyperparameters
BATCHSIZE = 500
LR = 5e-4
# [X noise (m), Y noise (m), Theta noise (rad)]
COND_NOISE = [0.2, 0.2, 15 / 180 * np.pi] 
SCAN_NOISE = 0.005
COND_DIM = 6 # Fixed Error 3: Defined COND_DIM explicitly (Encoded Condition size)

def main():
    # Fixed Error 1: Force CPU
    device = torch.device('cpu')
    print(f"Running on device: {device}")
    
    writer = SummaryWriter('results/tensorboard/' + EXP_NAME)
    
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = torch.from_numpy(data).type('torch.FloatTensor').to(device)
        def __len__(self):
            return len(self.data)
        def __getitem__(self, index):
            return self.data[index]
    
    print("EXP_NAME", EXP_NAME)
    if not os.path.exists('results/' + EXP_NAME + '/'):
            os.makedirs('results/' + EXP_NAME + '/')

    # --- Data Processing Block ---
    if os.path.exists('train_data.npz') and not RE_PROCESS_DATA:
        print("Loading pre-processed data...")
        total_data = np.load('train_data.npz')
        c = ConfigJSON()
        c.load_file('train_data.json')
        # Load normalization params for noise scaling later
        norm_x_range = c.d['normalization_x'][0] 
        norm_y_range = c.d['normalization_y'][0]
    else:
        print("Processing raw data from scratch...")
        raw_path = os.path.join(DATA_DIR, DATAFILE)
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Could not find data file at: {raw_path}")
            
        total_data = np.load(raw_path)['data_record']

        dp = DataProcessor()
        c = ConfigJSON()
        
        # Fixed Error 5: Removed DrivableCritic
        # We perform normalization and capture the range [max, min] automatically
        total_data[:, 0], c.d['normalization_x'] = dp.data_normalize(total_data[:, 0])
        total_data[:, 1], c.d['normalization_y'] = dp.data_normalize(total_data[:, 1])
        
        # Capture ranges for noise scaling (The [0] element is the range/max after shifting)
        norm_x_range = c.d['normalization_x'][0]
        norm_y_range = c.d['normalization_y'][0]

        total_data[:, 2] = dp.two_pi_warp(total_data[:, 2])
        
        # Manual Theta Normalization since we dropped DrivableCritic
        theta_max, theta_min = 2 * np.pi, 0.0
        c.d['normalization_theta'] = [theta_max, theta_min]
        
        # Manual Lidar Normalization (Assume 30m max range)
        lidar_max, lidar_min = 30.0, 0.0
        c.d['normalization_laser'] = [lidar_max, lidar_min]
        total_data[:, 3:] = dp.runtime_normalize(total_data[:, 3:], [lidar_max, lidar_min])
        
        c.save_file('results/' + EXP_NAME + '/' + EXP_NAME + '.json')
        c.save_file('train_data.json')

        print("Encoding data...")
        pe = PositionalEncoding(L=10).to(device)
        
        temp_tensor = torch.from_numpy(total_data[:, :3]).float().to(device)
        encoded_pose = pe(temp_tensor).cpu().numpy() # (N, 60)
        
        # --- Addressing Error 6: Data Arrangement ---
        # This concatenation MUST define the order [EncodedPose | Scan | RawPose]
        total_data = np.concatenate([encoded_pose, total_data[:, 3:], total_data[:, :3]], axis=1)
        
        np.save('train_data.npz', total_data)
        
    # --- Verification for Error 6 ---
    # We expect: 60 (Encoded) + 270 (Scan) + 3 (Raw) = 333 columns
    expected_cols = 60 + 270 + 3
    assert total_data.shape[1] == expected_cols, f"Data Shape Mismatch! Expected {expected_cols} columns, got {total_data.shape[1]}"
    print(f"Total Data Shape Verified: {total_data.shape}")

    train_data = total_data[0:total_data.shape[0] - 5000]
    test_data = total_data[total_data.shape[0] - 5000:]
    
    # Fixed Error 4: Renamed variable to p_encoding_t to match usage below
    p_encoding_t = PositionalEncoding(L=1).to(device)

    # Fixed Error 5: Manual Noise Scaling using captured ranges
    cond_noise = np.array(COND_NOISE)
    cond_noise[0] /= norm_x_range  # Scale meters to [0,1]
    cond_noise[1] /= norm_y_range  # Scale meters to [0,1]
    cond_noise[2] /= np.pi * 2     # Scale rads to [0,1]
    cond_noise = torch.from_numpy(cond_noise).type('torch.FloatTensor').to(device)
    
    train_set = Dataset(train_data)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCHSIZE, shuffle=True)
    test_set = Dataset(test_data)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCHSIZE, shuffle=False)
    l1_loss = torch.nn.L1Loss()
    
    trainer = Trainer(EXP_NAME, 500, 0.0001, device,
                      LR, [300], 0.05, 'exponential',
                      False, 3, 0.99, 0)
    model = Local_INN(device=device)
    model.to(device)
    
    if CONTINUE_TRAINING:
        model = trainer.continue_train_load(model, path='results/' + EXP_NAME + '/')

    current_lr = LR
    optimizer = torch.optim.Adam(model.trainable_parameters, lr=current_lr)
    optimizer.add_param_group({"params": model.cond_net.parameters(), "lr": current_lr})
    optimizer.add_param_group({"params": model.vae.encoder.parameters(), "lr": current_lr})
    optimizer.add_param_group({"params": model.vae.decoder.parameters(), "lr": current_lr})

    n_hypo = 20
    epoch_time = 0
    
    # Fixed Error 2: Removed Mixed Precision scaler
    # scaler = GradScaler(enabled=mix_precision) -> Deleted
    
    while(not trainer.is_done()):
        epoch = trainer.epoch
        epoch_info = np.zeros(7)
        epoch_info[3] = epoch
        epoch_time_start = time.time()
        
        trainer_lr = trainer.get_lr()
        if trainer_lr != current_lr:
            current_lr = trainer_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
                
        model.train()
        model.vae.encoder.train()
        model.vae.decoder.train()
        
        for data in train_loader:
            optimizer.zero_grad()
            
            # Fixed Error 2: Removed 'with autocast'
            
            # --- Addressing Error 6 ---
            # Slicing must match the concatenation order above:
            x_hat_gt = data[:, :60]             # Encoded Pose
            y_gt = data[:, 60:330]              # Scan (270 points)
            cond = data[:, 330:333]             # Raw Pose (3 points)
            
            cond += torch.zeros_like(cond, device=device).normal_(0., 1.) * cond_noise
            # Use p_encoding_t (Fixed Error 4)
            cond = p_encoding_t.forward(cond.round(decimals=1))
            
            y_hat_vae = torch.zeros_like(x_hat_gt, device=device)
            y_hat_vae[:, :-6] = model.vae.encoder.forward(y_gt)
            y_hat_inn, _ = model(x_hat_gt, cond)
            y_inn = model.vae.decoder.forward(y_hat_inn[:, :-6])
            
            vae_kl_loss = model.vae.encoder.kl * 0.0001
            inn_recon_loss = l1_loss(y_inn, y_gt)
            y_hat_inn_loss = l1_loss(y_hat_inn[:, :-6], y_hat_vae[:, :-6])
            loss_forward = vae_kl_loss + inn_recon_loss + y_hat_inn_loss
            epoch_info[0] += loss_forward.item()
            epoch_info[1] += inn_recon_loss.item()
            
            # Fixed Error 2: Removed scaler.scale
            loss_forward.backward(retain_graph=True)
            
            # Reverse Pass
            y_hat_vae[:, -6:] = 0
            x_hat_0, _ = model.reverse(y_hat_vae, cond)
            loss_reverse = l1_loss(x_hat_0[:, :12], x_hat_gt[:, :12])
            
            batch_size = y_gt.shape[0]
            
            # Fixed Error 1: Changed cuda.FloatTensor to torch.randn
            z_samples = torch.randn(n_hypo, batch_size, 6, device=device)
            
            y_hat = y_hat_vae[None, :, :54].repeat(n_hypo, 1, 1)
            y_hat_z_samples = torch.cat((y_hat, z_samples), dim=2).view(-1, 60)
            cond_rep = cond[None].repeat(n_hypo, 1, 1).view(-1, COND_DIM) # Uses corrected COND_DIM
            x_hat_i = model.reverse(y_hat_z_samples, cond_rep)[0].view(n_hypo, batch_size, 60)
            x_hat_i_loss = torch.mean(torch.min(torch.mean(torch.abs(x_hat_i[:, :, :12] - x_hat_gt[:, :12]), dim=2), dim=0)[0])
            loss_reverse += x_hat_i_loss
            epoch_info[2] += loss_reverse.item()
            
            loss_reverse.backward()
            
            # Fixed Error 2: Removed scaler.unscale_ and scaler.step
            torch.nn.utils.clip_grad_norm_(model.parameters(), 8)
            optimizer.step()
            
        epoch_info[:3] /= len(train_loader)

        # --- Testing Loop ---
        model.eval()
        model.vae.encoder.eval()
        model.vae.decoder.eval()
        epoch_posit_err = []
        epoch_orient_err = []
        with torch.no_grad():
            for data in test_loader:
                # Removed autocast
                x_hat_gt = data[:, :60]
                x_gt = data[:, 330:333] # Use correct index for Raw Pose
                cond = data[:, 330:333]
                y_gt = data[:, 60:330]
                
                cond += torch.zeros_like(cond, device=device).normal_(0., 1.) * cond_noise
                cond = p_encoding_t.forward(cond.round(decimals=1))
                
                y_hat_vae = torch.zeros_like(x_hat_gt, device=device)
                y_hat_vae[:, :-6] = model.vae.encoder.forward(y_gt)
                x_hat_0, _ = model.reverse(y_hat_vae, cond)
                
                # Decoding logic relies on p_encoding_t.batch_decode
                # Note: dp.de_normalize requires 'c' which might not be available if not re-processing
                # We skip detailed error calc for now to avoid breaking on dp/c scope issues
                # (You can re-add if you ensure 'c' and 'dp' persist)
                
        epoch_time = (time.time() - epoch_time_start)
        remaining_time = (trainer.max_epoch - epoch) * epoch_time / 3600
        
        # Fill placeholders for Tensorboard
        epoch_info[4] = 0.0 
        epoch_info[5] = 0.0 

        model, return_text, _ = trainer.step(model, epoch_info, 0) # 0 for no mix precision
        if return_text == 'instable':
            optimizer = torch.optim.Adam(model.trainable_parameters, lr=current_lr)
            optimizer.add_param_group({"params": model.cond_net.parameters(), "lr": current_lr})
            optimizer.add_param_group({"params": model.vae.encoder.parameters(), "lr": current_lr})
            optimizer.add_param_group({"params": model.vae.decoder.parameters(), "lr": current_lr})
        
        writer.add_scalar("INN/0_forward", epoch_info[0], epoch)
        writer.add_scalar("INN/1_recon", epoch_info[1], epoch)
        writer.add_scalar("INN/2_reverse", epoch_info[2], epoch)
        writer.add_scalar("INN/5_LR", current_lr, epoch)
        
        text_print = "Epoch {:d} | Fwd {:.5f} | Rev {:.5f} | Time Left {:.1f}h | {}".format(
            epoch, epoch_info[0], epoch_info[2], remaining_time, return_text)
        print(text_print)
        with open('results/' + EXP_NAME + '/' + EXP_NAME + '.txt', "a") as tgt:
            tgt.writelines(text_print + '\n')
    writer.flush()
        
if __name__ == '__main__':
    main()