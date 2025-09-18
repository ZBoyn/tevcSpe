import torch
import torch.nn as nn
import torch.nn.functional as F
from actor_critic import PFSPNet, CriticNetwork
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from gene_data import generate_and_save_instance
from calc import calculate_objectives_pytorch
import argparse

def train_one_batch(
    actor_model, critic_model, optimizer_actor, optimizer_critic,
    batch_instance_features,
    batch_num_machines_scalar,
    P_instance, E_instance, R_instance, u_instance, s_instance, f_instance,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    batch_size = batch_instance_features.size(0)
    num_total_jobs = batch_instance_features.size(1)
    num_machines = batch_instance_features.size(2)

    actor_model.train()
    critic_model.train()

    batch_instance_features = batch_instance_features.to(device)
    batch_num_machines_scalar = batch_num_machines_scalar.to(device)

    selected_indices, log_probs_distributions, encoded_jobs = actor_model(
        batch_instance_processing_times=batch_instance_features,
        batch_num_machines_scalar=batch_num_machines_scalar,
        max_decode_len=num_total_jobs
    )

    put_off_matrices_zeros = torch.zeros(batch_size, num_machines, num_total_jobs, device=device, dtype=torch.long)

    actual_cmax_tensor, _ = calculate_objectives_pytorch(
        job_sequences=selected_indices,
        put_off_matrices=put_off_matrices_zeros,
        P=P_instance,
        E=E_instance,
        R=R_instance,
        u=u_instance,
        s=s_instance,
        f=f_instance,
        device=device
    )

    actual_cmax_tensor[torch.isinf(actual_cmax_tensor)] = 5000.0

    baseline_estimates_b = critic_model(encoded_jobs).squeeze(-1)
    critic_loss = F.mse_loss(baseline_estimates_b, actual_cmax_tensor.detach())
    optimizer_critic.zero_grad()
    critic_loss.backward(retain_graph=True)
    optimizer_critic.step()

    advantage = actual_cmax_tensor - baseline_estimates_b.detach()
    gathered_log_probs = torch.gather(log_probs_distributions, 2, selected_indices.unsqueeze(2)).squeeze(2)
    sum_log_probs_for_sequence = gathered_log_probs.sum(dim=1)
    actor_loss = (advantage * sum_log_probs_for_sequence).mean()
    optimizer_actor.zero_grad()
    actor_loss.backward()
    optimizer_actor.step()

    return actor_loss.item(), critic_loss.item(), actual_cmax_tensor.mean().item()

def plot_training_results(epochs_range, history_cmax, history_best_cmax, history_actor_loss, history_critic_loss):
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history_cmax, label='Avg Cmax per Epoch', marker='o', color='blue')
    plt.plot(epochs_range, history_best_cmax, label='Best Cmax seen in Epoch', marker='x', linestyle='--', color='cyan')
    plt.xlabel("Epoch")
    plt.ylabel("Cmax")
    plt.title("Average and Best Cmax per Epoch")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history_actor_loss, label='Avg Actor Loss per Epoch', marker='o', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Actor Loss")
    plt.title("Average Actor Loss per Epoch")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history_critic_loss, label='Avg Critic Loss per Epoch', marker='o', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Critic Loss")
    plt.title("Average Critic Loss per Epoch")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False 
    plt.show()


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_jobs', type=int, default=10)
    parser.add_argument('--num_machines', type=int, default=3)
    parser.add_argument('--k_intervals', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--steps_per_epoch', type=int, default=50)
    args = parser.parse_args()

    config_batch_size = args.batch_size
    config_num_jobs = args.num_jobs
    config_num_machines = args.num_machines
    config_k_intervals = args.k_intervals
    config_epochs = args.epochs
    config_steps_per_epoch = args.steps_per_epoch
    
    instance_save_dir = 'PFSPNET/data'
    instance_filename = f"instance_{config_num_jobs}j_{config_num_machines}m_{config_k_intervals}k.pt"
    instance_path = os.path.join(instance_save_dir, instance_filename)

    if not os.path.exists(instance_path):
        print(f"Not Found {instance_path}, Generating new instance")
        generate_and_save_instance(
            num_jobs=config_num_jobs,
            num_machines=config_num_machines,
            k_intervals=config_k_intervals,
            save_path=instance_save_dir
        )
    
    print(f"Loading instance: {instance_path}")
    instance_data = torch.load(instance_path, map_location=device)
    P_instance = instance_data['P_instance']
    E_instance = instance_data['E_instance']
    R_instance = instance_data['R_instance']
    u_starts = instance_data['u_starts']
    s_durations = instance_data['s_durations']
    f_factors = instance_data['f_factors']
    
    enc_part1_args = {'scalar_input_dim': 1, 'embedding_dim': 32, 'hidden_dim': 64, 'rnn_type': 'RNN', 'num_rnn_layers':1}
    enc_part2_args = {'p_vector_dim': enc_part1_args['hidden_dim'], 'm_embedding_dim': 16, 'output_dim': 48}
    ENC_OUT_CHANNELS = 96
    enc_part3_args = {'p_tilde_dim': enc_part2_args['output_dim'], 'conv_out_channels': ENC_OUT_CHANNELS, 'conv_kernel_size': 3, 'conv_padding': 'same'}
    encoder_config_args = {'part1_args': enc_part1_args, 'part2_args': enc_part2_args, 'part3_args': enc_part3_args}

    dec_step1_pt_enc_args = {'scalar_input_dim': 1, 'embedding_dim': 30, 'hidden_dim': 50, 'rnn_type': 'RNN', 'num_rnn_layers': 1}
    decoder_config_args = {
        'step1_pt_encoder_args': dec_step1_pt_enc_args,
        'step1_m_embedding_dim': 20, 'step1_di_output_dim': 40, 'step1_fc_hidden_dims': [35],
        'step2_rnn2_hidden_dim': 70, 'step2_rnn_type': 'LSTM', 'step2_num_rnn_layers': 1,
        'attention_job_encoding_dim': ENC_OUT_CHANNELS,
        'attention_hidden_dim': 55,
        'ptr_dim': 25
    }

    critic_config_args = {
        'encoder_output_dim': ENC_OUT_CHANNELS,
        'conv_channels_list': [64, 128, 64],
        'conv_kernel_sizes_list': [3, 3, 3],
        'final_fc_hidden_dims': [32]
    }

    actor = PFSPNet(encoder_args=encoder_config_args, decoder_args=decoder_config_args).to(device)
    critic = CriticNetwork(**critic_config_args).to(device)
    opt_actor = optim.Adam(actor.parameters(), lr=1e-4)
    opt_critic = optim.Adam(critic.parameters(), lr=1e-3)

    model_save_dir = 'PFSPNET/models'
    os.makedirs(model_save_dir, exist_ok=True)
    best_model_path = os.path.join(model_save_dir, f"best_actor_{config_num_jobs}j_{config_num_machines}m.pt")

    print("Model and optimizer instantiated. Starting training loop...")

    history_epoch_avg_actor_loss = []
    history_epoch_avg_critic_loss = []
    history_epoch_avg_cmax = []
    history_epoch_best_cmax = []
    global_best_cmax = float('inf')

    instance_features = P_instance.unsqueeze(-1) # 形状: (N, M, 1)
    batch_features = instance_features.unsqueeze(0).expand(config_batch_size, -1, -1, -1)
    dummy_m_scalar = torch.full((config_batch_size, 1), float(config_num_machines), device=device)

    for epoch in range(config_epochs):
        actor.train()
        critic.train()
        batch_actor_losses = []
        batch_critic_losses = []
        batch_cmaxes = []
        
        for _ in range(config_steps_per_epoch):
            actor_loss_val, critic_loss_val, batch_avg_cmax_val = train_one_batch(
                actor, critic, opt_actor, opt_critic,
                batch_features, dummy_m_scalar,
                P_instance, E_instance, R_instance, u_starts, s_durations, f_factors,
                device
            )
            
            batch_actor_losses.append(actor_loss_val)
            batch_critic_losses.append(critic_loss_val)
            batch_cmaxes.append(batch_avg_cmax_val)

        actor.eval()
        epoch_best_cmax_this_epoch = float('inf')
        with torch.no_grad():
            eval_batch_size = 128
            eval_features = instance_features.unsqueeze(0).expand(eval_batch_size, -1, -1, -1)
            eval_m_scalar = torch.full((eval_batch_size, 1), float(config_num_machines), device=device)

            selected_indices, _, _ = actor(
                eval_features, eval_m_scalar, max_decode_len=config_num_jobs
            )
            
            put_off_eval = torch.zeros(eval_batch_size, config_num_machines, config_num_jobs, device=device, dtype=torch.long)
            
            cmax_vals, _ = calculate_objectives_pytorch(
                    selected_indices, put_off_eval,
                    P_instance, E_instance, R_instance, u_starts, s_durations, f_factors,
                    device
            )
            
            min_cmax_in_batch_val = torch.min(cmax_vals)
            
            if min_cmax_in_batch_val.item() < epoch_best_cmax_this_epoch:
                epoch_best_cmax_this_epoch = min_cmax_in_batch_val.item()

            if epoch_best_cmax_this_epoch < global_best_cmax:
                global_best_cmax = epoch_best_cmax_this_epoch
                torch.save(actor.state_dict(), best_model_path)
                print(f"  ** 新的全局最优 Cmax: {global_best_cmax:.2f}。模型已保存到 {best_model_path} **")
        
        avg_epoch_actor_loss = np.mean(batch_actor_losses)
        avg_epoch_critic_loss = np.mean(batch_critic_losses)
        avg_epoch_cmax = np.mean(batch_cmaxes)

        history_epoch_avg_actor_loss.append(avg_epoch_actor_loss)
        history_epoch_avg_critic_loss.append(avg_epoch_critic_loss)
        history_epoch_avg_cmax.append(avg_epoch_cmax)
        history_epoch_best_cmax.append(epoch_best_cmax_this_epoch if epoch_best_cmax_this_epoch != float('inf') else avg_epoch_cmax)

        print("#############################################")
        print(f" Epoch {epoch+1} Summary ")
        print(f"平均 Actor 损失: {avg_epoch_actor_loss:.4f}, 平均 Critic 损失: {avg_epoch_critic_loss:.4f}, "
              f"本 Epoch 平均 Cmax: {avg_epoch_cmax:.2f}, 本 Epoch 最优 Cmax: {epoch_best_cmax_this_epoch:.2f}\n")

    print("训练完成!")
    print(f"全局最优 Cmax: {global_best_cmax:.2f}")
    print(f"最优模型已保存在: {best_model_path}")

    plot_training_results(
        epochs_range = range(1, config_epochs + 1),
        history_cmax=history_epoch_avg_cmax,
        history_best_cmax=history_epoch_best_cmax,
        history_actor_loss=history_epoch_avg_actor_loss,
        history_critic_loss=history_epoch_avg_critic_loss
    )