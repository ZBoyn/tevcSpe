import torch
import torch.nn as nn
import torch.nn.functional as F
from actor_critic import PFSPNet, CriticNetwork
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from calc import calculate_objectives_pytorch

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