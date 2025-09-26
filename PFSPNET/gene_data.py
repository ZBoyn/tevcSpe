import torch
import os
import numpy as np

def generate_and_save_instance(
    num_jobs, 
    num_machines, 
    k_intervals, 
    num_days=1,
    save_path='.'
):
    os.makedirs(save_path, exist_ok=True)

    device = torch.device("cpu")

    P_instance = torch.rand((num_jobs, num_machines), device=device) * 9 + 1
    E_instance = torch.rand((num_jobs, num_machines), device=device) * 4 + 1
    R_instance = torch.randint(0, 5 * num_jobs, (num_jobs,), device=device, dtype=torch.float)
    
    s_durations_oneday = torch.randint(
        low=20, high=81, size=(k_intervals,), device=device, dtype=torch.float
    )
    f_factors_oneday = torch.randint(1, 7, (k_intervals,), device=device, dtype=torch.float)

    s_durations = s_durations_oneday.repeat(num_days)
    f_factors = f_factors_oneday.repeat(num_days)
    
    u_starts = torch.cat([
        torch.tensor([0.0], device=device),
        torch.cumsum(s_durations[:-1], dim=0)
    ])
    
    total_k_intervals = k_intervals * num_days

    instance_data = {
        'P_instance': P_instance,
        'E_instance': E_instance,
        'R_instance': R_instance,
        'u_starts': u_starts,
        's_durations': s_durations,
        'f_factors': f_factors,
        'num_jobs': num_jobs,
        'num_machines': num_machines,
        'k_intervals': total_k_intervals 
    }
    
    instance_id = np.random.randint(10000, 99999)
    file_name = f"instance_{num_jobs}j_{num_machines}m_{k_intervals}k_{num_days}d_{instance_id}.pt"
    full_path = os.path.join(save_path, file_name)
    
    torch.save(instance_data, full_path)
    return full_path

if __name__ == '__main__':
    base_save_directory = 'pfspnet/data'
    
    job_sizes = [20, 50]
    machine_sizes = [5, 10]
    instances_per_size = 100
    config_num_days = 3

    total_generated = 0
    for n_jobs in job_sizes:
        for n_machines in machine_sizes:
            size_dir_name = f"{n_jobs}j_{n_machines}m"
            
            save_directory = os.path.join(base_save_directory, 'train', size_dir_name)
            
            print(f"--- Generating {instances_per_size} instances for size {n_jobs}j x {n_machines}m ---")
            for i in range(instances_per_size):
                config_k_intervals_per_day = np.random.randint(3, 6)
                
                generate_and_save_instance(
                    num_jobs=n_jobs,
                    num_machines=n_machines,
                    k_intervals=config_k_intervals_per_day,
                    num_days=config_num_days,
                    save_path=save_directory
                )
                total_generated += 1
    
    print(f"\n✅ Generation complete. Total instances generated: {total_generated}")