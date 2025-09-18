import torch
import os

def generate_and_save_instance(
    num_jobs, 
    num_machines, 
    k_intervals, 
    save_path='.'
):
    os.makedirs(save_path, exist_ok=True)

    device = torch.device("cpu")

    P_instance = torch.rand(num_jobs, num_machines, device=device) * 20 + 1
    E_instance = torch.rand(num_jobs, num_machines, device=device) * 10
    R_instance = torch.randint(0, 50, (num_jobs,), device=device, dtype=torch.float)
    
    u_starts = torch.arange(0, 50 * k_intervals, 50, device=device, dtype=torch.float)
    s_durations = torch.full((k_intervals,), 50, device=device)
    f_factors = torch.randint(1, 6, (k_intervals,), device=device, dtype=torch.float)

    instance_data = {
        'P_instance': P_instance,
        'E_instance': E_instance,
        'R_instance': R_instance,
        'u_starts': u_starts,
        's_durations': s_durations,
        'f_factors': f_factors,
        'num_jobs': num_jobs,
        'num_machines': num_machines,
        'k_intervals': k_intervals
    }
    
    file_name = f"instance_{num_jobs}j_{num_machines}m_{k_intervals}k.pt"
    full_path = os.path.join(save_path, file_name)
    
    torch.save(instance_data, full_path)
    return full_path

if __name__ == '__main__':
    config_num_jobs = 20
    config_num_machines = 5
    config_k_intervals = 10
    
    save_directory = 'pfspnet/data'
    
    generate_and_save_instance(
        num_jobs=config_num_jobs,
        num_machines=config_num_machines,
        k_intervals=config_k_intervals,
        save_path=save_directory
    )
