import torch
import numpy as np

Num_jobs = 10
Num_machines = 3
Num_periods = 10

data = torch.load(f"MIP/data/PyTorch/instance_{Num_jobs}j_{Num_machines}m_{Num_periods}k.pt", map_location="cpu")


n_jobs = Num_jobs
m_machines = Num_machines
k_intervals = Num_periods

u = data['u_starts'].numpy()
s = data['s_durations'].numpy()
f = data['f_factors'].numpy()

release_times = [None] + data['R_instance'].tolist()

p_numpy = data['P_instance'].numpy()
base_processing_time = np.vstack([
    np.full(m_machines, None, dtype=object),
    p_numpy
])

e_numpy = data['E_instance'].numpy()
base_energy = np.vstack([
    np.full(m_machines, None, dtype=object),
    e_numpy
])

speedFactor = 1
energyFactor = 1.5
e_idle = 1
