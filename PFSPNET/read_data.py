import torch

Num_jobs = 20
Num_machines = 5
Num_periods = 10

data = torch.load(f"PFSPNET/data/instance_{Num_jobs}j_{Num_machines}m_{Num_periods}k.pt", map_location="cpu")
print(data)

P = data['P_instance']
E = data['E_instance']
u = data['u_starts']
