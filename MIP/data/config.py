import numpy as np

n_jobs = 5
m_machines = 3
k_intervals = 3

u = np.array([0, 15, 45]) 
s = np.array([15, 30, 20])
f = np.array([2, 8, 3])   

release_times = [None, 0, 8, 18, 16, 20]
base_processing_time = np.array([
    [None, None, None],
    [4, 6, 5],
    [6, 4, 4],
    [5, 4, 3],
    [4, 5, 4],
    [5, 4, 4],
])

base_energy = np.array([
    [None, None, None],
    [7, 4, 4],
    [3, 3, 3],
    [4, 3, 6],
    [2, 6, 3],
    [7, 2, 6],
])

speedFactor = 0.7
energyFactor = 1.5
e_idle = 1
