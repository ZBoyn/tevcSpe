import os
import json
import csv
import numpy as np
from datetime import datetime
from typing import List

def save_results(population: List, params: dict, instance_name: str, base_dir: str = "MOCO/results"):
    objectives = np.array([sol.objectives for sol in population])
    num_solutions = objectives.shape[0]
    is_dominated = np.zeros(num_solutions, dtype=bool)
    for i in range(num_solutions):
        for j in range(num_solutions):
            if i == j:
                continue
            if np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                is_dominated[i] = True
                break
    
    pareto_solutions = [sol for i, sol in enumerate(population) if not is_dominated[i]]
    pareto_objectives = np.array([sol.objectives for sol in pareto_solutions])

    if pareto_objectives.shape[0] > 0:
        unique_objectives, unique_indices = np.unique(pareto_objectives, axis=0, return_index=True)
        unique_solutions = [pareto_solutions[i] for i in unique_indices]
    else:
        unique_objectives = np.array([])
        unique_solutions = []

    print(f"Found {len(pareto_solutions)} non-dominated solutions. After deduplication, {len(unique_solutions)} unique solutions remain.")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(base_dir, f"{instance_name}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    csv_path = os.path.join(result_dir, "objectives.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Cmax', 'TEC', 'TE'])
        writer.writerows(unique_objectives)
    print(f"Objectives saved to {csv_path}")

    if unique_solutions:
        sequences = np.array([sol.sequence for sol in unique_solutions])
        modes = np.array([sol.mode for sol in unique_solutions])
        put_offs = np.array([sol.put_off for sol in unique_solutions])
        
        npz_path = os.path.join(result_dir, "solutions.npz")
        np.savez(npz_path, sequence=sequences, mode=modes, put_off=put_offs)
        print(f"Solutions saved to {npz_path}")

    json_path = os.path.join(result_dir, "parameters.json")
    with open(json_path, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"Parameters saved to {json_path}")
