import numpy as np
from typing import List, Dict, Tuple
import random

from pro_def import ProblemDefinition, Solution
from decode import Decoder



class Operation:
    def __init__(self, problem_definition: ProblemDefinition):
        self.problem = problem_definition
        self.decoder = Decoder(problem_definition)

    def _find_critical_path(self, solution: Solution) -> List[Tuple[int, int]]:
        if solution.completion_times is None or solution.prev is None:
            raise ValueError("Solution must be decoded first to have completion_times and prev.")

        seq_map = {job_id: i for i, job_id in enumerate(solution.sequence)}

        cmax_job_id = solution.sequence[-1]
        cmax_machine_id = self.problem.num_machines - 1
        
        critical_path = []
        curr_job, curr_machine = int(cmax_job_id), int(cmax_machine_id)

        while True:
            critical_path.append((curr_job, curr_machine))
            predecessor_type = solution.prev[curr_job, curr_machine]

            if predecessor_type.endswith("H"):
                curr_machine -= 1
            elif predecessor_type.endswith("B"):
                curr_job_seq_idx = seq_map.get(curr_job)
                if curr_job_seq_idx is None or curr_job_seq_idx == 0:
                    break 
                curr_job = solution.sequence[curr_job_seq_idx - 1]
            else: 
                break
            
            if curr_machine < 0:
                break
                
        return critical_path[::-1]

    def _identify_block_starts(self, solution: Solution) -> Dict[int, List[int]]:
        if solution.prev is None:
            raise ValueError("Solution must be decoded first to have a prev matrix.")

        num_machines = self.problem.num_machines
        block_starts = {m: [] for m in range(num_machines)}

        for m_id in range(num_machines):
            for job_id in solution.sequence:
                predecessor_type = solution.prev[job_id, m_id]
                if predecessor_type not in ["B", "TB"]:
                    block_starts[m_id].append(job_id)
        
        return block_starts

    def _find_common_blocks(self, sequence: np.ndarray, block_starts_per_machine: Dict[int, List[int]]) -> List[List[int]]:
        if not sequence.any() or not block_starts_per_machine:
            return []

        seq_map = {job_id: i for i, job_id in enumerate(sequence)}

        all_cut_indices = set()
        for machine_id, start_jobs in block_starts_per_machine.items():
            for job_id in start_jobs:
                if job_id in seq_map:
                    all_cut_indices.add(seq_map[job_id])

        sorted_indices = sorted(list(all_cut_indices))

        if not sorted_indices:
             if len(sequence) > 1:
                 return [list(sequence)]
             else:
                 return []


        common_blocks = []
        
        for i in range(len(sorted_indices) - 1):
            start_idx = sorted_indices[i]
            end_idx = sorted_indices[i+1]
            block = list(sequence[start_idx:end_idx])
            common_blocks.append(block)

        last_start_idx = sorted_indices[-1]
        last_block = list(sequence[last_start_idx:])
        common_blocks.append(last_block)
        
        final_blocks = [block for block in common_blocks if len(block) > 1]

        return final_blocks

    def _find_common_blocks_optimized(self, solution: Solution) -> List[List[int]]:
        if solution.prev is None or not solution.sequence.any():
            raise ValueError("Solution must be decoded and have a valid sequence and prev matrix.")
        
        seq_map = {job_id: i for i, job_id in enumerate(solution.sequence)}
        all_cut_indices = set()

        # O(M * N)
        for m_id in range(self.problem.num_machines):
            for job_id in solution.sequence:
                predecessor_type = solution.prev[job_id, m_id]
                if predecessor_type not in ["B", "TB"]:
                    all_cut_indices.add(seq_map[job_id])
        
        # O(N log N)
        sorted_indices = sorted(list(all_cut_indices))

        if not sorted_indices:
            return [list(solution.sequence)] if len(solution.sequence) > 1 else []

        common_blocks = []
        for i in range(len(sorted_indices) - 1):
            start_idx = sorted_indices[i]
            end_idx = sorted_indices[i+1]
            common_blocks.append(list(solution.sequence[start_idx:end_idx]))

        last_start_idx = sorted_indices[-1]
        common_blocks.append(list(solution.sequence[last_start_idx:]))
        
        return [block for block in common_blocks if len(block) > 1]

    def select_blocks(self, blocks: List[List[int]], k: int = 1, method: str = 'longest') -> List[List[int]]:
        if not blocks or k <= 0:
            return []
        
        num_blocks_to_select = min(k, len(blocks))

        if method == 'longest':
            sorted_blocks = sorted(blocks, key=len, reverse=True)
            return sorted_blocks[:num_blocks_to_select]
        elif method == 'random':
            return random.sample(blocks, num_blocks_to_select)
        else:
            raise ValueError("Method must be 'longest' or 'random'.")

""" 
if __name__ == "__main__":
    from data_manager import load_instance
    
    print("##########################################################")
    try:
        problem = load_instance("MOCO\\data\\instance_5j_3m.npz")
    except FileNotFoundError:
        print("File not found")
        exit()
    
    print("-" * 20)

    cmax_sequence = np.array([0, 1, 3, 2, 4])
    cmax_mode = np.ones((problem.num_jobs, problem.num_machines), dtype=int)
    cmax_mode[0, 2] = 0
    cmax_mode[1, 0] = 0
    cmax_put_off = np.zeros((problem.num_jobs, problem.num_machines), dtype=int)

    cmax_solution = Solution(
        sequence=cmax_sequence,
        mode=cmax_mode,
        put_off=cmax_put_off
    )
    print("-" * 20)
    print("Cmax:")

    decoder = Decoder(problem)
    decoded_solution = decoder.decode(cmax_solution)

    objectives = decoded_solution.objectives
    print(f"prev:{decoded_solution.prev}")
    print(f"Cmax: {objectives[0]:.2f}")
    print(f"TE: {objectives[1]:.2f}")
    print(f"TEC: {objectives[2]:.2f}")

    operation = Operation(problem)
    critical_path = operation._find_critical_path(decoded_solution)
    block_starts = operation._identify_block_starts(decoded_solution)
    block = operation._find_common_blocks(cmax_sequence,block_starts)
    print(f"critical_path:{critical_path}")
    print(f"block_starts:{block_starts}")
    print(f"block:{block}")

    print("##########################################################")

    te_sequence = np.array([0, 3, 4, 1, 2])
    te_mode = np.zeros((problem.num_jobs, problem.num_machines), dtype=int)
    te_put_off = np.zeros((problem.num_jobs, problem.num_machines), dtype=int)
    te_put_off[0, 0] = 1
    te_solution = Solution(
        sequence=te_sequence,
        mode=te_mode,
        put_off=te_put_off
    )
    print("-" * 20)
    print("TE:")

    decoded_solution = decoder.decode(te_solution)

    objectives = decoded_solution.objectives
    print(f"prev:{decoded_solution.prev}")
    print(f"Cmax: {objectives[0]:.2f}")
    print(f"TE: {objectives[1]:.2f}")
    print(f"TEC: {objectives[2]:.2f}")

    print("-" * 20)
    critical_path = operation._find_critical_path(decoded_solution)
    block_starts = operation._identify_block_starts(decoded_solution)
    block = operation._find_common_blocks(te_sequence,block_starts)
    print(f"critical_path:{critical_path}")
    print(f"block_starts:{block_starts}")
    print(f"block:{block}")
    print("##########################################################")
"""