import numpy as np
from pro_def import ProblemDefinition, Solution
from decode import Decoder

from typing import List, Dict, Tuple


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