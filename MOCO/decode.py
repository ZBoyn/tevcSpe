import re
import numpy as np
from pro_def import ProblemDefinition, Solution

class Decoder:
    def __init__(self, problem_definition: ProblemDefinition):
        self.problem = problem_definition

    def decode(self, solution: Solution, mode: str = "normal") -> Solution:
        if mode == "normal":
            results = self._calculate_schedule_and_objectives(solution)
            
            solution.start_times = results["start_times"]
            solution.completion_times = results["completion_times"]
            solution.objectives = results["objectives"]
        
        elif mode == "LocalOpTEC":
            pass
        
        elif mode == "RightOpTE":
            pass
        return solution

    def _calculate_schedule_and_objectives(self, solution: Solution) -> dict:
        start_times = np.zeros((self.problem.num_jobs, self.problem.num_machines))
        completion_times = np.zeros((self.problem.num_jobs, self.problem.num_machines))
        
        machine_start_times = np.full(self.problem.num_machines, np.inf)
        machine_end_times = np.zeros(self.problem.num_machines)

        processing_energy = 0.0
        processing_cost = 0.0

        for j_idx, job_id in enumerate(solution.sequence):
            for m_id in range(self.problem.num_machines):
                proc_time = self.problem.processing_times[job_id, m_id]
                mode_val = solution.mode[job_id, m_id]
                speed_factor = self.problem.speed_factors[mode_val]
                actual_proc_time = proc_time * speed_factor

                est_from_prev_job = completion_times[solution.sequence[j_idx - 1], m_id] if j_idx > 0 else 0
                est_from_prev_machine = completion_times[job_id, m_id - 1] if m_id > 0 else 0
                est = max(est_from_prev_job, est_from_prev_machine, self.problem.release_times[job_id])
                
                base_period_idx = np.searchsorted(self.problem.period_start_times, est, side='right') - 1
                put_off_periods = solution.put_off[job_id, m_id]
                target_period_idx = min(base_period_idx + put_off_periods, self.problem.num_periods - 1)
                target_period_start_time = self.problem.period_start_times[target_period_idx]
                delayed_est = max(target_period_start_time, est)

                actual_start_time = self._find_valid_start_time(delayed_est, actual_proc_time)
                
                if np.isinf(actual_start_time):
                    return {
                        "start_times": start_times, "completion_times": completion_times,
                        "objectives": np.full(3, np.inf, dtype=float)
                    }

                start_times[job_id, m_id] = actual_start_time
                completion_times[job_id, m_id] = actual_start_time + actual_proc_time

                machine_start_times[m_id] = min(machine_start_times[m_id], actual_start_time)
                machine_end_times[m_id] = max(machine_end_times[m_id], completion_times[job_id, m_id])

                power_factor = self.problem.power_factors[mode_val]
                actual_power = self.problem.power_consumption[job_id, m_id] * power_factor
                op_energy = actual_proc_time * actual_power
                
                period_idx = np.searchsorted(self.problem.period_start_times, actual_start_time, side='right') - 1
                price = self.problem.period_prices[max(0, period_idx)]
                op_cost = op_energy * price

                processing_energy += op_energy
                processing_cost += op_cost
        
        idle_energy = 0.0
        idle_cost = 0.0
        for m_id in range(self.problem.num_machines):
            total_active_time = machine_end_times[m_id] - machine_start_times[m_id]
            total_processing_time_on_machine = np.sum(
                [self.problem.processing_times[j, m_id] * self.problem.speed_factors[solution.mode[j, m_id]] for j in solution.sequence]
            )
            idle_time = total_active_time - total_processing_time_on_machine
            idle_energy += idle_time * self.problem.IDLE_MODE_POWER

            sorted_jobs_on_machine = sorted(solution.sequence, key=lambda j: start_times[j, m_id])
            idle_start = machine_start_times[m_id]
            idle_end = start_times[sorted_jobs_on_machine[0], m_id]
            idle_cost += self._get_cost_for_interval(idle_start, idle_end, self.problem.IDLE_MODE_POWER)
            for i in range(len(sorted_jobs_on_machine) - 1):
                idle_start = completion_times[sorted_jobs_on_machine[i], m_id]
                idle_end = start_times[sorted_jobs_on_machine[i+1], m_id]
                idle_cost += self._get_cost_for_interval(idle_start, idle_end, self.problem.IDLE_MODE_POWER)

        cmax = np.max(completion_times)
        total_energy = processing_energy + idle_energy
        total_energy_cost = processing_cost + idle_cost

        return {
            "start_times": start_times,
            "completion_times": completion_times,
            "objectives": np.array([round(cmax, 2), round(total_energy, 2), round(total_energy_cost, 2)])
        }


    def _find_valid_start_time(self, est: float, proc_time: float) -> float:
        current_start_time = est
        while True:
            period_idx = np.searchsorted(self.problem.period_start_times, current_start_time, side='right') - 1
            period_idx = max(0, period_idx)
            if period_idx >= self.problem.num_periods: return np.inf
            period_end_time = self.problem.period_start_times[period_idx + 1]
            if current_start_time + proc_time <= period_end_time: return current_start_time
            else:
                if period_idx + 1 >= len(self.problem.period_start_times) - 1: return np.inf
                current_start_time = self.problem.period_start_times[period_idx + 1]
    
    def _get_cost_for_interval(self, start_time: float, end_time: float, power: float) -> float:
        if start_time >= end_time: return 0.0
        cost = 0
        current_time = start_time
        while current_time < end_time:
            period_idx = np.searchsorted(self.problem.period_start_times, current_time, side='right') - 1
            period_idx = max(0, period_idx)
            if period_idx >= self.problem.num_periods: break
            period_end_time = self.problem.period_start_times[period_idx + 1]
            price = self.problem.period_prices[period_idx]
            duration_in_period = min(end_time, period_end_time) - current_time
            cost += duration_in_period * power * price
            current_time += duration_in_period
        return cost

""" 
if __name__ == "__main__":
    from data_manager import load_instance
    # Problem: 当前的编码方式并不能完全的解码出所有信息 eg.
    # 1. 注意到TE每台机器上的开始时间是主动延后的 仅用put_off无法完全解码出来
    
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
    # start_times = decoded_solution.start_times
    # complete_times = decoded_solution.completion_times
    print(f"Cmax: {objectives[0]:.2f}")
    print(f"TE: {objectives[1]:.2f}")
    print(f"TEC: {objectives[2]:.2f}")
    # print(f"start_times: {start_times}")
    # print(f"complete_times: {complete_times}")

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
    start_times = decoded_solution.start_times
    complete_times = decoded_solution.completion_times
    print(f"Cmax: {objectives[0]:.2f}")
    print(f"TE: {objectives[1]:.2f}")
    print(f"TEC: {objectives[2]:.2f}")
    print(f"start_times: {start_times}")
    print(f"complete_times: {complete_times}")
    print("##########################################################")
"""