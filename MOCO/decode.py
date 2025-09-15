from ast import Tuple, While
import numpy as np
from pro_def import ProblemDefinition, Solution

class Decoder:
    def __init__(self, problem_definition: ProblemDefinition):
        self.problem = problem_definition

    def decode(self, solution: Solution, mode: str = "normal"):
        completion_times = np.zeros((self.problem.num_jobs, self.problem.num_machines), float)
        
        if mode == "normal":
            for j_idx, job_id in enumerate(solution.sequence):
                for machine_id in range(self.problem.num_machines):
                    proc_time = self.problem.processing_times[job_id, machine_id]
                    prev_job_id = solution.sequence[j_idx - 1] if j_idx > 0 else -1

                    est_from_prev_job = completion_times[prev_job_id, machine_id] if prev_job_id != -1 else 0
                    est_from_prev_machine = completion_times[job_id, machine_id - 1] if machine_id > 0 else 0
                    
                    est = max(est_from_prev_job, est_from_prev_job, self.problem.release_times[job_id])

                    base_period_idx = np.searchsorted(self.problem.period_start_times, est, side='right') - 1
                    target_period_idx = min(base_period_idx + solution.put_off[job_id, machine_id], self.problem.num_periods - 1)
                    target_period_start_time = self.problem.period_start_times[target_period_idx]
                    delayed_est = max(target_period_start_time, est)

                    # 找到能容纳的实际开始加工时间
                    actual_start_time = self._findCompletePeriod(delayed_est, proc_time)
                    completion_times[job_id, machine_id] = actual_start_time + proc_time
            
            # 应用惩罚
            if completion_times.max() > self.problem.period_start_times[-1]:
                solution.objectives = np.array([np.inf, np.inf, np.inf])
                return solution.objectives


            # 计算TEC


            # 计算TE



            







        if mode == "localRightShift":
            pass
        
        return solution, np.array([0.0, 0.0, 0.0])


    def _findCompletePeriod(self, delayed_est: float, proc_time: float):
        # 找到能容纳加工完成的时间段的工件开始时间
        actual_start_time = delayed_est

        while True:
            p_idx = np.searchsorted(self.problem.period_start_times, actual_start_time, side='right') - 1
            p_idx = max(0, p_idx)

            if p_idx + 1 < len(self.problem.period_start_times):
                p_end_time = self.problem.period_start_times[p_idx + 1]
            else:
                p_end_time = np.inf

            if actual_start_time + proc_time <= p_end_time:
                actual_start_time = delayed_est
                break
            else:
                if p_idx + 1 < len(self.problem.period_start_times):
                    actual_start_time = self.problem.period_start_times[p_idx + 1]
                else:
                    actual_start_time = np.inf


            delayed_est = p_end_time

if __name__ == "__main__":
    #############################################################
    #                  Test Decoder Fuction                     #
    # ###########################################################
    print("test")