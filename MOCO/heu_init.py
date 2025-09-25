import numpy as np
from pro_def import ProblemDefinition, Solution
from decode import Decoder
from typing import List, Tuple
import torch
import yaml
import os

from actor_critic import PFSPNet
from calc import calculate_objectives_pytorch


class Initializer:
    """根据不同策略初始化解的种群"""
    def __init__(self, problem_definition: ProblemDefinition, moead_params: dict, init_params: dict):
        self.problem = problem_definition
        self.pop_size = moead_params["population_size"]
        self.params = init_params
        self.decoder = Decoder(problem_definition)
        
        with open('MOCO/params.yaml', 'r') as f:
            self.config = yaml.safe_load(f)

    def _generate_sequence(self, seq_type: str = "Min_Cmax") -> np.ndarray:
        """使用PFSPNET推理, 生成比较好的初始序列 sequence"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        P_instance = torch.tensor(self.problem.processing_times, dtype=torch.float32).to(device)
        E_instance = torch.tensor(self.problem.power_consumption, dtype=torch.float32).to(device)
        R_instance = torch.tensor(self.problem.release_times, dtype=torch.float32).to(device)
        u_bounds = torch.tensor(self.problem.period_start_times, dtype=torch.float32).to(device)
        
        f_factors = torch.tensor(self.problem.period_prices, dtype=torch.float32).to(device)

        config_num_jobs = self.problem.num_jobs
        config_num_machines = self.problem.num_machines

        model_params = self.config['pfspnet_model_config']
        enc_part1_args = model_params['encoder']['part1']
        enc_part2_args = model_params['encoder']['part2']
        enc_part3_args = model_params['encoder']['part3']
        encoder_config_args = {'part1_args': enc_part1_args, 'part2_args': enc_part2_args, 'part3_args': enc_part3_args}
        
        decoder_params = model_params['decoder']
        dec_step1_pt_enc_args = decoder_params['step1_pt_encoder']
        decoder_config_args = {
            'step1_pt_encoder_args': dec_step1_pt_enc_args,
            'step1_m_embedding_dim': decoder_params['step1_m_embedding_dim'], 
            'step1_di_output_dim': decoder_params['step1_di_output_dim'], 
            'step1_fc_hidden_dims': decoder_params['step1_fc_hidden_dims'],
            'step2_rnn2_hidden_dim': decoder_params['step2_rnn2_hidden_dim'], 
            'step2_rnn_type': decoder_params['step2_rnn_type'], 
            'step2_num_rnn_layers': decoder_params['step2_num_rnn_layers'],
            'attention_job_encoding_dim': decoder_params['attention_job_encoding_dim'],
            'attention_hidden_dim': decoder_params['attention_hidden_dim'],
            'ptr_dim': decoder_params['ptr_dim']
        }
        
        actor_model = PFSPNet(encoder_args=encoder_config_args, decoder_args=decoder_config_args).to(device)
        
        inference_params = self.config['inference_params']
        if seq_type == "Min_Cmax":
            inference_params['model_path'] = inference_params['model_path_cmax']
        elif seq_type == "Min_TEC":
            inference_params['model_path'] = inference_params['model_path_tec']
        
        model_path = os.path.join(os.path.dirname(__file__), inference_params['model_path'])
        actor_model.load_state_dict(torch.load(model_path, map_location=device))
        actor_model.eval()
        
        num_samples = inference_params['num_samples']
        instance_features = P_instance.unsqueeze(-1)
        batch_features = instance_features.unsqueeze(0).expand(num_samples, -1, -1, -1)
        dummy_m_scalar = torch.full((num_samples, 1), float(config_num_machines), device=device)

        with torch.no_grad():
            candidate_sequences, _, _ = actor_model(
                batch_features, dummy_m_scalar, max_decode_len=config_num_jobs
            )
            
            put_off_eval = torch.zeros(num_samples, config_num_machines, config_num_jobs, device=device, dtype=torch.long)
            
            cmax_values, _ = calculate_objectives_pytorch(
                job_sequences=candidate_sequences,
                put_off_matrices=put_off_eval,
                P=P_instance, E=E_instance, R=R_instance, u_bounds=u_bounds, f=f_factors,
                device=device
            )
            
            _, best_idx = torch.min(cmax_values, dim=0)
            best_sequence = candidate_sequences[best_idx].cpu().numpy()

        return best_sequence

    def _perturb_solution(self, solution: Solution, seq_swap_num: int = 2, mode_flip_num: int = 3) -> Solution:
        new_sol = solution.copy()
        
        # Sequence Perturbation
        for _ in range(seq_swap_num):
            idx1, idx2 = np.random.choice(self.problem.num_jobs, 2, replace=False)
            new_sol.sequence[idx1], new_sol.sequence[idx2] = new_sol.sequence[idx2], new_sol.sequence[idx1]

        # Mode Perturbation
        for _ in range(mode_flip_num):
            m_idx = np.random.randint(self.problem.num_machines)
            j_idx = np.random.randint(self.problem.num_jobs)
            new_sol.mode[j_idx, m_idx] = 1 - new_sol.mode[j_idx, m_idx]

        new_sol = self.decoder.decode(new_sol)
        return new_sol

    def generate_with_heuristic1(self) -> Solution:
        """ 生成Cmax最优的初始解 即在推理得到的sequence基础上 全部开启高功耗 零推迟"""
        best_sequence = self._generate_sequence()

        high_modes = np.ones((self.problem.num_jobs, self.problem.num_machines), dtype=int)
        no_put_off = np.zeros((self.problem.num_jobs, self.problem.num_machines), dtype=int)
        
        initial_solution = Solution(
            sequence=best_sequence.copy(),
            mode=high_modes,
            put_off=no_put_off
        )

        initial_solution = self.decoder.decode(initial_solution)
        return initial_solution

    def generate_with_heuristic2(self) -> Solution:
        """ 生成TEC最优的初始解 只对第一个遇到的高电价操作进行一次延迟 """
        best_sequence = self._generate_sequence(seq_type="Min_Cmax")
        low_modes = np.zeros((self.problem.num_jobs, self.problem.num_machines), dtype=int)
        put_off_matrix = np.zeros((self.problem.num_jobs, self.problem.num_machines), dtype=int)

        temp_solution = Solution(sequence=best_sequence.copy(), mode=low_modes.copy(), put_off=put_off_matrix.copy())
        temp_solution = self.decoder.decode(temp_solution)

        prices = self.problem.period_prices
        max_price = np.max(prices)
        high_price_periods_indices = np.where(prices == max_price)[0]

        delay_applied = False
        for job_idx in best_sequence:
            for machine_idx in range(self.problem.num_machines):
                start_time = temp_solution.start_times[job_idx, machine_idx]
                current_period_idx = np.searchsorted(self.problem.period_start_times, start_time, side='right') - 1

                if current_period_idx in high_price_periods_indices:
                    next_cheaper_period_start = -1
                    for p_idx in range(current_period_idx + 1, len(prices)):
                        if prices[p_idx] < prices[current_period_idx]:
                            next_cheaper_period_start = self.problem.period_start_times[p_idx]
                            break
                    
                    if next_cheaper_period_start != -1:
                        put_off_matrix[job_idx, machine_idx] = next_cheaper_period_start
                        delay_applied = True
                        break
            
            if delay_applied:
                break

        final_solution = Solution(
            sequence=best_sequence.copy(),
            mode=low_modes,
            put_off=put_off_matrix
        )
        final_solution = self.decoder.decode(final_solution)
        return final_solution

    def generate_with_heuristic3(self) -> Solution:
        """ 生成TE最优的初始解 """
        best_sequence = self._generate_sequence(seq_type="Min_Cmax")
        if self.problem.HIGH_MODE_POWER_FACTOR * self.problem.HIGH_MODE_SPEED_FACTOR > 1:
            mode = np.zeros((self.problem.num_jobs, self.problem.num_machines), dtype=int)
        else:
            mode = np.ones((self.problem.num_jobs, self.problem.num_machines), dtype=int)
        
        no_put_off = np.zeros((self.problem.num_jobs, self.problem.num_machines), dtype=int)
        initial_solution = Solution(
            sequence=best_sequence.copy(),
            mode=mode,
            put_off=no_put_off
        )
        initial_solution = self.decoder.decode(initial_solution)
        return initial_solution
    
    def generate_randomly(self) -> Solution:
        """ 随机生成初始解 """
        sequence = np.random.permutation(self.problem.num_jobs)
        mode = np.random.randint(0, 2, size=(self.problem.num_jobs, self.problem.num_machines))
        put_off = np.zeros(shape=(self.problem.num_jobs, self.problem.num_machines), dtype=int)
        
        initial_solution = Solution(sequence=sequence, mode=mode, put_off=put_off)
        initial_solution = self.decoder.decode(initial_solution)
        return initial_solution

    def initialize_population(self, partial_solutions: List[Tuple[List[int], np.ndarray]] = None) -> List[Solution]:
        population = []

        # Heuristic Solution
        h1_count = self.params.get('h1_count', 1)
        h2_count = self.params.get('h2_count', 1)
        h3_count = self.params.get('h3_count', 1)
        # Perturbed Solution
        h1_perturb_count = self.params.get('h1_perturb_count', 2)
        h2_perturb_count = self.params.get('h2_perturb_count', 2)
        h3_perturb_count = self.params.get('h3_perturb_count', 2)

        s1 = self.generate_with_heuristic1()
        for _ in range(h1_count):
            population.append(s1.copy())
        for _ in range(h1_perturb_count):
            population.append(self._perturb_solution(s1))

        s2 = self.generate_with_heuristic2()
        for _ in range(h2_count):
            population.append(s2.copy())
        for _ in range(h2_perturb_count):
            population.append(self._perturb_solution(s2))

        s3 = self.generate_with_heuristic3()
        for _ in range(h3_count):
            population.append(s3.copy())
        for _ in range(h3_perturb_count):
            population.append(self._perturb_solution(s3))

        current_pop_size = len(population)
        num_random = self.pop_size - current_pop_size
        for _ in range(max(0, num_random)):
            population.append(self.generate_randomly())

        return population[:self.pop_size]

        


""" 
if __name__ == "__main__":

    problem_definition = ProblemDefinition(
        processing_times=np.array([[4, 6, 5, 4, 5], [6, 4, 4, 5, 4], [5, 4, 3, 4, 4]]).T,
        power_consumption=np.array([[7, 3, 4, 2, 7], [4, 3, 3, 6, 2], [4, 3, 6, 3, 6]]).T,
        release_times=np.array([0, 8, 18, 16, 20]),
        period_start_times=np.array([0, 15, 45, 65]),
        period_prices=np.array([2, 8, 3]),
    )
    
    initializer = Initializer(problem_definition, 3, {'h1_count': 1, 'h2_count': 1, 'h3_count': 1, 'h1_perturb_count': 0, 'h2_perturb_count': 0, 'h3_perturb_count': 0})
    population = initializer.initialize_population()
    print(population)
 """