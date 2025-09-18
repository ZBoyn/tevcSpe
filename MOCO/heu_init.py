from uu import decode
import numpy as np
from pro_def import ProblemDefinition, Solution
from decode import Decoder
from typing import List, Tuple
import torch
import yaml
import os
import sys

from actor_critic import PFSPNet
from calc import calculate_objectives_pytorch


class Initializer:
    """根据不同策略初始化解的种群"""
    def __init__(self, problem_definition: ProblemDefinition, pop_size: int, init_params: dict):
        self.problem = problem_definition
        self.pop_size = pop_size
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
        pass

    def generate_randomly(self) -> Solution:
        pass

    def initialize_population(self, partial_solutions: List[Tuple[List[int], np.ndarray]] = None) -> List[Solution]:
        h1_count = self.params.get('h1_count', 1)
        h2_count = self.params.get('h2_count', 1)

""" 
if __name__ == "__main__":

    problem_definition = ProblemDefinition(
        processing_times=np.array([[4, 6, 5, 4, 5], [6, 4, 4, 5, 4], [5, 4, 3, 4, 4]]).T,
        power_consumption=np.array([[7, 3, 4, 2, 7], [4, 3, 3, 6, 2], [4, 3, 6, 3, 6]]).T,
        release_times=np.array([0, 8, 18, 16, 20]),
        period_start_times=np.array([0, 15, 45, 65]),
        period_prices=np.array([2, 8, 3]),
    )
    
    initializer = Initializer(problem_definition, 10, {'h1_count': 1, 'h2_count': 1})
    population = initializer.generate_with_heuristic1()
    print(population)
"""