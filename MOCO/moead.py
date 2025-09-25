import numpy as np
import random
import math
from itertools import combinations
from typing import List
from tqdm import tqdm

from pro_def import ProblemDefinition, Solution
from decode import Decoder
from heu_init import Initializer
from operation import Operation
from util.plot_pf import plot_pareto_front
from util.results_manager import save_results

class MOEAD:
    def __init__(self,
                 problem: ProblemDefinition,
                 operation: Operation,
                 decoder: Decoder,
                 initializer: Initializer,
                 moead_params: dict,

                 
    ):

        self.problem = problem
        self.operation = operation
        self.decoder = decoder
        self.initializer = initializer
        self.N = moead_params["population_size"]
        self.T = moead_params["neighborhood_size"]
        self.max_gen = moead_params["max_generations"]
        self.cr = moead_params["crossover_rate"]
        self.mr = moead_params["mutation_rate"]
        self.num_objectives = 3

        self.block_cr_prob = moead_params["block_crossover_prob"]
        self.mode_repair_prob = moead_params["mode_repair_prob"]
        self.ls_prob = moead_params["local_search_prob"]
        self.mode_repair_high_probability = moead_params["mode_repair_high_probability"]
        
        # 初始化权重向量
        self.weights = self._initialize_weights()

        # 初始化邻域
        self.neighborhoods = self._calculate_neighborhoods()

        # 初始化种群
        self.population = self._initialize_population()
        
        # 初始化边界
        self.z = np.full(self.num_objectives, np.inf)
        self._update_ideal_point_with_population()

        self.z_nad = np.full(self.num_objectives, -np.inf)
        self._update_nadir_point_with_population()

    def _initialize_weights(self) -> np.ndarray:
        """使用 Das-Dennis 方法生成均匀分布的权重向量"""
        H = math.floor((math.comb(self.N + self.num_objectives - 1, self.num_objectives - 1))**(1/(self.num_objectives-1))) -1
        while math.comb(H + self.num_objectives - 1, self.num_objectives - 1) < self.N:
            H += 1
        
        weights = []
        for combo in combinations(range(H + self.num_objectives - 1), self.num_objectives - 1):
            temp_w = np.zeros(self.num_objectives)
            temp_w[0] = combo[0]
            for i in range(1, len(combo)):
                temp_w[i] = combo[i] - combo[i-1]
            temp_w[-1] = H - combo[-1]
            weights.append(temp_w / H)

        while len(weights) < self.N:
            w = np.random.rand(self.num_objectives)
            weights.append(w / np.sum(w))
        
        return np.array(weights[:self.N])

    def _calculate_neighborhoods(self) -> np.ndarray:
        """基于权重向量之间的欧氏距离计算每个个体的邻域"""
        distances = np.linalg.norm(self.weights[:, np.newaxis, :] - self.weights, axis=2)
        return np.argsort(distances, axis=1)[:, :self.T]

    def _initialize_population(self) -> List[Solution]:
        """ 使用 Initializer 和 智能分配 策略来初始化种群 """
        print("##########################################################")
        print("Initializing population with heuristic and advanced assignment strategy...")
        population = [None] * self.N
        
        # Heuristic Rule Mapping: h1 -> Cmax, h3 -> TE, h2 -> TEC
        idx_cmax = np.argmax(self.weights[:, 0])
        idx_te   = np.argmax(self.weights[:, 1])
        idx_tec  = np.argmax(self.weights[:, 2])
        
        specialist_indices = {idx_cmax, idx_te, idx_tec}
        print(f"Specialist subproblems indices: Cmax->{idx_cmax}, TE->{idx_te}, TEC->{idx_tec}")

        # Generate elite solutions and directly assign
        s_cmax = self.initializer.generate_with_heuristic1()
        s_te = self.initializer.generate_with_heuristic3()
        # s_tec = self.initializer.generate_with_heuristic2()
        s_tec = self.initializer.generate_with_heuristic3()
        
        population[idx_cmax] = s_cmax
        population[idx_te] = s_te
        population[idx_tec] = s_tec
        
        filler_pool = []
        for _ in range(self.initializer.params.get('h1_perturb_count', 0)):
            filler_pool.append(self.initializer._perturb_solution(s_cmax))
        for _ in range(self.initializer.params.get('h2_perturb_count', 0)):
             filler_pool.append(self.initializer._perturb_solution(s_tec))
        for _ in range(self.initializer.params.get('h3_perturb_count', 0)):
             filler_pool.append(self.initializer._perturb_solution(s_te))

        num_specialists = len(specialist_indices)
        num_fillers_needed = self.N - num_specialists
        num_randoms_needed = num_fillers_needed - len(filler_pool)

        for _ in range(max(0, num_randoms_needed)):
            filler_pool.append(self.initializer.generate_randomly())
            
        random.shuffle(filler_pool)

        filler_idx = 0
        for i in range(self.N):
            if population[i] is None:
                if filler_idx < len(filler_pool):
                    population[i] = filler_pool[filler_idx]
                    filler_idx += 1
                else:
                    population[i] = self.initializer.generate_randomly()

        # print(population)
        
        return population

    def _update_ideal_point_with_population(self):
        """用整个种群的目标值更新理想点Z"""
        objectives = np.array([sol.objectives for sol in self.population])
        self.z = np.min(np.vstack((self.z, objectives)), axis=0)

    def _update_ideal_point(self, new_objectives: np.ndarray):
        """用新解的目标值更新理想点Z"""
        self.z = np.min(np.vstack((self.z, new_objectives)), axis=0)

    def _update_nadir_point_with_population(self):
        """用当前整个种群的目标值来更新最差边界"""
        objectives = np.array([sol.objectives for sol in self.population])
        self.z_nad = np.max(objectives, axis=0)
        
    def _tchebycheff(self, objectives: np.ndarray, weight: np.ndarray) -> float:
        """计算包含动态归一化的 Tchebycheff 聚合函数值"""
        norm_den = self.z_nad - self.z
        norm_den[norm_den < 1e-6] = 1e-6
        
        normalized_objectives = (objectives - self.z) / norm_den
        return np.max(normalized_objectives * weight)

    def _generate_offspring(self, parent1: Solution, parent2: Solution) -> Solution:
        child = Solution(sequence=np.copy(parent1.sequence), 
                        mode=np.copy(parent1.mode), 
                        put_off=np.copy(parent1.put_off))

        if random.random() < self.cr:
            crossover_mode = 'block' if random.random() < self.block_cr_prob else 'normal'
            try:
                child1, _ = self.operation.sequence_OX(parent1, parent2, mode=crossover_mode)
                child.sequence = child1.sequence
            except Exception as e:
                child1, _ = self.operation.sequence_OX(parent1, parent2, mode='normal')
                child.sequence = child1.sequence

        if random.random() < self.mr:
            child = self.operation.sequence_MUT(child)

        mask = np.random.rand(*parent1.mode.shape) < 0.5
        child.mode = np.where(mask, parent1.mode, parent2.mode)

        if random.random() < self.mode_repair_prob:
            child = self.operation.mode_MUT(child, self.mode_repair_high_probability)
            
        mask_putoff = np.random.rand(*parent1.put_off.shape) < 0.5
        child.put_off = np.where(mask_putoff, parent1.put_off, parent2.put_off)
        
        if random.random() < self.ls_prob:
            ls_operator = random.choice(["LocalOpTEC"])
            
            child = self.decoder.decode(child, ls_operator)

        return child
    
    def run(self):
        for gen in tqdm(range(self.max_gen), desc="MOEAD Generation"):
            self._update_nadir_point_with_population()
            for i in range(self.N):
                # 1. 从邻域中选择父母
                p_indices = np.random.choice(self.neighborhoods[i], 2, replace=False)
                parent1 = self.population[p_indices[0]]
                parent2 = self.population[p_indices[1]]

                # 2. 生成子代并解码
                child = self._generate_offspring(parent1, parent2)
                child = self.decoder.decode(child)

                # child = self.decoder.decode(child, "LocalOpTEC")
                # child = self.decoder.decode(child, "RightOpTE")

                # 3. 更新理想点 Z
                self._update_ideal_point(child.objectives)

                # 4. 更新邻域内的个体
                for j in self.neighborhoods[i]:
                    current_obj_val = self._tchebycheff(self.population[j].objectives, self.weights[j])
                    new_obj_val = self._tchebycheff(child.objectives, self.weights[j])
                    
                    if new_obj_val < current_obj_val:
                        self.population[j] = child
            
        return self.population

if __name__ == "__main__":
    from data_manager import load_instance
    
    instance_name = "instance_5j_3m"
    problem = load_instance(f"MOCO\\data\\{instance_name}.npz")
    
    decoder = Decoder(problem)
    operation = Operation(problem)
    
    moead_params = {
        "population_size": 100,
        "neighborhood_size": 20,
        "max_generations": 100,
        "crossover_rate": 0.8,
        "mutation_rate": 0.1,
        "block_crossover_prob": 0.2,
        "mode_repair_prob": 0.8,
        "local_search_prob": 0.3,
        "mode_repair_high_probability": 0.9
    }

    init_params = {
        'h1_count': 1,           
        'h2_count': 1,           
        'h3_count': 1,           
        'h1_perturb_count': 5,  # Cmax
        'h2_perturb_count': 5,  # TEC
        'h3_perturb_count': 5,  # TE
    }
    
    initializer = Initializer(problem, moead_params, init_params)

    moead = MOEAD(problem, 
                  operation,
                  decoder,
                  initializer,
                  moead_params,
    )
    
    population = moead.run()
    
    print("MOEAD run finished.")
    
    save_results(population, moead_params, instance_name)
    
    plot_pareto_front(population, instance_name)
    
    print("##########################################################")