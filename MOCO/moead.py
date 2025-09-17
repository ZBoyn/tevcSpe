import numpy as np
import random
import math
from itertools import combinations
from pro_def import ProblemDefinition, Solution
from decode import Decoder
from util.plot_pf import plot_pareto_front
from util.results_manager import save_results

class MOEAD:
    def __init__(self,
                 problem: ProblemDefinition,
                 decoder: Decoder,
                 population_size: int,
                 neighborhood_size: int,
                 max_generations: int,
                 crossover_rate: float,
                 mutation_rate: float):

        self.problem = problem
        self.decoder = decoder
        self.N = population_size
        self.T = neighborhood_size
        self.max_gen = max_generations
        self.cr = crossover_rate
        self.mr = mutation_rate
        self.num_objectives = 3

        self.weights = self._initialize_weights()

        self.neighborhoods = self._calculate_neighborhoods()

        self.population = self._initialize_population()

        self.z = np.full(self.num_objectives, np.inf)
        self._update_ideal_point_with_population()

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

    def _initialize_population(self):
        """随机生成初始种群"""
        population = []
        for _ in range(self.N):
            sequence = np.random.permutation(self.problem.num_jobs)
            mode = np.random.randint(0, 2, size=(self.problem.num_jobs, self.problem.num_machines))
            put_off = np.zeros(shape=(self.problem.num_jobs, self.problem.num_machines), dtype=int)
            
            sol = Solution(sequence=sequence, mode=mode, put_off=put_off)
            population.append(self.decoder.decode(sol))
        return population

    def _update_ideal_point_with_population(self):
        """用整个种群的目标值更新理想点Z"""
        objectives = np.array([sol.objectives for sol in self.population])
        self.z = np.min(np.vstack((self.z, objectives)), axis=0)

    def _update_ideal_point(self, new_objectives: np.ndarray):
        """用新解的目标值更新理想点Z"""
        self.z = np.min(np.vstack((self.z, new_objectives)), axis=0)

    def _tchebycheff(self, objectives: np.ndarray, weight: np.ndarray) -> float:
        """计算 Tchebycheff 聚合函数值"""
        return np.max(np.abs(objectives - self.z) * weight)

    def _genetic_operators(self, parent1: Solution, parent2: Solution) -> Solution:
        """
        执行交叉和变异操作生成一个子代.
        
        注意：这里的实现是示例，你需要根据你的问题特性选择或设计更优的操作.
        """
        # --- 交叉 (Crossover) ---
        # 工件序列 (Sequence) 使用顺序交叉 (Order Crossover - OX1)
        start, end = sorted(random.sample(range(self.problem.num_jobs), 2))
        child_seq = np.full(self.problem.num_jobs, -1, dtype=int)
        child_seq[start:end+1] = parent1.sequence[start:end+1]
        
        p2_idx = 0
        for i in range(self.problem.num_jobs):
            if child_seq[i] == -1:
                while parent2.sequence[p2_idx] in child_seq:
                    p2_idx += 1
                child_seq[i] = parent2.sequence[p2_idx]
        
        # 模式 (Mode) 和延迟 (put_off) 使用均匀交叉
        child_mode = np.where(np.random.rand(self.problem.num_jobs, self.problem.num_machines) < 0.5, parent1.mode, parent2.mode)
        child_putoff = np.where(np.random.rand(self.problem.num_jobs, self.problem.num_machines) < 0.5, parent1.put_off, parent2.put_off)

        # --- 变异 (Mutation) ---
        if random.random() < self.mr:
            # 序列变异：交换两个位置
            i, j = random.sample(range(self.problem.num_jobs), 2)
            child_seq[i], child_seq[j] = child_seq[j], child_seq[i]

            # 模式变异：随机翻转一位
            job_idx, machine_idx = random.randint(0, self.problem.num_jobs-1), random.randint(0, self.problem.num_machines-1)
            child_mode[job_idx, machine_idx] = 1 - child_mode[job_idx, machine_idx]

            # put_off 变异：随机赋予一个新值 (例如0-3)
            job_idx, machine_idx = random.randint(0, self.problem.num_jobs-1), random.randint(0, self.problem.num_machines-1)
            child_putoff[job_idx, machine_idx] = random.randint(0, 3)
            
        return Solution(sequence=child_seq, mode=child_mode, put_off=child_putoff)
    
    def run(self):
        for gen in range(self.max_gen):
            for i in range(self.N):
                # 1. 从邻域中选择父母
                p_indices = np.random.choice(self.neighborhoods[i], 2, replace=False)
                parent1 = self.population[p_indices[0]]
                parent2 = self.population[p_indices[1]]

                # 2. 生成子代并解码
                child = self._genetic_operators(parent1, parent2)
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
            
            if (gen + 1) % 10 == 0:
                print(f"Generation {gen + 1}/{self.max_gen} completed.")
        
        return self.population

if __name__ == "__main__":
    from data_manager import load_instance
    
    instance_name = "instance_5j_3m"
    problem = load_instance(f"MOCO\\data\\{instance_name}.npz")
    
    decoder = Decoder(problem)
    
    params = {
        "population_size": 100,
        "neighborhood_size": 20,
        "max_generations": 100,
        "crossover_rate": 0.8,
        "mutation_rate": 0.1
    }
    
    moead = MOEAD(problem, 
                  decoder, 
                  params["population_size"], 
                  params["neighborhood_size"], 
                  params["max_generations"], 
                  params["crossover_rate"], 
                  params["mutation_rate"])
    
    population = moead.run()
    
    print("MOEAD run finished.")
    
    save_results(population, params, instance_name)
    
    plot_pareto_front(population, instance_name)
    
    print("##########################################################")