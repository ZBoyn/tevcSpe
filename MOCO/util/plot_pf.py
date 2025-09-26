import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import List

def plot_pareto_front(population: List, problem_name: str = "Pareto Front"):
    objectives = np.array([sol.objectives for sol in population])
    
    # 找到非支配解
    num_solutions = objectives.shape[0]
    is_dominated = np.zeros(num_solutions, dtype=bool)
    for i in range(num_solutions):
        for j in range(num_solutions):
            if i == j:
                continue
            # 检查解 j 是否支配解 i (最小化问题)
            if np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                is_dominated[i] = True
                break
    
    pareto_front_objectives = objectives[~is_dominated]

    # 绘制三维帕累托前沿
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(pareto_front_objectives[:, 0], pareto_front_objectives[:, 1], pareto_front_objectives[:, 2])
    ax1.set_xlabel('Cmax')
    ax1.set_ylabel('TEC')
    ax1.set_zlabel('TE')
    ax1.set_title(f'3D Pareto Front for {problem_name}')

    # 绘制两两组合的帕累托前沿
    fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Cmax vs TEC
    axes[0].scatter(pareto_front_objectives[:, 0], pareto_front_objectives[:, 1])
    axes[0].set_xlabel('Cmax')
    axes[0].set_ylabel('TEC')
    axes[0].set_title('Cmax vs TEC')
    axes[0].grid(True)
    
    # Cmax vs TE
    axes[1].scatter(pareto_front_objectives[:, 0], pareto_front_objectives[:, 2])
    axes[1].set_xlabel('Cmax')
    axes[1].set_ylabel('TE')
    axes[1].set_title('Cmax vs TE')
    axes[1].grid(True)
    
    # TEC vs TE
    axes[2].scatter(pareto_front_objectives[:, 1], pareto_front_objectives[:, 2])
    axes[2].set_xlabel('TEC')
    axes[2].set_ylabel('TE')
    axes[2].set_title('TEC vs TE')
    axes[2].grid(True)

    fig2.suptitle(f'2D Pareto Fronts for {problem_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_min_objectives(gen, min_Cmax, min_TE, min_TEC):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = range(1, gen+1, 10)
    color1 = 'tab:blue'
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Cmax', color=color1)
    ax1.plot(x, min_Cmax, color=color1, marker='o', label='min_Cmax')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    color3 = 'tab:green'
    ax2.set_ylabel('TE / TEC', color='black')
    ln2 = ax2.plot(x, min_TE, color=color2, marker='s', label='min_TE')
    ln3 = ax2.plot(x, min_TEC, color=color3, marker='^', label='min_TEC')
    ax2.tick_params(axis='y', labelcolor='black')

    lns = [ax1.lines[0]] + ln2 + ln3
    labels = [l.get_label() for l in lns]
    ax1.legend(lns, labels, loc='upper right')

    plt.title("Min Objectives over Generations")
    plt.show()