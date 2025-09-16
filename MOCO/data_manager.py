import numpy as np
from pro_def import ProblemDefinition

def create_and_save_instance(filename: str):
    HIGH_MODE_SPEED_FACTOR = 0.7
    HIGH_MODE_POWER_FACTOR = 1.5
    IDLE_MODE_POWER = 1.0

    period_start_times = np.array([0, 15, 45, 65])
    period_prices = np.array([2, 8, 3])

    release_times = np.array([0, 8, 18, 16, 20])

    processing_times_T = np.array([
        [4, 6, 5, 4, 5], [6, 4, 4, 5, 4], [5, 4, 3, 4, 4]
    ])
    processing_times = processing_times_T.T

    power_consumption_T = np.array([
        [7, 3, 4, 2, 7], [4, 3, 3, 6, 2], [4, 3, 6, 3, 6]
    ])
    power_consumption = power_consumption_T.T
    
    np.savez_compressed(
        filename,
        processing_times=processing_times,
        power_consumption=power_consumption,
        release_times=release_times,
        period_start_times=period_start_times,
        period_prices=period_prices,
        HIGH_MODE_SPEED_FACTOR=HIGH_MODE_SPEED_FACTOR,
        HIGH_MODE_POWER_FACTOR=HIGH_MODE_POWER_FACTOR,
        IDLE_MODE_POWER=IDLE_MODE_POWER
    )
    print("######################################")

def load_instance(filename: str) -> ProblemDefinition:
    data = np.load(filename)

    problem = ProblemDefinition(
        processing_times=data['processing_times'],
        power_consumption=data['power_consumption'],
        release_times=data['release_times'],
        period_start_times=data['period_start_times'],
        period_prices=data['period_prices'],
        HIGH_MODE_SPEED_FACTOR=data['HIGH_MODE_SPEED_FACTOR'].item(),
        HIGH_MODE_POWER_FACTOR=data['HIGH_MODE_POWER_FACTOR'].item(),
        IDLE_MODE_POWER=data['IDLE_MODE_POWER'].item()
    )
    return problem

if __name__ == '__main__':
    instance_filename = "instance_5j_3m.npz"
    create_and_save_instance(instance_filename)
    problem_from_file = load_instance(instance_filename)
    print(f"加载的工件数: {problem_from_file.num_jobs}")
    print(f"加载的机器数: {problem_from_file.num_machines}")
    print("处理时间矩阵 (形状):", problem_from_file.processing_times.shape)