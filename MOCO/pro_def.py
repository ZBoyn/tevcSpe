import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class ProblemDefinition:
    processing_times: np.ndarray      # P[N, M]: Job x Machine
    power_consumption: np.ndarray     # E[N, M]: Job x Machine
    release_times: np.ndarray         # R[N]: Job release time
    period_start_times: np.ndarray    # U[K+1]: Start times of price periods
    period_prices: np.ndarray         # W[K]: Price for each period

    HIGH_MODE_SPEED_FACTOR: float = 0.7
    HIGH_MODE_POWER_FACTOR: float = 1.5
    LOW_MODE_SPEED_FACTOR: float = 1.0
    LOW_MODE_POWER_FACTOR: float = 1.0

    IDLE_MODE_POWER: float = 1.0

    num_jobs: int = field(init=False)
    num_machines: int = field(init=False)
    num_periods: int = field(init=False)
    
    def __post_init__(self):
        self.num_jobs, self.num_machines = self.processing_times.shape
        self.num_periods = len(self.period_prices)
        if len(self.period_start_times) != self.num_periods + 1:
            raise ValueError("period_start_times length must be num_periods + 1")
        
        self.speed_factors = {0: self.LOW_MODE_SPEED_FACTOR, 1: self.HIGH_MODE_SPEED_FACTOR}
        self.power_factors = {0: self.LOW_MODE_POWER_FACTOR, 1: self.HIGH_MODE_POWER_FACTOR}

@dataclass
class Solution:
    sequence: np.ndarray  # Shape: (N,) - Order of jobs
    mode: np.ndarray      # Shape: (N, M) - Processing mode for each op
    put_off: np.ndarray   # Shape: (N, M) - Delay decision for each op

    start_times: np.ndarray = field(init=False, default=None)
    completion_times: np.ndarray = field(init=False, default=None) # C[N, M]
    objectives: np.ndarray = field(init=False, default_factory=lambda: np.full(3, np.inf)) # [Cmax, TEC, TE]
    prev: np.ndarray = field(init=False, default=None) # [N, M] 记录前驱         # S源点 H纵向 B横向 T时间
    # block: np.ndarray = field(init=False, default=None) # [N, M] 记录哪些工件是一个块
    
    def __post_init__(self):
        self.start_times = np.zeros_like(self.mode, dtype=float)
        self.completion_times = np.zeros_like(self.mode, dtype=float)
        self.prev = np.zeros_like(self.mode, dtype=object)

    def copy(self):
        new_solution = Solution(
            sequence=self.sequence.copy(),
            mode=self.mode.copy(),
            put_off=self.put_off.copy()
        )
        if self.start_times is not None:
            new_solution.start_times = self.start_times.copy()
        if self.completion_times is not None:
            new_solution.completion_times = self.completion_times.copy()
        if self.prev is not None:
            new_solution.prev = self.prev.copy()
        if self.objectives is not None:
            new_solution.objectives = self.objectives.copy()
        return new_solution