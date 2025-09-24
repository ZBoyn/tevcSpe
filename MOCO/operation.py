import numpy as np
from pro_def import ProblemDefinition, Solution
from decode import Decoder

class Operation:
    def __init__(self, problem_definition: ProblemDefinition):
        self.problem = problem_definition
        self.decoder = Decoder(problem_definition)

    def _find_critical_path(self, solution: Solution) -> np.ndarray:
        pass

    def _identify_block(self, solution: Solution) -> np.ndarray:
        pass