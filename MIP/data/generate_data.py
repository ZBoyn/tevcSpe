import numpy as np
import random
def generate_instance(n, m, seed=42):
    random.seed(seed)

    # 1. Number of periods: U[3, 5]
    num_periods = 5

    speedFactor = random.uniform(0.8, 1)
    energyFactor = random.uniform(1, 1.5)
    e_idle = 1
    
    # 2.Period Duration: U[20, 80] 
    s = [random.randint(10, 35) for _ in range(num_periods)]
    
    # 3. Period Start
    # u_start = np.zeros(num_periods, dtype=int)
    # if num_periods > 1:
    #     u_start[1:] = np.cumsum(s[:-1])
    # u = u_start.tolist()
    u = [0, 15, 40, 50, 70]

    # 4. Period Energy Cost: U[1, 7]
    f = [5, 7, 12, 6, 5]

    # 5. Release Times: U[0, 5n]
    r = [None] + [random.randint(0, 5 * n) for _ in range(n)]
    # r = [None] + [random.randint(0, 1) for _ in range(n)]

    # 6. Base Processing Time: U[1, 10]
    p = [[None] * m]
    for _ in range(n):
        p.append([random.randint(1, 10) for _ in range(n)])
        
    e = [[None] * m]
    for _ in range(n):
        e.append([random.randint(1, 8) for _ in range(m)])
        
    return p, e, r, u, s, f, num_periods, speedFactor, energyFactor, e_idle