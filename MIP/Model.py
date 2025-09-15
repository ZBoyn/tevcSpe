from gurobipy import Model, GRB, quicksum
from data.generate_data import generate_instance
import random
import time
from data.config import n_jobs as n, m_machines as m, k_intervals as k, base_processing_time as p, base_energy as e, release_times as r, u, s, f, speedFactor, energyFactor, e_idle


for i in range(1):
    time.sleep(1)
    
    # n, m = 5, 3
    # p, e, r, u, s, f, k, speedFactor, energyFactor, e_idle = generate_instance(n, m, seed = i)
    # current_time = time.strftime("%Y%m%d_%H%M%S")
    # with open(f"data/data_{i}.txt_{current_time}", "w") as fp:
    #     fp.write(f"p: {p}\n")
    #     fp.write(f"e: {e}\n")
    #     fp.write(f"r: {r}\n")
    #     fp.write(f"u: {u}\n")
    #     fp.write(f"s: {s}\n")
    #     fp.write(f"f: {f}\n")
    #     fp.write(f"k: {k}\n")
    #     fp.write(f"speedFactor: {speedFactor}\n")
    #     fp.write(f"B: {energyFactor}\n")
    #     fp.write(f"e_idle: {e_idle}\n")



    """
    print("p:", p)
    print("e:", e)
    print("r:", r)
    print("u:", u)
    print("s:", s)
    print("f:", f)
    print("k:", k)
    """

    M = 100000

    model_total = Model("ToU_PFFSP")
    model_total.setParam("Seed", 42)
    random.seed(42)
    model_total.setParam("TimeLimit", 1800)

    # 决策变量
    # Y[j,i]：机器功率模式（0为低功率，1为高功率）， j=0,...,n-1, i=0,...,m-1
    Y = model_total.addVars(n, m, vtype=GRB.BINARY, name="Y")
    # S[j,i] 与 C[j,i]：操作 (j,i) 的开始与完成时间
    S = model_total.addVars(n, m, name="S")
    C = model_total.addVars(n, m, name="C")

    # 变量 A[j,i,t]：操作 (j,i) 被分配到时段 t，保证整个操作时间落在所选时段内
    A = model_total.addVars(n, m, k, vtype=GRB.BINARY, name="A")
    X = model_total.addVars(n, n, vtype=GRB.BINARY, name="X")
    for j in range(n):
        X[j, j] = 0

    # 新增变量 Z[j,i,t] = A[j,i,t] * Y[j,i]，用于线性化
    Z = model_total.addVars(n, m, k, vtype=GRB.BINARY, name="Z")

    # 定义 Cmax
    Cmax = model_total.addVar(name="Cmax")

    # 定义总能耗 TE（目标变量）
    TE_var = model_total.addVar(name="TE")

    # 总电费
    TEC_var = model_total.addVar(name="TEC")

    # 每台机器的最早开始和最晚结束时间
    S_machine = model_total.addVars(m, name="S_machine")
    C_machine = model_total.addVars(m, name="C_machine")

    def setmodel(obj):
        if obj == "Cmax":
            model_total.setObjective(Cmax, GRB.MINIMIZE)
        elif obj == "TE":
            model_total.setObjective(TE_var, GRB.MINIMIZE)
        elif obj == "TEC":
            model_total.setObjective(TEC_var, GRB.MINIMIZE)

    def save(obj):
        current_time = time.strftime("%Y%m%d_%H%M%S")
        with open(f"{obj}_{n}_{m}_{k}_solution.txt_{current_time}", "w") as f:
            f.write(f"Runtime = {model_total.Runtime:.2f} seconds\n")
            f.write(f"Objective = {obj}\n")
            f.write(f"Num_Jobs = {n}\n")
            f.write(f"Num_Machines = {m}\n")
            f.write(f"Num_Periods = {k}\n")
            f.write(f"TE = {TE_var.X}\n")
            f.write(f"Cmax = {Cmax.X}\n")
            f.write(f"TEC = {TEC_var.X}\n")
            for j in range(n):
                for i in range(m):
                    period = [t for t in range(k) if A[j,i,t].X > 0.5][0]
                    mode = "High" if Y[j,i].X > 0.5 else "Low"
                    f.write(f"Job{j+1}, Machine{i}, S={S[j,i].X:.2f}, C={C[j,i].X:.2f}, Period={period}, Mode={mode}\n")

    #########################################################
    ################### 约束定义 #############################

    # 约束1：每个操作 (j,i) 必须恰好分配到一个电价时段上
    model_total.addConstrs(
        (quicksum(A[j,i,t] for t in range(k)) == 1
        for j in range(n) for i in range(m)),
        name="TimePeriodAssign"
    )

    # 约束2：利用大M保证操作时间 S 与 C 落在所选时段内
    for j in range(n):
        for i in range(m):
            for t in range(k):
                model_total.addConstr(S[j,i] >= u[t] - M*(1 - A[j,i,t]), name=f"StartInPeriod_{j}_{i}_{t}")
                model_total.addConstr(C[j,i] <= u[t] + s[t] + M*(1 - A[j,i,t]), name=f"FinishInPeriod_{j}_{i}_{t}")

    # 约束3：定义加工时间及操作完成关系
    for j in range(n):
        for i in range(m):
            model_total.addConstr(C[j,i] == S[j,i] + p[j+1][i]*(1 - (1 - speedFactor) * Y[j,i]), name=f"OpDuration_{j}_{i}")

    # 约束4：流水作业约束
    # 对第一台机器，操作开始时间须不早于释放时间
    for j in range(n):
        model_total.addConstr(S[j,0] >= r[j+1], name=f"Release_j{j}")
    # 后续机器：操作 i+1 不早于前一道操作的完成
    for j in range(n):
        for i in range(m-1):
            model_total.addConstr(S[j,i+1] >= C[j,i], name=f"Flow_j{j}_mach{i}")

    # 约束5：同一机器上工件间不重叠（对全局顺序变量 X 仅在机器0给出约束）
    for j in range(n):
        for jp in range(j+1, n):
            model_total.addConstr(S[jp,0] >= C[j,0] - M*(1 - X[j,jp]), name=f"Seq_{j}_{jp}_1")
            model_total.addConstr(S[j,0] >= C[jp,0] - M*X[j,jp], name=f"Seq_{j}_{jp}_2")

    # 同样，可将排序约束延伸到其它机器
    for i in range(m):
        for j in range(n):
            for jp in range(j+1, n):
                model_total.addConstr(S[jp,i] >= C[j,i] - M*(1 - X[j,jp]), name=f"Seq_m{i}_{j}_{jp}")
                model_total.addConstr(S[j,i] >= C[jp,i] - M*X[j,jp], name=f"Seq_m{i}_{j}_{jp}_rev")

    # 约束6：定义 Cmax 为最后一台机器上所有操作的最大完工时间
    for j in range(n):
        model_total.addConstr(Cmax >= C[j, m-1], name=f"Cmax_ge_C_{j}")

    for i in range(m):
        model_total.addGenConstrMin(S_machine[i], [S[j,i] for j in range(n)], name=f"Min_S_m{i}")
        model_total.addGenConstrMax(C_machine[i], [C[j,i] for j in range(n)], name=f"Max_C_m{i}")

    #########################################################
    ################### 能耗计算 #############################
    # 1. 加工能耗
    processing_energy = quicksum(
        p[j+1][i]*e[j+1][i] * (1 + (speedFactor * energyFactor - 1) * Y[j,i])
        for j in range(n) for i in range(m)
    )

    # 2. 空闲能耗
    # 2.1. 每台机器的总加工时间
    total_processing_time_per_machine = {
        i: quicksum(p[j+1][i] * (1 - speedFactor * Y[j,i]) for j in range(n))
        for i in range(m)
    }
    
    # 2.2. 每台机器的总空闲时间 = (最晚完工 - 最早开工) - 总加工时间
    total_idle_time_per_machine = {
        i: C_machine[i] - S_machine[i] - total_processing_time_per_machine[i]
        for i in range(m)
    }

    # 2.3. 总空闲能耗 = sum(每台机器的空闲时间 * 空闲功率)
    total_idle_energy = quicksum(
        total_idle_time_per_machine[i] * e_idle for i in range(m)
    )

    model_total.addConstr(TE_var == processing_energy + total_idle_energy, name="TotalEnergy")

    # const_TE = quicksum(p[j+1][i]*e[j+1][i] for j in range(n) for i in range(m))
    # lin_TE   = quicksum((speedFactor * energyFactor - 1) * p[j+1][i]*e[j+1][i] * Y[j,i] for j in range(n) for i in range(m))

    # model_total.addConstr(TE_var == const_TE + lin_TE, name="TotalEnergy")

    # ---------------------- 线性化部分 -------------------------
    # 对每个操作 (j,i) 及时段 t, 定义 Z[j,i,t] = A[j,i,t] * Y[j,i]
    for j in range(n):
        for i in range(m):
            for t in range(k):
                model_total.addConstr(Z[j,i,t] <= A[j,i,t], name=f"Z_ub1_{j}_{i}_{t}")
                model_total.addConstr(Z[j,i,t] <= Y[j,i], name=f"Z_ub2_{j}_{i}_{t}")
                model_total.addConstr(Z[j,i,t] >= A[j,i,t] + Y[j,i] - 1, name=f"Z_lb_{j}_{i}_{t}")

                model_total.addGenConstrAnd(Z[j, i, t], [A[j, i, t], Y[j, i]])

    # -------------------- 总电费 TEC 的计算 --------------------
    # TEC_expr = quicksum( A[j,i,t]*p[j+1][i]*e[j+1][i]*f[t] +
    #                     (speedFactor * energyFactor - 1) * p[j+1][i]*e[j+1][i]*f[t] * Z[j,i,t]
    #                     for j in range(n) for i in range(m) for t in range(k))
    
    # --- 1. 计算加工电费 (与之前逻辑相同) ---
    processing_TEC = quicksum(
        f[t] * p[j+1][i] * e[j+1][i] * (A[j,i,t] + (speedFactor * energyFactor - 1) * Z[j,i,t])
        for j in range(n) for i in range(m) for t in range(k)
    )

    # 空闲时间电费
    OverlapStart = model_total.addVars(m, k, name="OverlapStart")
    OverlapEnd = model_total.addVars(m, k, name="OverlapEnd")
    OverlapDuration = model_total.addVars(m, k, name="OverlapDuration")
    IdleTimeInPeriod = model_total.addVars(m, k, name="IdleTimeInPeriod")

    for i in range(m):
        for t in range(k):
            # 约束：定义重叠区间的开始和结束
            model_total.addConstr(OverlapStart[i,t] >= S_machine[i], name=f"OS_ge_Sm_{i}_{t}")
            model_total.addConstr(OverlapStart[i,t] >= u[t], name=f"OS_ge_ut_{i}_{t}")
            model_total.addConstr(OverlapEnd[i,t] <= C_machine[i], name=f"OE_le_Cm_{i}_{t}")
            model_total.addConstr(OverlapEnd[i,t] <= u[t] + s[t], name=f"OE_le_utst_{i}_{t}")

            # 约束：计算重叠时长
            model_total.addConstr(OverlapDuration[i,t] >= OverlapEnd[i,t] - OverlapStart[i,t], name=f"OD_calc_{i}_{t}")

            # 约束：计算时段 t 内，机器 i 的总加工时间
            processing_time_in_period = quicksum(
                p[j+1][i] * (1 - speedFactor * Y[j,i]) * A[j,i,t] 
                for j in range(n)
            )

            # 约束：计算时段 t 内，机器 i 的净空闲时间
            model_total.addConstr(IdleTimeInPeriod[i,t] == OverlapDuration[i,t] - processing_time_in_period, name=f"IdleTime_calc_{i}_{t}")

    idle_TEC = quicksum(
        IdleTimeInPeriod[i,t] * e_idle * f[t] 
        for i in range(m) for t in range(k)
    )

    model_total.addConstr(TEC_var == processing_TEC + idle_TEC, name="TotalCost")
    # model_total.addConstr(TEC_var == TEC_expr, name="TotalCost")

    # TEC_expr = quicksum(
    #     A[j,i,t] * p[j+1][i] * e[j+1][i] * f[t] * (1 + 0.05*Y[j,i])
    #     for j in range(n) for i in range(m) for t in range(k))
    # model_total.addConstr(TEC_var == TEC_expr, name="TotalCost")
    # TEC_expr0 = quicksum( A[j,2,t]*p[j+1][2]*e[j+1][2]*f[t] +
    #                     0.05 * p[j+1][2]*e[j+1][2]*f[t] * Z[j,2,t]
    #                     for j in range(n) for t in range(k))
    # ------------------- 求解模型 -------------------
    # model_total.optimize()

    for obj in ["Cmax", "TE", "TEC"]:
        setmodel(obj)
        model_total.update()
        model_total.optimize()
        if model_total.status == GRB.OPTIMAL:
            print(f"Runtime: {model_total.Runtime:.2f} seconds")
            print("最优总能耗 TE =", TE_var.X)
            print("最优 Cmax =", Cmax.X)
            print("最小总电费 TEC =", TEC_var.X)
            # print("test_var =", test_var.X)
            for j in range(n):
                for i in range(m):
                    # 查找对应操作所分配的时段
                    period = [t for t in range(k) if A[j,i,t].X > 0.5][0]
                    mode = "高档" if Y[j,i].X > 0.5 else "低档"
                    print(f"工件{j+1}, 机器{i}, S={S[j,i].X:.2f}, C={C[j,i].X:.2f}, 时段={period}, 模式={mode}")
            # 输出排序变量值（仅打印部分信息）
            print("全局加工顺序：")
            order = sorted(range(n), key=lambda j: S[j, 0].X)
            print(" -> ".join(f"工件{j + 1}" for j in order))
            
            # 保存结果到文件
            save(obj)
        else:
            print("模型未找到最优解")