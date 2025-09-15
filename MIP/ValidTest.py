from gurobipy import Model, GRB, quicksum
from data.generate_data import generate_instance
import random
import time
import json

def create_model_with_inequalities(inequality_config, n, m):
    A = 0.7
    B = 1.5
    
    p, e, r, u, s, f, num_periods = generate_instance(n, m)
    k = num_periods
    
    M = 100000
    
    model = Model("Scheduling_Min_TE")
    model.setParam("Seed", 42)
    random.seed(42)
    model.setParam("TimeLimit", 1800)
    
    # 决策变量
    Y = model.addVars(n, m, vtype=GRB.BINARY, name="Y")
    S = model.addVars(n, m, name="S")
    C = model.addVars(n, m, name="C")
    A = model.addVars(n, m, k, vtype=GRB.BINARY, name="A")
    X = model.addVars(n, n, vtype=GRB.BINARY, name="X")
    for j in range(n):
        X[j, j] = 0
    Z = model.addVars(n, m, k, vtype=GRB.BINARY, name="Z")
    Cmax = model.addVar(name="Cmax")
    TE_var = model.addVar(name="TE")
    TEC_var = model.addVar(name="TEC")
    
    # 目标函数
    model.setObjective(TEC_var, GRB.MINIMIZE)
    
    # 基础约束
    # 约束1：每个操作必须恰好分配到一个电价时段上
    model.addConstrs(
        (quicksum(A[j,i,t] for t in range(k)) == 1
         for j in range(n) for i in range(m)),
        name="TimePeriodAssign"
    )
    
    # 约束2：利用大M保证操作时间落在所选时段内
    for j in range(n):
        for i in range(m):
            for t in range(k):
                model.addConstr(S[j,i] >= u[t] - M*(1 - A[j,i,t]), name=f"StartInPeriod_{j}_{i}_{t}")
                model.addConstr(C[j,i] <= u[t] + s[t] + M*(1 - A[j,i,t]), name=f"FinishInPeriod_{j}_{i}_{t}")
    
    # 约束3：定义加工时间及操作完成关系
    for j in range(n):
        for i in range(m):
            model.addConstr(C[j,i] == S[j,i] + p[j+1][i]*(1 - 0.3*Y[j,i]), name=f"OpDuration_{j}_{i}")
    
    # 约束4：流水作业约束
    for j in range(n):
        model.addConstr(S[j,0] >= r[j+1], name=f"Release_j{j}")
    for j in range(n):
        for i in range(m-1):
            model.addConstr(S[j,i+1] >= C[j,i], name=f"Flow_j{j}_mach{i}")
    
    # 约束5：同一机器上工件间不重叠
    for j in range(n):
        for jp in range(j+1, n):
            model.addConstr(S[jp,0] >= C[j,0] - M*(1 - X[j,jp]), name=f"Seq_{j}_{jp}_1")
            model.addConstr(S[j,0] >= C[jp,0] - M*X[j,jp], name=f"Seq_{j}_{jp}_2")
    
    for i in range(m):
        for j in range(n):
            for jp in range(j+1, n):
                model.addConstr(S[jp,i] >= C[j,i] - M*(1 - X[j,jp]), name=f"Seq_m{i}_{j}_{jp}")
                model.addConstr(S[j,i] >= C[jp,i] - M*X[j,jp], name=f"Seq_m{i}_{j}_{jp}_rev")
    
    # 约束6：定义 Cmax
    for j in range(n):
        model.addConstr(Cmax >= C[j, m-1], name=f"Cmax_ge_C_{j}")
    
    # 能耗计算
    const_TE = quicksum(p[j+1][i]*e[j+1][i] for j in range(n) for i in range(m))
    lin_TE = quicksum(0.05 * p[j+1][i]*e[j+1][i] * Y[j,i] for j in range(n) for i in range(m))
    model.addConstr(TE_var == const_TE + lin_TE, name="TotalEnergy")
    
    # 线性化部分
    for j in range(n):
        for i in range(m):
            for t in range(k):
                model.addConstr(Z[j,i,t] <= A[j,i,t], name=f"Z_ub1_{j}_{i}_{t}")
                model.addConstr(Z[j,i,t] <= Y[j,i], name=f"Z_ub2_{j}_{i}_{t}")
                model.addConstr(Z[j,i,t] >= A[j,i,t] + Y[j,i] - 1, name=f"Z_lb_{j}_{i}_{t}")
                model.addGenConstrAnd(Z[j, i, t], [A[j, i, t], Y[j, i]])
    
    # 总电费计算
    TEC_expr = quicksum(
        A[j,i,t] * p[j+1][i] * e[j+1][i] * f[t] * (1 + 0.05*Y[j,i])
        for j in range(n) for i in range(m) for t in range(k))
    model.addConstr(TEC_var == TEC_expr, name="TotalCost")
    
    # 添加有效不等式
    if inequality_config.get('ineq1', False):
        # 有效不等式1
        model.addConstrs(
            (A[j,i,t] <= quicksum(A[j,i,tp] for tp in range(t,k))
             for j in range(n) for i in range(m) for t in range(k)),
            name="ValidInequalityI"
        )
    
    if inequality_config.get('ineq2', False):
        # 有效不等式2
        for i in range(m):
            for t in range(k):
                model.addConstr(
                    quicksum(p[j+1][i] * A[j,i,t] - 0.3*p[j+1][i] * Z[j,i,t]
                             for j in range(n)) <= s[t],
                    name=f"Cap_m{i}_t{t}"
                )
    
    """ 
    if inequality_config.get('ineq3', False):
        # 有效不等式3
        for j in range(n):
            sum_p_ji = sum(p[j+1][i] for i in range(m))
            variable_duration_part = quicksum(0.3 * p[j+1][i] * Y[j,i] for i in range(m))
            model.addConstr(
                Cmax >= r[j+1] + sum_p_ji - variable_duration_part,
                name=f"CmaxLowerBound_Job{j}"
            )
    
    if inequality_config.get('ineq4', False):
        # 有效不等式4
        alpha = 0.7
        for j in range(n):
            for i in range(1, m):
                earliest_start_lb = r[j+1] + sum(p[j+1][k] * alpha for k in range(i))
                for t in range(k):
                    period_end_time = u[t] + s[t]
                    if period_end_time < earliest_start_lb:
                        model.addConstr(A[j,i,t] == 0, name=f"ForbiddenSlot_{j}_{i}_{t}")
    
    if inequality_config.get('ineq5', False):
        # 有效不等式5
        e_bound = min(min(row[1:]) for row in e[1:])
        price_periods = sorted([(f[t], s[t]) for t in range(k)])
        
        cumulative_machine_hours = 0.0
        cumulative_cost = 0.0
        tradeoff_points_corrected = []
        
        for price, duration in price_periods:
            machine_hours_in_period = duration * m
            cumulative_machine_hours += machine_hours_in_period
            cost_in_period = machine_hours_in_period * e_bound * price
            cumulative_cost += cost_in_period
            tradeoff_points_corrected.append({'H': cumulative_machine_hours, 'C': cumulative_cost})
        
        D_total_expr = quicksum(
            p[j+1][i] * (1 - 0.3 * Y[j,i]) 
            for j in range(n) for i in range(m)
        )
        
        for i in range(len(tradeoff_points_corrected) - 1):
            Hk = tradeoff_points_corrected[i]['H']
            Ck = tradeoff_points_corrected[i]['C']
            next_price = price_periods[i+1][0]
            slope = e_bound * next_price
            model.addConstr(
                TEC_var >= Ck + slope * (D_total_expr - Hk),
                name=f"CorrectedGlobalCut_{i}"
            )
    
    if inequality_config.get('ineq6', False):
        # 有效不等式6
        f_min = {}
        for j in range(n):
            cumulative_fastest_time = 0
            for i in range(m):
                s_ji_lb = r[j+1] + cumulative_fastest_time
                feasible_periods = [t for t in range(k) if u[t] + s[t] > s_ji_lb]
                
                if feasible_periods:
                    f_min[(j, i)] = min(f[t] for t in feasible_periods)
                else:
                    f_min[(j, i)] = 1e9
                
                cumulative_fastest_time += p[j+1][i] * 0.7
        
        tec_lower_bound_expr = quicksum(
            f_min[(j, i)] * (p[j+1][i] * e[j+1][i] + 0.05 * p[j+1][i] * e[j+1][i] * Y[j, i])
            for j in range(n) for i in range(m)
        )
        model.addConstr(TEC_var >= tec_lower_bound_expr, name="TECLowerBound")
     """
    return model, TE_var, Cmax, TEC_var

def run_experiment_for_size(n, m):
    """运行特定规模问题的所有有效不等式组合实验"""
    
    # 定义8种配置
    configurations = [
        {"name": "无有效不等式", "config": {}},
        {"name": "仅有效不等式1", "config": {"ineq1": True}},
        {"name": "仅有效不等式2", "config": {"ineq2": True}},
        {"name": "仅有效不等式3", "config": {"ineq3": True}},
        {"name": "仅有效不等式4", "config": {"ineq4": True}},
        {"name": "仅有效不等式5", "config": {"ineq5": True}},
        {"name": "仅有效不等式6", "config": {"ineq6": True}},
        {"name": "全部有效不等式", "config": {"ineq1": True, "ineq2": True, "ineq3": True, "ineq4": True, "ineq5": True, "ineq6": True}},
        {"name": "有效不等式1+2+5", "config": {"ineq1": True, "ineq2": True, "ineq5": True}},
    ]
    
    results = []
    
    print(f"\n开始测试问题规模: n={n}, m={m}")
    
    for config in configurations:
        print(f"正在测试: {config['name']}")
        
        try:
            start_time = time.time()
            model, TE_var, Cmax, TEC_var = create_model_with_inequalities(config['config'], n, m)
            model.optimize()
            end_time = time.time()
            
            runtime = end_time - start_time
            
            # 检查是否有可行解
            if model.status == GRB.OPTIMAL:
                result = {
                    "配置": config['name'],
                    "运行时间(秒)": round(runtime, 2),
                    "TE": round(TE_var.X, 2),
                    "Cmax": round(Cmax.X, 2),
                    "TEC": round(TEC_var.X, 2),
                    "状态": "最优解",
                    "有效不等式": config['config']
                }
            elif model.status == GRB.TIME_LIMIT and model.solCount > 0:
                # 时间限制但找到了可行解
                result = {
                    "配置": config['name'],
                    "运行时间(秒)": round(runtime, 2),
                    "TE": round(TE_var.X, 2),
                    "Cmax": round(Cmax.X, 2),
                    "TEC": round(TEC_var.X, 2),
                    "状态": f"时间限制 (状态: {model.status})",
                    "有效不等式": config['config']
                }
            elif model.solCount > 0:
                # 其他情况但有可行解
                result = {
                    "配置": config['name'],
                    "运行时间(秒)": round(runtime, 2),
                    "TE": round(TE_var.X, 2),
                    "Cmax": round(Cmax.X, 2),
                    "TEC": round(TEC_var.X, 2),
                    "状态": f"可行解 (状态: {model.status})",
                    "有效不等式": config['config']
                }
            else:
                # 没有找到任何可行解
                result = {
                    "配置": config['name'],
                    "运行时间(秒)": round(runtime, 2),
                    "TE": "N/A",
                    "Cmax": "N/A", 
                    "TEC": "N/A",
                    "状态": f"无可行解 (状态: {model.status})",
                    "有效不等式": config['config']
                }
            
            results.append(result)
            print(f"完成: {config['name']}, 运行时间: {runtime:.2f}秒")
            
        except Exception as e:
            result = {
                "配置": config['name'],
                "运行时间(秒)": "N/A",
                "TE": "N/A",
                "Cmax": "N/A",
                "TEC": "N/A", 
                "状态": f"错误: {str(e)}",
                "有效不等式": config['config']
            }
            results.append(result)
            print(f"错误: {config['name']} - {str(e)}")
    
    # 保存结果到特定文件
    current_time = time.strftime("%Y%m%d_%H%M%S")
    filename = f"有效不等式测试结果_n{n}_m{m}_{current_time}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("有效不等式组合测试结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"问题规模: n={n}, m={m}\n")
        f.write("=" * 50 + "\n\n")
        
        # 表格形式输出
        f.write(f"{'配置':<20} {'运行时间(秒)':<12} {'TE':<10} {'Cmax':<10} {'TEC':<10} {'状态':<15}\n")
        f.write("-" * 80 + "\n")
        
        for result in results:
            f.write(f"{result['配置']:<20} {str(result['运行时间(秒)']):<12} {str(result['TE']):<10} {str(result['Cmax']):<10} {str(result['TEC']):<10} {result['状态']:<15}\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("详细结果 (JSON格式):\n")
        f.write(json.dumps(results, ensure_ascii=False, indent=2))
    
    print(f"结果已保存到: {filename}")
    
    # 打印汇总
    print(f"\n问题规模 n={n}, m={m} 的测试结果汇总:")
    print(f"{'配置':<20} {'运行时间(秒)':<12} {'TE':<10} {'Cmax':<10} {'TEC':<10}")
    print("-" * 70)
    for result in results:
        print(f"{result['配置']:<20} {str(result['运行时间(秒)']):<12} {str(result['TE']):<10} {str(result['Cmax']):<10} {str(result['TEC']):<10}")
    
    return results, filename

def run_all_experiments():
    """运行所有问题规模的实验"""
    
    # 定义要测试的问题规模
    problem_sizes = [
        (7, 5),
        (10, 5),
        (12, 5),
        (15, 5),
    ]
    
    all_results = {}
    
    print("开始多规模有效不等式测试")
    print("=" * 60)
    
    for n, m in problem_sizes:
        try:
            results, filename = run_experiment_for_size(n, m)
            all_results[f"n{n}_m{m}"] = {
                "problem_size": (n, m),
                "results": results,
                "filename": filename
            }
            print(f"\n完成问题规模 n={n}, m={m} 的测试")
            print("-" * 40)
        except Exception as e:
            print(f"问题规模 n={n}, m={m} 测试失败: {str(e)}")
    
    # 保存总体汇总结果
    current_time = time.strftime("%Y%m%d_%H%M%S")
    summary_filename = f"有效不等式测试汇总_{current_time}.txt"
    
    with open(summary_filename, "w", encoding="utf-8") as f:
        f.write("多规模有效不等式测试汇总\n")
        f.write("=" * 60 + "\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        for size_key, data in all_results.items():
            n, m = data["problem_size"]
            f.write(f"问题规模: n={n}, m={m}\n")
            f.write(f"详细结果文件: {data['filename']}\n")
            f.write("-" * 40 + "\n")
            
            # 找到最佳TEC的配置（包括非最优解）
            best_config = None
            best_tec = float('inf')
            best_status = None
            for result in data['results']:
                if result['TEC'] != 'N/A' and isinstance(result['TEC'], (int, float)):
                    if result['TEC'] < best_tec:
                        best_tec = result['TEC']
                        best_config = result['配置']
                        best_status = result['状态']
            
            if best_config:
                f.write(f"最佳TEC配置: {best_config}, TEC值: {best_tec}, 状态: {best_status}\n")
            else:
                f.write("未找到可行解\n")
            
            f.write("\n")
    
    print(f"\n总体汇总结果已保存到: {summary_filename}")
    print("所有测试完成！")

if __name__ == "__main__":
    run_all_experiments() 