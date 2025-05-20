import numpy as np
import matplotlib.pyplot as plt
import config as cfg

# l1-l5得分映射
# fun_L = {1: 0, 2: 25, 3: 50, 4: 75, 5: 100}
fun_L = {1: 20, 2: 40, 3: 60, 4: 80, 5: 100}
# data_capability = [2, 2, 3, 2, 2, 3, 3, 5, 2, 3, 3, 2, 3]
# data_adoption = [3, 3, 3, 5, 5, 2, 1, 3, 4]
# data_utility = [4, 4, 5, 5, 5, 4, 4, 5]
# data_capability = [5,] * 13
# capability

def draw_capability():
    # 数据整理 (根据实际数据调整)
    material = {'name': 'Capability',
                'Multimodal Perception': 2,        # Yield strength (MPa)
                'Multimodal Alignment': 2,  # Electrical resistivity (μΩ·cm)
                'Multimodal Fusion': 3,       # Ultimate tensile strength (MPa)
                'Multimodal Reasoning': 2,                # Elongation to fracture (%)
                'risk judgment acc': 2,
                'risk pre-judgment timeliness': 3,
                'task execution completeness': 3,
                'task execution timeliness': 5,
                'task execution robustness': 2,
                'Model generalization': 3,
                'equipment Environmental adaptability': 3,
                'equipment sensors fusion degree': 2,
                'human-machine interaction maturity': 3,
            }
    score = {
        'Perception Capability': (fun_L[data[0]]+fun_L[data[1]])/2, 
        'Analytical Capability': (fun_L[data[2]]+fun_L[data[3]])/2,
        'Decision-Making Capability': (fun_L[data[4]]+fun_L[data[5]])/2,
        'Execution Capability': (fun_L[data[6]]+fun_L[data[7]]+fun_L[data[8]])/3,
        'Evolution Capability': fun_L[data[9]],
        'Environmental Adaptation': fun_L[data[10]],
        'Sensor Integration': fun_L[data[11]],
        'Human-Machine Interaction Maturity': fun_L[data[12]],}
    weight = {
        'Perception Capability': 0.195, 
        'Analytical Capability': 0.1625,
        'Decision-Making Capability': 0.14625,
        'Execution Capability': 0.0975,
        'Evolution Capability': 0.04875,
        'Environmental Adaptation': 0.14,
        'Sensor Integration': 0.1225,
        'Human-Machine Interaction Maturity': 0.0875,
    }
    res = score['Perception Capability']*weight['Perception Capability']+\
    score['Analytical Capability']*weight['Analytical Capability']+\
    score['Decision-Making Capability']*weight['Decision-Making Capability']+\
    score['Execution Capability']*weight['Execution Capability']+\
    score['Evolution Capability']*weight['Evolution Capability']+\
    score['Environmental Adaptation']*weight['Environmental Adaptation']+\
    score['Sensor Integration']*weight['Sensor Integration']+\
    score['Human-Machine Interaction Maturity']*weight['Human-Machine Interaction Maturity']
    # 创建极坐标图
    fig = plt.figure(figsize=(8, 8), dpi=300)
    # 提取标签和数值
    labels = list(material.keys())
    labels.remove('name')
    num_vars = len(labels)

    # 计算雷达图角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    ax1 = fig.add_subplot(111, projection='polar')
    ax1.set_theta_offset(np.pi / 2)  # 起始角度在顶部
    ax1.set_theta_direction(-1)      # 顺时针方向

    # 绘制每个材料
    values = [fun_L[material[f'{k}']] for k in labels]
    values += values[:1]  # 闭合图形

    ax1.tick_params(axis='y', labelsize=8)
    
    # 自定义数值范围 (根据实际数据调整)
    ax1.set_ylim(0, max([5, 4, 3, 2, 1]))
    
    # 绘制线条
    ax1.plot(angles, values, linewidth=2, linestyle='solid', label=material['name'], marker='^')
    ax1.fill(angles, values, alpha=0.1)

    # 设置轴标签
    r_label_position = 0  # 可以根据需要调整这个角度值
    ax1.set_rlabel_position(r_label_position)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(labels, size=6.5)
    ax1.set_title(f'{material["name"]}', y=1.1)

    # 添加图例和标题
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # 显示图形
    plt.tight_layout()
    plt.savefig(f'./{material["name"]}_1.jpg')

def draw_capability0(data, name, weight):
    # 数据整理 (根据实际数据调整)
    material = {'name': f'{name} Capability',
            }
    score = {
        'Perception Capability': (fun_L[data[0]]+fun_L[data[1]])/2, 
        'Analytical Capability': (fun_L[data[2]]+fun_L[data[3]])/2,
        'Decision-Making Capability': (fun_L[data[4]]+fun_L[data[5]])/2,
        'Execution Capability': (fun_L[data[6]]+fun_L[data[7]]+fun_L[data[8]])/3,
        'Evolution Capability': fun_L[data[9]],
        'Environmental Adaptation': fun_L[data[10]],
        'Sensor Integration': fun_L[data[11]],
        'Human-Machine Interaction Maturity': fun_L[data[12]],}
    
    res = score['Perception Capability']*weight['Perception Capability']+\
    score['Analytical Capability']*weight['Analytical Capability']+\
    score['Decision-Making Capability']*weight['Decision-Making Capability']+\
    score['Execution Capability']*weight['Execution Capability']+\
    score['Evolution Capability']*weight['Evolution Capability']+\
    score['Environmental Adaptation']*weight['Environmental Adaptation']+\
    score['Sensor Integration']*weight['Sensor Integration']+\
    score['Human-Machine Interaction Maturity']*weight['Human-Machine Interaction Maturity']

    material.update(score)

    # 创建极坐标图
    fig = plt.figure(figsize=(8, 8), dpi=300)
    # 提取标签和数值
    labels = list(material.keys())
    labels.remove('name')
    num_vars = len(labels)

    # 计算雷达图角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    ax1 = fig.add_subplot(111, projection='polar')
    ax1.set_theta_offset(np.pi / 2)  # 起始角度在顶部
    ax1.set_theta_direction(-1)      # 顺时针方向

    # 绘制每个材料
    values = [material[f'{k}'] for k in labels]
    values += values[:1]  # 闭合图形

    ax1.tick_params(axis='y', labelsize=8)
    
    # 自定义数值范围 (根据实际数据调整)
    ax1.set_ylim(0, max([100, 75, 50, 25, 0]))
    
    # 绘制线条
    ax1.plot(angles, values, linewidth=2, linestyle='solid', label=material['name'], marker='^')
    ax1.fill(angles, values, alpha=0.1)

    # 设置轴标签
    r_label_position = 0  # 可以根据需要调整这个角度值
    ax1.set_rlabel_position(r_label_position)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(labels, size=10)
    ax1.set_title(f'{material["name"]}', y=1.1)

    # 添加图例和标题
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # 显示图形
    plt.tight_layout()
    plt.savefig(f'./{material["name"]}_1.jpg')
    return res

def draw_capability0_new(data, name, weight):
    # 数据整理 (根据实际数据调整)
    material = {'name': f'{name} Capability Level',
            }
    score = {
        'Perception Capability': (data[0]+data[1])//2, 
        'Analytical Capability': (data[2]+data[3])//2,
        'Decision-Making Capability': (data[4]+data[5])//2,
        'Execution Capability': (data[6]+data[7]+data[8])//3,
        'Evolution Capability': data[9],
        'Environmental Adaptation': data[10],
        'Sensor Integration': data[11],
        'Human-Machine Interaction Maturity': data[12],}
    
    res = fun_L[score['Perception Capability']]*weight['Perception Capability']+\
    fun_L[score['Analytical Capability']]*weight['Analytical Capability']+\
    fun_L[score['Decision-Making Capability']]*weight['Decision-Making Capability']+\
    fun_L[score['Execution Capability']]*weight['Execution Capability']+\
    fun_L[score['Evolution Capability']]*weight['Evolution Capability']+\
    fun_L[score['Environmental Adaptation']]*weight['Environmental Adaptation']+\
    fun_L[score['Sensor Integration']]*weight['Sensor Integration']+\
    fun_L[score['Human-Machine Interaction Maturity']]*weight['Human-Machine Interaction Maturity']

    material.update(score)

    # 创建极坐标图
    fig = plt.figure(figsize=(8, 8), dpi=300)
    # 提取标签和数值
    labels = list(material.keys())
    labels.remove('name')
    num_vars = len(labels)

    # 计算雷达图角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    ax1 = fig.add_subplot(111, projection='polar')
    ax1.set_theta_offset(np.pi / 2)  # 起始角度在顶部
    ax1.set_theta_direction(-1)      # 顺时针方向

    # 绘制每个材料
    values = [material[f'{k}'] for k in labels]
    values += values[:1]  # 闭合图形

    ax1.tick_params(axis='y', labelsize=8)
    
    # 自定义数值范围 (根据实际数据调整)
    ax1.set_ylim(0, max([5, 4, 3, 2, 1, 0]))
    
    # 绘制线条
    ax1.plot(angles, values, linewidth=2, linestyle='solid', label=material['name'], marker='^')
    ax1.fill(angles, values, alpha=0.1)

    # 设置轴标签
    r_label_position = 0  # 可以根据需要调整这个角度值
    ax1.set_rlabel_position(r_label_position)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(labels, size=10)
    ax1.set_title(f'{material["name"]}', y=1.1)

    # 添加图例和标题
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # 显示图形
    plt.tight_layout()
    plt.savefig(f'./{material["name"]}.jpg')
    return res

def draw_capability0_score(data, name, weight):
    # 数据整理 (根据实际数据调整)
    material = {'name': f'{name} Capability Score',
            }
    score = {
        'Perception Capability': fun_L[(data[0]+data[1])//2], 
        'Analytical Capability': fun_L[(data[2]+data[3])//2],
        'Decision-Making Capability': fun_L[(data[4]+data[5])//2],
        'Execution Capability': fun_L[(data[6]+data[7]+data[8])//3],
        'Evolution Capability': fun_L[data[9]],
        'Environmental Adaptation': fun_L[data[10]],
        'Sensor Integration': fun_L[data[11]],
        'Human-Machine Interaction Maturity': fun_L[data[12]],}
    
    res = score['Perception Capability']*weight['Perception Capability']+\
    score['Analytical Capability']*weight['Analytical Capability']+\
    score['Decision-Making Capability']*weight['Decision-Making Capability']+\
    score['Execution Capability']*weight['Execution Capability']+\
    score['Evolution Capability']*weight['Evolution Capability']+\
    score['Environmental Adaptation']*weight['Environmental Adaptation']+\
    score['Sensor Integration']*weight['Sensor Integration']+\
    score['Human-Machine Interaction Maturity']*weight['Human-Machine Interaction Maturity']

    material.update(score)

    # 创建极坐标图
    fig = plt.figure(figsize=(8, 8), dpi=300)
    # 提取标签和数值
    labels = list(material.keys())
    labels.remove('name')
    num_vars = len(labels)

    # 计算雷达图角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    ax1 = fig.add_subplot(111, projection='polar')
    ax1.set_theta_offset(np.pi / 2)  # 起始角度在顶部
    ax1.set_theta_direction(-1)      # 顺时针方向

    # 绘制每个材料
    values = [material[f'{k}'] for k in labels]
    values += values[:1]  # 闭合图形

    ax1.tick_params(axis='y', labelsize=8)
    
    # 自定义数值范围 (根据实际数据调整)
    ax1.set_ylim(0, max([100, 75, 50, 25, 0]))
    
    # 绘制线条
    ax1.plot(angles, values, linewidth=2, linestyle='solid', label=material['name'], marker='^')
    ax1.fill(angles, values, alpha=0.1)

    # 设置轴标签
    r_label_position = 0  # 可以根据需要调整这个角度值
    ax1.set_rlabel_position(r_label_position)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(labels, size=10)
    ax1.set_title(f'{material["name"]}', y=1.1)

    # 添加图例和标题
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # 显示图形
    plt.tight_layout()
    plt.savefig(f'./{material["name"]}.jpg')
    return res

def draw_capability1(data, name, weight):
    # 数据整理 (根据实际数据调整)
    material = {'name': f'{name} Capability Level',
            }
    score = {
        'Perception Capability': data[0], 
        'Analytical Capability': data[1],
        'Decision-Making Capability': data[2],
        'Execution Capability': data[3],
        'Evolution Capability': data[4],
        'Environmental Adaptation': data[5],
        'Sensor Integration': data[6],
        'Human-Machine Interaction Maturity': data[7],}
    
    material.update(score)

    # 创建极坐标图
    fig = plt.figure(figsize=(8, 8), dpi=300)
    # 提取标签和数值
    labels = list(material.keys())
    labels.remove('name')
    num_vars = len(labels)

    # 计算雷达图角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    ax1 = fig.add_subplot(111, projection='polar')
    ax1.set_theta_offset(np.pi / 2)  # 起始角度在顶部
    ax1.set_theta_direction(-1)      # 顺时针方向

    # 绘制每个材料
    values = [material[f'{k}'] for k in labels]
    values += values[:1]  # 闭合图形

    ax1.tick_params(axis='y', labelsize=8)
    
    # 自定义数值范围 (根据实际数据调整)
    ax1.set_ylim(0, max([5, 4, 3, 2, 1, 0]))
    
    # 绘制线条
    ax1.plot(angles, values, linewidth=2, linestyle='solid', label=material['name'], marker='^')
    ax1.fill(angles, values, alpha=0.1)

    # 设置轴标签
    r_label_position = 0  # 可以根据需要调整这个角度值
    ax1.set_rlabel_position(r_label_position)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(labels, size=10)
    ax1.set_title(f'{material["name"]}', y=1.1)

    # 添加图例和标题
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # 显示图形
    plt.tight_layout()
    plt.savefig(f'./{material["name"]}.jpg')
    
def draw_capability_score(data, name, weight):
    # 数据整理 (根据实际数据调整)
    material = {'name': f'{name} Capability Score',
            }
    score = {
        'Perception Capability': fun_L[data[0]], 
        'Analytical Capability': fun_L[data[1]],
        'Decision-Making Capability': fun_L[data[2]],
        'Execution Capability': fun_L[data[3]],
        'Evolution Capability': fun_L[data[4]],
        'Environmental Adaptation': fun_L[data[5]],
        'Sensor Integration': fun_L[data[6]],
        'Human-Machine Interaction Maturity': fun_L[data[7]],}
    
    res = score['Perception Capability']*weight['Perception Capability']+\
    score['Analytical Capability']*weight['Analytical Capability']+\
    score['Decision-Making Capability']*weight['Decision-Making Capability']+\
    score['Execution Capability']*weight['Execution Capability']+\
    score['Evolution Capability']*weight['Evolution Capability']+\
    score['Environmental Adaptation']*weight['Environmental Adaptation']+\
    score['Sensor Integration']*weight['Sensor Integration']+\
    score['Human-Machine Interaction Maturity']*weight['Human-Machine Interaction Maturity']

    material.update(score)

    # 创建极坐标图
    fig = plt.figure(figsize=(8, 8), dpi=300)
    # 提取标签和数值
    labels = list(material.keys())
    labels.remove('name')
    num_vars = len(labels)

    # 计算雷达图角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    ax1 = fig.add_subplot(111, projection='polar')
    ax1.set_theta_offset(np.pi / 2)  # 起始角度在顶部
    ax1.set_theta_direction(-1)      # 顺时针方向

    # 绘制每个材料
    values = [material[f'{k}'] for k in labels]
    values += values[:1]  # 闭合图形

    ax1.tick_params(axis='y', labelsize=8)
    
    # 自定义数值范围 (根据实际数据调整)
    ax1.set_ylim(0, max([100, 80, 60, 40, 20]))
    
    # 绘制线条
    ax1.plot(angles, values, linewidth=2, linestyle='solid', label=material['name'], marker='^')
    ax1.fill(angles, values, alpha=0.1)

    # 设置轴标签
    r_label_position = 0  # 可以根据需要调整这个角度值
    ax1.set_rlabel_position(r_label_position)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(labels, size=10)
    ax1.set_title(f'{material["name"]}', y=1.1)

    # 添加图例和标题
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # 显示图形
    plt.tight_layout()
    plt.savefig(f'./{material["name"]}.jpg')
    return res

# adoption 
def draw_adoption():
    # 数据整理 (根据实际数据调整)
    material = {'name': 'Adaptability',
                'Transformation Difficulty': 3,        # Yield strength (MPa)
                'Scenario Migration Ability': 3,  # Electrical resistivity (μΩ·cm)
                'Technical Absorption Capacity': 3,       # Ultimate tensile strength (MPa)
                'Digital Infrastructure': 5,                # Elongation to fracture (%)
                'Change Management Ability & Willingness for Change': 5,
                'Upstream and Downstream Ecology': 2,
                'Perfection Degree of Industry Standards': 1,
                'Value Chain Optimization': 4,
            }
    # 创建极坐标图
    fig = plt.figure(figsize=(8, 8), dpi=300)
    # 提取标签和数值
    labels = list(material.keys())
    labels.remove('name')
    num_vars = len(labels)

    # 计算雷达图角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    ax1 = fig.add_subplot(111, projection='polar')
    ax1.set_theta_offset(np.pi / 2)  # 起始角度在顶部
    ax1.set_theta_direction(-1)      # 顺时针方向

    # 绘制每个材料
    values = [material[f'{k}'] for k in labels]
    values += values[:1]  # 闭合图形

    ax1.tick_params(axis='y', labelsize=8)
    
    # 自定义数值范围 (根据实际数据调整)
    ax1.set_ylim(0, max([5, 4, 3, 2, 1]))
    
    # 绘制线条
    ax1.plot(angles, values, linewidth=2, linestyle='solid', label=material['name'], marker='s', color='r')
    ax1.fill(angles, values, alpha=0.1, color='r')

    # 设置轴标签
    r_label_position = 0  # 可以根据需要调整这个角度值
    ax1.set_rlabel_position(r_label_position)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(labels, size=6.5)
    ax1.set_title(f'{material["name"]}', y=1.1)

    # 添加图例和标题
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # 显示图形
    plt.tight_layout()
    plt.savefig(f'./{material["name"]}.jpg')

def draw_adoption_rate(data, name, weight):
    # 数据整理 (根据实际数据调整)
    material = {'name': f'{name} Adoption Level',
            }

    score = {
        'Modification Difficulty': data[0], 
        'Scene Transfer Difficulty': data[1], 
        'Technical Absorption Capacity': data[2],
        'Digital Infrastructure': data[3],
        'Change Management Capability': data[4],
        'Upstream-Downstream Ecosystem': data[5],
        'Standards Completeness': data[6],
        'Policy Compatibility': data[7],
        'Value Chain Optimization': data[8],}

    res = fun_L[score['Modification Difficulty']]*weight['Modification Difficulty']+\
     fun_L[score['Scene Transfer Difficulty']]*weight['Scene Transfer Difficulty']+\
     fun_L[score['Technical Absorption Capacity']]*weight['Technical Absorption Capacity']+\
     fun_L[score['Digital Infrastructure']]*weight['Digital Infrastructure']+\
     fun_L[score['Change Management Capability']]*weight['Change Management Capability']+\
     fun_L[score['Upstream-Downstream Ecosystem']]*weight['Upstream-Downstream Ecosystem']+\
     fun_L[score['Standards Completeness']]*weight['Standards Completeness']+\
     fun_L[score['Policy Compatibility']]*weight['Policy Compatibility']+\
     fun_L[score['Value Chain Optimization']]*weight['Value Chain Optimization']

    material.update(score)
    
    # 创建极坐标图
    fig = plt.figure(figsize=(8, 8), dpi=300)
    # 提取标签和数值
    labels = list(material.keys())
    labels.remove('name')
    num_vars = len(labels)

    # 计算雷达图角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    ax1 = fig.add_subplot(111, projection='polar')
    ax1.set_theta_offset(np.pi / 2)  # 起始角度在顶部
    ax1.set_theta_direction(-1)      # 顺时针方向

    # 绘制每个材料
    values = [material[f'{k}'] for k in labels]
    values += values[:1]  # 闭合图形

    ax1.tick_params(axis='y', labelsize=8)
    
    # 自定义数值范围 (根据实际数据调整)
    ax1.set_ylim(0, max([5, 4, 3, 2, 1, 0]))
    
    # 绘制线条
    ax1.plot(angles, values, linewidth=2, linestyle='solid', label=material['name'], marker='s', color='r')
    ax1.fill(angles, values, alpha=0.1, color='r')

    # 设置轴标签
    r_label_position = 0  # 可以根据需要调整这个角度值
    ax1.set_rlabel_position(r_label_position)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(labels, size=10)
    ax1.set_title(f'{material["name"]}', y=1.1)

    # 添加图例和标题
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # 显示图形
    plt.tight_layout()
    plt.savefig(f'./{material["name"]}.jpg')
    return res

def draw_adoption_score(c, data, name, weight):
    # 数据整理 (根据实际数据调整)
    material = {'name': f'{name} Adoption Score',
            }

    score = {
        'Modification Difficulty': c/100*fun_L[data[0]], 
        'Scene Transfer Difficulty': c/100*fun_L[data[1]], 
        'Technical Absorption Capacity': c/100*fun_L[data[2]],
        'Digital Infrastructure': c/100*fun_L[data[3]],
        'Change Management Capability': c/100*fun_L[data[4]],
        'Upstream-Downstream Ecosystem': c/100*fun_L[data[5]],
        'Standards Completeness': c/100*fun_L[data[6]],
        'Policy Compatibility': c/100*fun_L[data[7]],
        'Value Chain Optimization': c/100*fun_L[data[8]],}

    res = score['Modification Difficulty']*weight['Modification Difficulty']+\
    score['Scene Transfer Difficulty']*weight['Scene Transfer Difficulty']+\
    score['Technical Absorption Capacity']*weight['Technical Absorption Capacity']+\
    score['Digital Infrastructure']*weight['Digital Infrastructure']+\
    score['Change Management Capability']*weight['Change Management Capability']+\
    score['Upstream-Downstream Ecosystem']*weight['Upstream-Downstream Ecosystem']+\
    score['Standards Completeness']*weight['Standards Completeness']+\
    score['Policy Compatibility']*weight['Policy Compatibility']+\
    score['Value Chain Optimization']*weight['Value Chain Optimization']

    material.update(score)
    
    # 创建极坐标图
    fig = plt.figure(figsize=(8, 8), dpi=300)
    # 提取标签和数值
    labels = list(material.keys())
    labels.remove('name')
    num_vars = len(labels)

    # 计算雷达图角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    ax1 = fig.add_subplot(111, projection='polar')
    ax1.set_theta_offset(np.pi / 2)  # 起始角度在顶部
    ax1.set_theta_direction(-1)      # 顺时针方向

    # 绘制每个材料
    values = [material[f'{k}'] for k in labels]
    values += values[:1]  # 闭合图形

    ax1.tick_params(axis='y', labelsize=8)
    
    # 自定义数值范围 (根据实际数据调整)
    ax1.set_ylim(0, max([100, 75, 50, 25, 0]))
    
    # 绘制线条
    ax1.plot(angles, values, linewidth=2, linestyle='solid', label=material['name'], marker='s', color='r')
    ax1.fill(angles, values, alpha=0.1, color='r')

    # 设置轴标签
    r_label_position = 0  # 可以根据需要调整这个角度值
    ax1.set_rlabel_position(r_label_position)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(labels, size=10)
    ax1.set_title(f'{material["name"]}', y=1.1)

    # 添加图例和标题
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # 显示图形
    plt.tight_layout()
    plt.savefig(f'./{material["name"]}.jpg')
    return res

# utility
def draw_utility():
    # 数据整理 (根据实际数据调整)
    material = {'name': 'Utility',
                'Quality Improvement Coeff': 4,        # Yield strength (MPa)
                'Cost Reduction Coeff': 4,  # Electrical resistivity (μΩ·cm)
                'Efficiency Improvement Coeff': 5,       # Ultimate tensile strength (MPa)
                'Risk Prevention Coeff': 5,                # Elongation to fracture (%)
                'Value Density Coeff': 5,
                'Environmental Coeff': 4,
                'Social Coeff': 4,
                'Governance Coeff': 5
            }
    # 创建极坐标图
    fig = plt.figure(figsize=(8, 8), dpi=300)
    # 提取标签和数值
    labels = list(material.keys())
    labels.remove('name')
    num_vars = len(labels)

    # 计算雷达图角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    ax1 = fig.add_subplot(111, projection='polar')
    ax1.set_theta_offset(np.pi / 2)  # 起始角度在顶部
    ax1.set_theta_direction(-1)      # 顺时针方向

    # 绘制每个材料
    values = [material[f'{k}'] for k in labels]
    values += values[:1]  # 闭合图形

    ax1.tick_params(axis='y', labelsize=8)
    
    # 自定义数值范围 (根据实际数据调整)
    ax1.set_ylim(0, max([5, 4, 3, 2, 1]))
    
    # 绘制线条
    ax1.plot(angles, values, linewidth=2, linestyle='solid', label=material['name'], marker='*', color='yellow')
    ax1.fill(angles, values, alpha=0.1, color='yellow')

    # 设置轴标签
    r_label_position = 0  # 可以根据需要调整这个角度值
    ax1.set_rlabel_position(r_label_position)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(labels, size=6.5)
    ax1.set_title(f'{material["name"]}', y=1.1)

    # 添加图例和标题
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # 显示图形
    plt.tight_layout()
    plt.savefig(f'./{material["name"]}.jpg')

def draw_utility_rate(data, name, weight):
    # 数据整理 (根据实际数据调整)
    material = {'name': f'{name} Utility Level',
            }
    
    score = {
        'Quality Improvement': data[0], 
        'Cost Reduction': data[1], 
        'Efficiency Enhancement': data[2],
        'Risk Prevention': data[3],
        'Value Density Coefficient': data[4],
        'Environmental Metrics': data[5],
        'Social Metrics': data[6],
        'Governance Metrics': data[7],}

    res = fun_L[score['Quality Improvement']]*weight['Quality Improvement']+\
    fun_L[score['Cost Reduction']]*weight['Cost Reduction']+\
    fun_L[score['Efficiency Enhancement']]*weight['Efficiency Enhancement']+\
    fun_L[score['Risk Prevention']]*weight['Risk Prevention']+\
    fun_L[score['Value Density Coefficient']]*weight['Value Density Coefficient']+\
    fun_L[score['Environmental Metrics']]*weight['Environmental Metrics']+\
    fun_L[score['Social Metrics']]*weight['Social Metrics']+\
    fun_L[score['Governance Metrics']]*weight['Governance Metrics']

    material.update(score)

    # 创建极坐标图
    fig = plt.figure(figsize=(8, 8), dpi=300)
    # 提取标签和数值
    labels = list(material.keys())
    labels.remove('name')
    num_vars = len(labels)

    # 计算雷达图角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    ax1 = fig.add_subplot(111, projection='polar')
    ax1.set_theta_offset(np.pi / 2)  # 起始角度在顶部
    ax1.set_theta_direction(-1)      # 顺时针方向

    # 绘制每个材料
    values = [material[f'{k}'] for k in labels]
    values += values[:1]  # 闭合图形

    ax1.tick_params(axis='y', labelsize=8)
    
    # 自定义数值范围 (根据实际数据调整)
    ax1.set_ylim(0, max([5, 4, 3, 2, 1, 0]))
    
    # 绘制线条
    ax1.plot(angles, values, linewidth=2, linestyle='solid', label=material['name'], marker='*', color='yellow')
    ax1.fill(angles, values, alpha=0.1, color='yellow')

    # 设置轴标签
    r_label_position = 0  # 可以根据需要调整这个角度值
    ax1.set_rlabel_position(r_label_position)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(labels, size=10)
    ax1.set_title(f'{material["name"]}', y=1.1)

    # 添加图例和标题
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # 显示图形
    plt.tight_layout()
    plt.savefig(f'./{material["name"]}.jpg')
    return res

def draw_utility_score(c, data, name, weight):
    # 数据整理 (根据实际数据调整)
    material = {'name': f'{name} Utility Score',
            }
    
    score = {
        'Quality Improvement': c/100*fun_L[data[0]], 
        'Cost Reduction': c/100*fun_L[data[1]], 
        'Efficiency Enhancement': c/100*fun_L[data[2]],
        'Risk Prevention': c/100*fun_L[data[3]],
        'Value Density Coefficient': c/100*fun_L[data[4]],
        'Environmental Metrics': c/100*fun_L[data[5]],
        'Social Metrics': c/100*fun_L[data[6]],
        'Governance Metrics': c/100*fun_L[data[7]],}

    res = score['Quality Improvement']*weight['Quality Improvement']+\
    score['Cost Reduction']*weight['Cost Reduction']+\
    score['Efficiency Enhancement']*weight['Efficiency Enhancement']+\
    score['Risk Prevention']*weight['Risk Prevention']+\
    score['Value Density Coefficient']*weight['Value Density Coefficient']+\
    score['Environmental Metrics']*weight['Environmental Metrics']+\
    score['Social Metrics']*weight['Social Metrics']+\
    score['Governance Metrics']*weight['Governance Metrics']

    material.update(score)

    # 创建极坐标图
    fig = plt.figure(figsize=(8, 8), dpi=300)
    # 提取标签和数值
    labels = list(material.keys())
    labels.remove('name')
    num_vars = len(labels)

    # 计算雷达图角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    ax1 = fig.add_subplot(111, projection='polar')
    ax1.set_theta_offset(np.pi / 2)  # 起始角度在顶部
    ax1.set_theta_direction(-1)      # 顺时针方向

    # 绘制每个材料
    values = [material[f'{k}'] for k in labels]
    values += values[:1]  # 闭合图形

    ax1.tick_params(axis='y', labelsize=8)
    
    # 自定义数值范围 (根据实际数据调整)
    ax1.set_ylim(0, max([100, 75, 50, 25, 0]))
    
    # 绘制线条
    ax1.plot(angles, values, linewidth=2, linestyle='solid', label=material['name'], marker='*', color='yellow')
    ax1.fill(angles, values, alpha=0.1, color='yellow')

    # 设置轴标签
    r_label_position = 0  # 可以根据需要调整这个角度值
    ax1.set_rlabel_position(r_label_position)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(labels, size=10)
    ax1.set_title(f'{material["name"]}', y=1.1)

    # 添加图例和标题
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # 显示图形
    plt.tight_layout()
    plt.savefig(f'./{material["name"]}.jpg')
    return res

# 门店
def draw_mendian():
    name = 'retail store inspection'
    data_capability = [2, 2, 3, 2, 2, 3, 3, 5, 2, 3, 3, 2, 3]
    data_adoption = [3, 3, 3, 5, 5, 2, 1, 3, 4]
    data_utility = [4, 4, 5, 5, 5, 4, 4, 5]
    weight = cfg.Store_weight_capability
    c = draw_capability0(data_capability, name, weight)
    # adoption
    weight = cfg.Store_weight_adoption
    r_a = draw_adoption_rate(data_adoption, name, weight)
    c_a = draw_adoption_score(c, data_adoption, name, weight)
    # utility
    weight = cfg.Store_weight_utility
    r_u = draw_utility_rate(data_utility, name, weight)
    c_u = draw_utility_score(c_a*r_a/100, data_utility, name, weight)
    print(f'初始capability={c}, adoption转化率为{r_a}%,经adoption转化capability为{c_a},utility转化价值比率为{r_u}%, 经价值转化评估体系评估capability价值转化值为{c_u}')

def draw_mendian_new():
    name = 'retail store inspection'
    data_capability = [4, 3, 2, 3, 2, 3, 2, 3]
    data_adoption = [3, 2, 4, 3, 2, 2, 3, 4, 3]
    data_utility = [3, 4, 5, 3, 4, 2, 3, 4]
    weight = cfg.Store_weight_capability_g
    draw_capability1(data_capability, name, weight)
    c = draw_capability_score(data_capability, name, weight)
    # adoption
    weight = cfg.Store_weight_adoption_g
    r_a = draw_adoption_rate(data_adoption, name, weight)
    c_a = draw_adoption_score(c, data_adoption, name, weight)
    # utility
    weight = cfg.Store_weight_utility_g
    r_u = draw_utility_rate(data_utility, name, weight)
    c_u = draw_utility_score(c_a, data_utility, name, weight)
    print(f'初始capability={c}, adoption转化率为{r_a}%,经adoption转化capability为{c_a},utility转化价值比率为{r_u}%, 经价值转化评估体系评估capability价值转化值为{c_u}')

# 清洁能源

def draw_energe():
    name = 'photovoltaic system inspection'
    data_capability = [4, 5, 2, 5, 2, 5, 5, 2, 5, 1, 5, 2, 5]
    data_adoption = [2, 4, 2, 4, 4, 2, 4, 4, 5, 4]
    data_utility = [4, 2, 4, 5, 4, 5, 4, 2]
    weight = cfg.PV_weight_capability
    draw_capability0_new(data_capability, name, weight)
    c = draw_capability0_score(data_capability, name, weight)
    # adoption
    weight = cfg.PV_weight_adoption
    r_a = draw_adoption_rate(data_adoption, name, weight)
    c_a = draw_adoption_score(c, data_adoption, name, weight)
    # utility
    weight = cfg.PV_weight_utility
    r_u = draw_utility_rate(data_utility, name, weight)
    c_u = draw_utility_score(c_a, data_utility, name, weight)
    print(f'初始capability={c}, adoption转化率为{r_a}%,经adoption转化capability为{c_a},utility转化价值比率为{r_u}%, 经价值转化评估体系评估capability价值转化值为{c_u}')

if __name__ == '__main__':
    # draw_mendian_new()
    draw_energe()