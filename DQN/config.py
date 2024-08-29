import numpy as np

class Config:
    # 静态配置项
    MEMORY_SIZE = 10000  # 经验回放缓冲区大小
    GAMMA = 0.95  # 折扣因子
    ALPHA = 0.0005  # 学习率
    EPSILON = 1.0  # 初始探索率
    EPSILON_MIN = 0.02  # 最小探索率
    EPSILON_DECAY = 0.995  # 探索率衰减
    BATCH_SIZE = 32  # 批量大小
    NUM_EPISODES = 5000  # 训练回合数
    MAX_TIMESTEPS = 150  # 每个回合的最大时间步数
    GOAL_POSITION = [11, 11, -1]  # 目标位置 (示例)
    GOAL_THRESHOLD = 1.0  # 目标位置的阈值
    REWARD_GOAL = 100.0  # 达到目标的奖励
    REWARD_DISTANCE_REDUCTION = 10  # 距离变小时的奖励
    PENALTY_DISTANCE_INCREASE = -10  # 距离增大时的惩罚
    REWARD_SPEED_INCREASE = 4  # 距离变化速度增加时的奖励
    PENALTY_SPEED_DECREASE = -4  # 距离变化速度减小时的惩罚
    REWARD_DIRECTION_CORRECT = 6  # 方向正确时的奖励
    PENALTY_DIRECTION_WRONG = -6  # 方向错误时的惩罚
    REWARD_OBSTACLE_AVOIDANCE = 10  # 远离障碍物的奖励
    PENALTY_OBSTACLE_APPROACH = -10  # 接近障碍物的惩罚
    PENALTY_COLLISION = -100  # 碰撞的惩罚
    CLOCK_SPEED = 5 # 仿真速度

    # 固定的 ACTIONS 列表
    ACTIONS = [
        [1, 0, 0, 1],  # 向前低速
        [2, 0, 0, 1],  # 向前中速
        [0, 1, 0, 1],  # 向右低速
        [0, 2, 0, 1],  # 向右中速
        [-1, 0, 0, 1],  # 向后低速
        [-2, 0, 0, 1],  # 向后中速
        [0, -1, 0, 1],  # 向左低速
        [0, -2, 0, 1],  # 向左中速
        [0, 0, 0.5, 1],  # 向上低速
        [0, 0, 1, 1],  # 向上中速
        [0, 0, -0.5, 1],  # 向下低速
        [0, 0, -1, 1],  # 向下中速
        [0.7, 0.7, 0, 1],  # 右前方低速
        [1.4, 1.4, 0, 1],  # 右前方中速
        [-0.7, 0.7, 0, 1],  # 右后方低速
        [-1.4, 1.4, 0, 1],  # 右后方中速
        [0.7, -0.7, 0, 1],  # 左前方低速
        [1.4, -1.4, 0, 1],  # 左前方中速
        [-0.7, -0.7, 0, 1],  # 左后方低速
        [-1.4, -1.4, 0, 1]
    ]# 左后方中速
# 其他方向的动作可以继续添加