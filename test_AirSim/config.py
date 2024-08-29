class Config:
    ALPHA = 0.1  # 学习率
    GAMMA = 0.9  # 折扣因子
    EPSILON =1  # 探索率
    EPSILON_DECAY = 0.995  # 探索率衰减
    EPSILON_MIN = 0.03  # 最小探索率
    NUM_EPISODES = 3000  # 训练回合数
    MAX_STEPS = 150  # 每回合最大步数
    NUM_BINS = 7  # 激光雷达数据的离散化bin数
    MAX_DISTANCE = 8  # 激光雷达最大距离
    ANG_VEL_BINS = 7  # 角速度离散化bin数
    LIN_ACC_BINS = 7  # 线性加速度离散化bin数
    ACTIONS = [
        [1, 1, 0, 1],  # 右前，低速
        [1, -1, 0, 1],  # 右后，低速
        [-1, 1, 0, 1],  # 左前，低速
        [-1, -1, 0, 1],  # 左后，低速
        [1, 0, 0, 1],  # 前进，低速
        [-1, 0, 0, 1],  # 后退，低速
        [0, 1, 0, 1],  # 左移，低速
        [0, -1, 0, 1],  # 右移，低速
        [0, 0, 1, 1],  # 上升，低速
        [0, 0, -1, 1],  # 下降，低速
    ]

    GOAL_POSITION = [11, 11, -1]  # 目标点
    GOAL_THRESHOLD = 1.0  # 目标点周围的范围（米）
    REWARD_GOAL = 100  # 到达目标点的奖励
    REWARD_COLLISION = -100  # 碰撞的惩罚
    REWARD_STEP = -2  # 每步的惩罚
    REWARD_APPROACH = 10  # 朝向目标点的奖励
    REWARD_DIRECTION = 10  # 朝向目标点方向的奖励
    COLLISION_THRESHOLD = 3  # 碰撞阈值（米）
    SAFE_DISTANCE = 3  # 安全距离（米）
    REWARD_AVOID_OBSTACLE = 8  # 避开障碍物的奖励