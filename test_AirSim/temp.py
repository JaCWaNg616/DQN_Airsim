import numpy as np

# 示例数组
lidar_state = np.zeros(4708)
ang_vel_state = np.zeros(3)
lin_acc_state = np.zeros(3)

# 将状态组合成一个数组
state = np.concatenate([lidar_state, ang_vel_state, lin_acc_state])

# 打印数组的形状和内容
print("lidar_state shape:", lidar_state.shape)
print("ang_vel_state shape:", ang_vel_state.shape)
print("lin_acc_state shape:", lin_acc_state.shape)
print("State shape:", state.shape)

# 假设这是要访问的索引
index = 4714

# 检查索引是否在有效范围内
if index < state.size:
    print("Accessing index:", index)
    element = state[index]
    print("Element at index", index, ":", element)
else:
    print(f"Index {index} is out of bounds for axis 0 with size {state.size}")
