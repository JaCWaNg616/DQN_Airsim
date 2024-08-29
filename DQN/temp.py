import airsim
import time

# 连接到AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# 起飞并悬停在高度为10米的位置
client.takeoffAsync().join()

# 设置风速向量，模拟风的影响 (例如, X方向-1 m/s, Y方向0.5 m/s, Z方向0 m/s)
wind = airsim.Vector3r(0, 10, 0.0)
client.simSetWind(wind)


# 保持飞行一段时间，观察风对无人机的影响
for _ in range(10):
    client.moveByVelocityAsync(1.0, 0.0, 0.0, 1)
    state = client.getMultirotorState()
    velocity = state.kinematics_estimated.linear_velocity
    print(f"当前速度: x={velocity.x_val:.2f}, y={velocity.y_val:.2f}, z={velocity.z_val:.2f}")
    time.sleep(1)

# 降落并结束
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
