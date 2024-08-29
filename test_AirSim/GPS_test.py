import airsim
import time

def main():
    # 连接到AirSim模拟器
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True, "Drone1")
    client.armDisarm(True, "Drone1")

    # 起飞并悬停
    print("Taking off...")
    client.takeoffAsync(vehicle_name="Drone1").join()
    client.moveToZAsync(-2, 1, vehicle_name="Drone1").join()
    print("Hovering...")
    time.sleep(2)  # 悬停2秒

    # 持续前进并打印位置
    print("Moving forward and printing precise position data...")
    vx = 0  # x方向速度为1 m/s
    vy = 1  # y方向速度为0
    vz = 0  # z方向速度为0
    duration = 15  # 每次移动的持续时间，单位为秒

    # 第一次飞行15秒
    for _ in range(duration):
        # 控制无人机前进
        client.moveByVelocityAsync(vx, vy, vz, 1, vehicle_name="Drone1").join()

        # 获取并打印无人机位置
        state = client.getMultirotorState(vehicle_name="Drone1")
        position = state.kinematics_estimated.position
        print(f"当前无人机位置 - x: {position.x_val:.2f}, y: {position.y_val:.2f}, z: {position.z_val:.2f}")

        time.sleep(1)  # 每秒打印一次位置

    # 反方向飞行15秒
    vx = -vx  # 取反方向速度
    for _ in range(duration):
        # 控制无人机反方向前进
        client.moveByVelocityAsync(vx, vy, vz, 1, vehicle_name="Drone1").join()

        # 获取并打印无人机位置
        state = client.getMultirotorState(vehicle_name="Drone1")
        position = state.kinematics_estimated.position
        print(f"当前无人机位置 - x: {position.x_val:.2f}, y: {position.y_val:.2f}, z: {position.z_val:.2f}")

        time.sleep(1)  # 每秒打印一次位置

    # 悬停并降落
    client.hoverAsync(vehicle_name="Drone1").join()
    time.sleep(2)
    print("Landing...")
    client.landAsync(vehicle_name="Drone1").join()
    client.armDisarm(False, "Drone1")
    client.enableApiControl(False, "Drone1")

if __name__ == "__main__":
    main()
