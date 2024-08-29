import numpy as np
from config import Config


def get_state(client, drone, num_bins=Config.NUM_BINS, max_distance=Config.MAX_DISTANCE,
              ang_vel_bins=Config.ANG_VEL_BINS, lin_acc_bins=Config.LIN_ACC_BINS):
    # 获取位置信息
    position = client.getMultirotorState(vehicle_name=drone).kinematics_estimated.position
    position_state = np.array([position.x_val, position.y_val, position.z_val])

    # 获取角速度和线性加速度
    angular_velocity = client.getMultirotorState(vehicle_name=drone).kinematics_estimated.angular_velocity
    angular_velocity_state = np.digitize(angular_velocity.to_numpy_array(), bins=np.linspace(-1, 1, ang_vel_bins))

    linear_acceleration = client.getMultirotorState(vehicle_name=drone).kinematics_estimated.linear_acceleration
    linear_acceleration_state = np.digitize(linear_acceleration.to_numpy_array(), bins=np.linspace(-1, 1, lin_acc_bins))

    # 获取激光雷达数据
    lidar_data = client.getLidarData(vehicle_name=drone)
    if len(lidar_data.point_cloud) < 3:
        lidar_state = np.zeros(num_bins)
    else:
        points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
        distances = np.linalg.norm(points[:, :2], axis=1)
        bins = np.linspace(0, max_distance, num_bins + 1)
        lidar_state = np.histogram(distances, bins=bins)[0]

    # 计算相对目标点的位置
    goal_position = np.array(Config.GOAL_POSITION)
    relative_position = goal_position - np.array([position.x_val, position.y_val, position.z_val])

    # 将所有状态信息合并
    state = np.concatenate(
        [position_state, angular_velocity_state, linear_acceleration_state, lidar_state, relative_position])
    return state


def take_action(client, drone, action):
    # 从动作索引中获取具体的动作参数
    vx, vy, vz, speed = action
    # 执行动作
    client.moveByVelocityAsync(vx, vy, vz, speed, vehicle_name=drone).join()

def calculate_distance_to_goal(position):
    # 计算当前位置到目标位置的距离
    goal_position = np.array(Config.GOAL_POSITION)
    current_position = np.array(position)
    distance = np.linalg.norm(goal_position - current_position)
    return distance

def is_collision(client, drone):
    # 检测碰撞
    collision_info = client.simGetCollisionInfo(vehicle_name=drone)
    has_collided = collision_info.has_collided
    return has_collided

