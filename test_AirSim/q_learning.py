import numpy as np
import pickle
import random
from hashlib import sha256
from config import Config
from drone_control import get_state, take_action, is_collision, calculate_distance_to_goal
from sklearn.cluster import DBSCAN
import os
# 初始化Q表
state_shape = (Config.NUM_BINS,) * 3 + (Config.ANG_VEL_BINS,) * 3 + (Config.LIN_ACC_BINS,) * 3
Q_table_shape = np.prod(state_shape + (len(Config.ACTIONS),))
Q = np.zeros(Q_table_shape)

def get_lidar_data(client, vehicle_name):
    lidar_data = client.getLidarData(vehicle_name=vehicle_name).point_cloud
    points = np.array(lidar_data).reshape(-1, 3)  # 将点云数据转换为 (N, 3) 形状
    return points

def cluster_points_dbscan(points, eps, min_samples):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    return clustering.labels_

def calculate_cluster_center(points):
    return np.mean(points, axis=0)

def calculate_cluster_width(points):
    return np.max(points, axis=0) - np.min(points, axis=0)

def extract_clusters(points, distances, eps_params):
    normal_vector = np.array([0, 0, 1])  # 假设无人机的朝向为Z轴正方向
    projection_distances = np.dot(points, normal_vector)
    min_projection = np.min(projection_distances)
    floor_threshold = min_projection + 0.5
    floor_points = projection_distances <= floor_threshold
    all_labels = np.full(points.shape[0], -1, dtype=int)
    all_labels[floor_points] = -2
    remaining_points = points[~floor_points]
    remaining_distances = distances[~floor_points]

    cluster_centers = []
    cluster_widths = []

    masks = [
        remaining_distances <= 4,
        (remaining_distances > 4) & (remaining_distances <= 6),
        (remaining_distances > 6) & (remaining_distances <= 8)
    ]

    for mask, (eps, min_samples) in zip(masks, eps_params):
        if np.any(mask):
            labels = cluster_points_dbscan(remaining_points[mask], eps, min_samples)
            unique_labels = set(labels)
            for label in unique_labels:
                if label != -1:
                    obstacle_indices = np.where(mask)[0][labels == label]
                    original_indices = np.where(~floor_points)[0][obstacle_indices]
                    all_labels[original_indices] = label
                    cluster_center = calculate_cluster_center(remaining_points[mask][labels == label])
                    cluster_width = calculate_cluster_width(remaining_points[mask][labels == label])
                    cluster_centers.append(cluster_center)
                    cluster_widths.append(cluster_width)

    return cluster_centers, cluster_widths

def save_q_table(q_table: np.ndarray, filename: str) -> None:
    with open(filename, 'wb') as f:
        pickle.dump(q_table, f)


def load_q_table(filename: str) -> np.ndarray:
    with open(filename, 'rb') as f:
        return pickle.load(f)


def hash_state(state, table_size):
    state_bytes = str(state).encode()
    hash_object = sha256(state_bytes)
    hash_digest = hash_object.hexdigest()
    # 将哈希值转为整数
    hash_int = int(hash_digest, 16)
    return hash_int % table_size


def calculate_reward(client, drone, distance_to_goal, last_distance_change, last_action, steps_same_action, last_min_distance_to_obstacle):
    reward = Config.REWARD_STEP
    position = client.getMultirotorState(vehicle_name=drone).kinematics_estimated.position
    new_distance_to_goal = calculate_distance_to_goal([position.x_val, position.y_val, position.z_val])

    # 计算接近目标点的奖励或惩罚
    if new_distance_to_goal < distance_to_goal:
        approach_reward = Config.REWARD_APPROACH
        reward += approach_reward
        print(f"接近目标点，增加奖励: {approach_reward}")
    else:
        approach_penalty = Config.REWARD_APPROACH
        reward -= approach_penalty
        print(f"远离目标点，减少奖励: {approach_penalty}")

    # 计算距离变化的差值
    distance_change = distance_to_goal - new_distance_to_goal
    if distance_change > last_distance_change:
        speed_reward = Config.REWARD_DIRECTION  # 额外的加速奖励
        reward += speed_reward
        print(f"朝向目标点更快，增加奖励: {speed_reward}")
    else:
        speed_penalty = Config.REWARD_DIRECTION  # 额外的减速惩罚
        reward -= speed_penalty
        print(f"朝向目标点更慢，减少奖励: {speed_penalty}")

    # 碰撞惩罚
    if is_collision(client, drone):
        collision_penalty = Config.REWARD_COLLISION
        reward += collision_penalty
        print(f"碰撞，减少奖励: {collision_penalty}")
    elif new_distance_to_goal < Config.GOAL_THRESHOLD:
        goal_reward = Config.REWARD_GOAL
        reward += goal_reward
        print(f"到达目标点，增加奖励: {goal_reward}")
    else:
        # 方向奖励
        direction = np.arctan2(Config.GOAL_POSITION[1] - position.y_val, Config.GOAL_POSITION[0] - position.x_val)
        current_direction = np.arctan2(last_action[1], last_action[0])
        angle_diff = abs(direction - current_direction)
        direction_reward = Config.REWARD_DIRECTION * (np.pi - angle_diff) / np.pi
        reward += direction_reward
        print(f"方向奖励: {direction_reward}")

    # LiDAR 避障奖励和惩罚（使用原始LiDAR数据的最小距离差值）
    points = get_lidar_data(client, drone)
    if points.size > 0:
        distances = np.linalg.norm(points, axis=1)
        min_distance_to_obstacle = np.min(distances)

        # 根据原始LiDAR数据的最小距离差值计算避障奖励和惩罚
        if min_distance_to_obstacle < last_min_distance_to_obstacle:
            obstacle_penalty = 8  # 调整这个值以合适的惩罚强度
            reward -= obstacle_penalty
            print(f"靠近障碍物，减少奖励: {obstacle_penalty}")
        elif min_distance_to_obstacle > last_min_distance_to_obstacle and min_distance_to_obstacle <= Config.SAFE_DISTANCE:
            obstacle_avoidance_reward = Config.REWARD_AVOID_OBSTACLE  # 调整这个值以合适的奖励强度
            reward += obstacle_avoidance_reward
            print(f"远离障碍物，增加奖励: {obstacle_avoidance_reward}")

        print(f"最小障碍物距离: {min_distance_to_obstacle}")
    else:
        min_distance_to_obstacle = last_min_distance_to_obstacle

    print(f"总奖励: {reward}")
    return reward, distance_change, min_distance_to_obstacle # 返回新的距离变化和最小障碍物距离


def train(client, drones: list[str]) -> None:
    global Q
    epsilon = Config.EPSILON
    log_file = os.path.join('C:\\Python project\\test_AirSim', 'training_log.txt')  # 设置日志文件路径

    # 清空之前的日志文件内容
    with open(log_file, 'w') as f:
        f.write("Episode, Drone, Total Reward, Loss, Collision Count\n")

    for episode in range(Config.NUM_EPISODES):
        episode_total_loss = 0  # 初始化本次训练的总损失
        collision_count = {drone: 0 for drone in drones}  # 初始化碰撞次数

        for drone in drones:
            client.reset()
            client.enableApiControl(True, vehicle_name=drone)
            client.armDisarm(True, vehicle_name=drone)
            client.takeoffAsync(vehicle_name=drone).join()
            client.moveToZAsync(-2, 1, vehicle_name=drone).join()

        states = {drone: get_state(client, drone) for drone in drones}
        total_rewards = {drone: 0 for drone in drones}
        last_actions = {drone: np.array([0, 0, 0, 0]) for drone in drones}
        steps_same_action = {drone: 0 for drone in drones}
        distance_to_goal = {drone: calculate_distance_to_goal([
            client.getMultirotorState(vehicle_name=drone).kinematics_estimated.position.x_val,
            client.getMultirotorState(vehicle_name=drone).kinematics_estimated.position.y_val,
            client.getMultirotorState(vehicle_name=drone).kinematics_estimated.position.z_val]) for drone in drones}
        last_distance_change = {drone: 0 for drone in drones}  # 初始化上一次距离变化
        last_min_distance_to_obstacle = {drone: np.inf for drone in drones}  # 初始化上一次最小障碍物距离
        done = {drone: False for drone in drones}
        steps = {drone: 0 for drone in drones}  # 初始化步数

        while not all(done.values()):
            for drone in drones:
                if done[drone]:
                    continue

                state = states[drone]
                state_index = hash_state(state, Q.shape[0] - len(Config.ACTIONS))

                if random.uniform(0, 1) < epsilon:
                    action_index = random.randint(0, len(Config.ACTIONS) - 1)
                else:
                    action_index = int(np.argmax(Q[state_index:state_index + len(Config.ACTIONS)]))

                action = Config.ACTIONS[action_index]

                take_action(client, drone, action)

                next_state = get_state(client, drone)
                next_state_index = hash_state(next_state, Q.shape[0] - len(Config.ACTIONS))

                # 计算新的距离到目标点
                new_distance_to_goal = calculate_distance_to_goal([
                    client.getMultirotorState(vehicle_name=drone).kinematics_estimated.position.x_val,
                    client.getMultirotorState(vehicle_name=drone).kinematics_estimated.position.y_val,
                    client.getMultirotorState(vehicle_name=drone).kinematics_estimated.position.z_val]
                )

                # 计算奖励并获取新的距离变化和最小障碍物距离
                reward, distance_change, min_distance_to_obstacle = calculate_reward(
                    client,
                    drone,
                    distance_to_goal[drone],
                    last_distance_change[drone],
                    last_actions[drone],
                    steps_same_action[drone],
                    last_min_distance_to_obstacle[drone]
                )

                # 更新 distance_to_goal、last_distance_change 和 last_min_distance_to_obstacle
                distance_to_goal[drone] = new_distance_to_goal
                last_distance_change[drone] = distance_change
                last_min_distance_to_obstacle[drone] = min_distance_to_obstacle

                if action_index == last_actions[drone][0]:
                    steps_same_action[drone] += 1
                else:
                    steps_same_action[drone] = 0
                    last_actions[drone] = np.array([action_index, 0])

                total_rewards[drone] += reward
                old_q_value = Q[state_index + action_index]  # 记录旧的 Q 值
                Q[state_index + action_index] = Q[state_index + action_index] + Config.ALPHA * (
                        reward + Config.GAMMA * np.max(Q[next_state_index:next_state_index + len(Config.ACTIONS)]) - Q[
                    state_index + action_index])

                # 计算 loss 并累加
                loss = np.abs(old_q_value - Q[state_index + action_index])
                episode_total_loss += loss

                states[drone] = next_state

                steps[drone] += 1

                # 处理碰撞逻辑
                if is_collision(client, drone):
                    collision_count[drone] += 1
                    print(f"Drone {drone} has collided {collision_count[drone]} times.")
                    if collision_count[drone] >= 3:
                        done[drone] = True
                        print(f"Drone {drone} training stopped after {collision_count[drone]} collisions.")

                if new_distance_to_goal < Config.GOAL_THRESHOLD or steps[
                    drone] >= Config.MAX_STEPS:
                    done[drone] = True
                    if new_distance_to_goal < Config.GOAL_THRESHOLD:
                        print(f"Drone {drone} has reached the goal.")
                    else:
                        print(f"Drone {drone} reached maximum steps.")
                    break

        if epsilon > Config.EPSILON_MIN:
            epsilon *= Config.EPSILON_DECAY

        # 记录每一轮的总奖励、总损失和碰撞次数
        with open(log_file, 'a') as f:
            for drone in drones:
                f.write(
                    f"{episode + 1}, {drone}, {total_rewards[drone]}, {episode_total_loss}, {collision_count[drone]}\n")

        print(
            f"Episode {episode + 1}/{Config.NUM_EPISODES}, Total Rewards: {total_rewards}, Loss: {episode_total_loss}, Collision Count: {collision_count}")

    save_path = os.path.join('C:\\Python project\\test_AirSim', 'q_table.pkl')
    save_q_table(Q, save_path)
    print(f"Q-table saved to {save_path}")
