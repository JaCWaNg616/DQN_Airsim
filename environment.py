import airsim
import numpy as np
import time
from config import Config


class AirSimEnvironment:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.action_size = len(Config.ACTIONS)  # 使用新的动作空间大小
        self.state_size = 3 + 6 + 8 + 3

        # 初始化保存变量
        self.prev_distance_to_goal = None
        self.prev_distance_change = None
        self.prev_min_distance_to_obstacle = None

    def _get_state(self):
        # 获取位置数据
        position = self.client.getMultirotorState().kinematics_estimated.position
        state = [position.x_val, position.y_val, position.z_val]

        # 获取IMU数据
        imu_data = self.client.getImuData(vehicle_name="Drone1")
        state.extend([
            imu_data.linear_acceleration.x_val,
            imu_data.linear_acceleration.y_val,
            imu_data.linear_acceleration.z_val,
            imu_data.angular_velocity.x_val,
            imu_data.angular_velocity.y_val,
            imu_data.angular_velocity.z_val
        ])

        # 获取Lidar数据，并计算8个扇区的最小距离
        lidar_data = self.client.getLidarData(lidar_name="LidarSensor1", vehicle_name="Drone1")
        if len(lidar_data.point_cloud) > 0:
            points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
            distances = np.linalg.norm(points[:, :2], axis=1)  # 只考虑xy平面距离
            sector_size = len(distances) // 8
            lidar_state = []
            for i in range(8):
                sector_distances = distances[i * sector_size:(i + 1) * sector_size]
                if sector_distances.size > 0:
                    sector_min_distance = np.min(sector_distances)
                else:
                    sector_min_distance = 8.0  # 假设最大距离为8米，可以根据实际情况调整
                lidar_state.append(sector_min_distance)
            state.extend(lidar_state)
        else:
            state.extend([8] * 8)  # 如果没有点云数据，假设所有方向上没有障碍物

        # 获取速度数据
        velocity = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        state.extend([velocity.x_val, velocity.y_val, velocity.z_val])

        return np.array(state)

    def _compute_reward(self, state):
        goal = np.array(Config.GOAL_POSITION)
        current_position = state[:3]  # 假设 state 的前 3 个元素是 x, y, z 位置
        distance_to_goal = np.linalg.norm(state[:3] - goal)
        #print(f"当前位置: ({state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}), 距离目标的距离: {distance_to_goal:.2f}")

        reward = 0.0
        distance_change = 0  # 初始化 distance_change

        # 1. 与目标点距离的变化
        if self.prev_distance_to_goal is not None:
            distance_change = self.prev_distance_to_goal - distance_to_goal
            if distance_change > 0:
                reward_distance = Config.REWARD_DISTANCE_REDUCTION
                #print(f"奖励: 接近目标，距离减少 {distance_change:.2f}, 奖励 {reward_distance}")
            else:
                reward_distance = Config.PENALTY_DISTANCE_INCREASE
                #print(f"惩罚: 远离目标，距离增加 {abs(distance_change):.2f}, 惩罚 {reward_distance}")
            reward += reward_distance
        self.prev_distance_to_goal = distance_to_goal

        # 2. 与目标点距离变化的速度
        if self.prev_distance_change is not None:
            if distance_change > self.prev_distance_change:
                reward_speed = Config.REWARD_SPEED_INCREASE
                #print(f"奖励: 距离变化加快, 奖励 {reward_speed}")
            else:
                reward_speed = Config.PENALTY_SPEED_DECREASE
                #print(f"惩罚: 距离变化减慢, 惩罚 {reward_speed}")
            reward += reward_speed
        self.prev_distance_change = distance_change

        # 3. 方向奖励
        velocity = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        goal_direction = np.arctan2(Config.GOAL_POSITION[1] - state[1], Config.GOAL_POSITION[0] - state[0])
        current_direction = np.arctan2(velocity.y_val, velocity.x_val)
        angle_diff = abs(goal_direction - current_direction)

        # 确保角度差异在 [0, pi] 范围内
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff

        if angle_diff <= np.pi / 4:
            direction_reward = Config.REWARD_DIRECTION_CORRECT
            reward += direction_reward
            #print(f"奖励: 方向正确, 奖励 {direction_reward:.2f}")
        else:
            direction_penalty = Config.PENALTY_DIRECTION_WRONG
            reward += direction_penalty
            #print(f"惩罚: 方向偏离, 惩罚 {direction_penalty:.2f}")

        # 4. 与障碍物距离的奖励
        lidar_data = self.client.getLidarData(lidar_name="LidarSensor1", vehicle_name="Drone1")
        if len(lidar_data.point_cloud) > 0:
            points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
            min_distance_to_obstacle = np.min(np.linalg.norm(points[:, :2], axis=1))

            if self.prev_min_distance_to_obstacle is not None:
                if min_distance_to_obstacle > self.prev_min_distance_to_obstacle:
                    reward_obstacle = Config.REWARD_OBSTACLE_AVOIDANCE
                    #print(f"奖励: 远离障碍物, 奖励 {reward_obstacle}")
                elif min_distance_to_obstacle < self.prev_min_distance_to_obstacle:
                    reward_obstacle = Config.PENALTY_OBSTACLE_APPROACH
                   # print(f"惩罚: 接近障碍物, 惩罚 {reward_obstacle}")
                else:
                    reward_obstacle = 0
                   # print(f"距离障碍物变化不大，无奖励或惩罚")
                reward += reward_obstacle

            self.prev_min_distance_to_obstacle = min_distance_to_obstacle

        # 5. 碰撞惩罚
        if self._is_collision():
            reward_collision = Config.PENALTY_COLLISION
            #print(f"惩罚: 碰撞检测, 惩罚 {reward_collision}")
            reward += reward_collision

        # 打印总奖励
        #print(f"总奖励: {reward:.2f}")


        return reward

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        self.client.moveToZAsync(-2, 1).join()

        # 重置保存变量
        self.prev_distance_to_goal = None
        self.prev_distance_change = None
        self.prev_min_distance_to_obstacle = None

        return self._get_state()

    def step(self, action):
        # 根据动作选择 quad_offset
        quad_offset = Config.ACTIONS[action]
        self.client.moveByVelocityAsync(quad_offset[0], quad_offset[1], quad_offset[2], quad_offset[3])

        start_time = time.time()
        duration = quad_offset[3]
        action_steps = 0  # 初始化动作步数
        done = False

        while time.time() - start_time < duration:
            state = self._get_state()
            reward = self._compute_reward(state)
            action_steps += 1  # 记录动作执行的步数

            # 稍作延迟以确保碰撞检测能够正确触发
            time.sleep(0.05)

            # 检查是否满足回合终止条件
            done = self._is_done(state)
            if done:
                break

            time.sleep(0.01)  # 添加一个短暂的sleep来避免CPU过度占用

        return state, reward, done

    def _is_done(self, state):

        if self._is_collision():
            return True

        # 检查是否达到目标点
        elif np.linalg.norm(state[:3] - np.array(Config.GOAL_POSITION)) < Config.GOAL_THRESHOLD:
            return True
        else:
            return False

    def _is_collision(self):
        collision_info = self.client.simGetCollisionInfo()
        return collision_info.has_collided
