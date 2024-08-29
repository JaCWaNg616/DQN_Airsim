import torch
import torch.nn as nn
import numpy as np
from airsim import MultirotorClient
from config import Config
import os

class TestDQNAgent:
    def __init__(self, state_size, action_size, model_path):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            act_values = self.model(state_tensor)
        return torch.argmax(act_values[0]).item()

def calculate_reward(state, next_state, collisions):
    distance_to_goal = np.linalg.norm(next_state[:3] - np.array(Config.GOAL_POSITION))
    prev_distance_to_goal = np.linalg.norm(state[:3] - np.array(Config.GOAL_POSITION))

    reward = 0
    if distance_to_goal < prev_distance_to_goal:
        reward += Config.REWARD_DISTANCE_REDUCTION
    else:
        reward += Config.PENALTY_DISTANCE_INCREASE

    if np.dot(next_state[:3] - state[:3], np.array(Config.GOAL_POSITION) - state[:3]) > 0:
        reward += Config.REWARD_DIRECTION_CORRECT
    else:
        reward += Config.PENALTY_DIRECTION_WRONG

    speed = np.linalg.norm(next_state[3:6])
    prev_speed = np.linalg.norm(state[3:6])
    if speed > prev_speed:
        reward += Config.REWARD_SPEED_INCREASE
    else:
        reward += Config.PENALTY_SPEED_DECREASE

    if collisions:
        reward += Config.PENALTY_COLLISION

    if distance_to_goal < Config.GOAL_THRESHOLD:
        reward += Config.REWARD_GOAL
        done = True
    else:
        done = False

    return reward, done

def run_simulation():
    client = MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()

    state_size = 20  # 确认状态空间维度为 20
    action_size = len(Config.ACTIONS)
    model_path = r"C:\Python project\DQN\train_model4 5000\dqn_model_final.pth"
    agent = TestDQNAgent(state_size, action_size, model_path)

    total_reward = 0
    steps = 0
    done = False
    collisions = False

    # 存储飞行轨迹
    flight_path = []

    def _get_state():
        position = client.getMultirotorState().kinematics_estimated.position
        state = [position.x_val, position.y_val, position.z_val]

        imu_data = client.getImuData(vehicle_name="Drone1")
        state.extend([
            imu_data.linear_acceleration.x_val,
            imu_data.linear_acceleration.y_val,
            imu_data.linear_acceleration.z_val,
            imu_data.angular_velocity.x_val,
            imu_data.angular_velocity.y_val,
            imu_data.angular_velocity.z_val
        ])

        lidar_data = client.getLidarData(lidar_name="LidarSensor1", vehicle_name="Drone1")
        if len(lidar_data.point_cloud) > 0:
            points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
            distances = np.linalg.norm(points[:, :2], axis=1)
            sector_size = len(distances) // 8
            lidar_state = []
            for i in range(8):
                sector_distances = distances[i * sector_size:(i + 1) * sector_size]
                if sector_distances.size > 0:
                    sector_min_distance = np.min(sector_distances)
                else:
                    sector_min_distance = 8.0
                lidar_state.append(sector_min_distance)
            state.extend(lidar_state)
        else:
            state.extend([8] * 8)

        velocity = client.getMultirotorState().kinematics_estimated.linear_velocity
        state.extend([velocity.x_val, velocity.y_val, velocity.z_val])

        return np.array(state), [position.x_val, position.y_val, position.z_val]

    state, position = _get_state()
    flight_path.append(position)

    while not done:
        action_idx = agent.act(state)
        action = Config.ACTIONS[action_idx]

        client.moveByVelocityAsync(action[0], action[1], action[2], action[3]).join()

        next_state, position = _get_state()
        flight_path.append(position)

        collision_info = client.simGetCollisionInfo()
        if collision_info.has_collided:
            collisions = True
        else:
            collisions = False

        reward, done = calculate_reward(state, next_state, collisions)

        total_reward += reward
        steps += 1

        if steps >= Config.MAX_TIMESTEPS:
            done = True

        state = next_state

    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)

    # 保存飞行轨迹
    os.makedirs("flight_paths", exist_ok=True)
    flight_path_file = os.path.join("flight_paths", "flight_path.npy")
    np.save(flight_path_file, np.array(flight_path))

    print(f"Simulation finished in {steps} steps with a total reward of {total_reward}. Path saved to {flight_path_file}.")

if __name__ == "__main__":
    run_simulation()
