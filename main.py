import numpy as np
import csv
from environment import AirSimEnvironment
from dqn_agent import DQNAgent
from config import Config

def main():
    env = AirSimEnvironment()
    state_size = env.state_size
    action_size = env.action_size
    agent = DQNAgent(state_size, action_size)

    # 定义阶段和碰撞容忍度
    stage_boundaries = [2000, 4000, 6000, 8000, 10000]
    collision_tolerances = [5, 4, 3, 2, 1]

    # 打开CSV文件准备写入
    with open('training_log.csv', 'w', newline='') as csvfile:
        fieldnames = ['Episode', 'Epsilon', 'Total Reward', 'Loss', 'Steps', 'Collisions', 'Exit Reason']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for e in range(Config.NUM_EPISODES):
            # 确定当前阶段的碰撞容忍度
            for stage, boundary in enumerate(stage_boundaries):
                if e < boundary:
                    max_collisions = collision_tolerances[stage]
                    break

            state = env.reset()
            state = np.reshape(state, [1, state_size])
            total_reward = 0
            total_loss = 0
            steps = 0
            collisions = 0

            while True:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                if done:
                    if np.linalg.norm(state[0][:3] - np.array(Config.GOAL_POSITION)) < Config.GOAL_THRESHOLD:
                        exit_reason = "Goal reached"
                        break
                    else:
                        collisions += 1
                        print(f"碰撞次数: {collisions}")
                next_state = np.reshape(next_state, [1, state_size])
                agent.remember(state, action, reward, next_state, done)
                state = next_state

                total_reward += reward
                steps += 1  # 记录整个回合的总步数
                #print(f"当前步数: {steps}")
                #print("\n")

                if len(agent.memory) > Config.BATCH_SIZE:
                    loss = agent.replay(Config.BATCH_SIZE)
                    if loss is not None:
                        total_loss += loss


                if np.linalg.norm(state[0][:3] - np.array(Config.GOAL_POSITION)) < Config.GOAL_THRESHOLD:
                    exit_reason = "Goal reached"
                    break
                elif steps >= Config.MAX_TIMESTEPS:
                    exit_reason = "Max steps reached"
                    break
                elif collisions > max_collisions:
                    exit_reason = "Collision tolerance exceeded"
                    break

            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay
            # 打印并记录当前回合的参数
            agent.update_target_model()
            print(f"Episode: {e+1}/{Config.NUM_EPISODES}, Score: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}, Collisions: {collisions}, Steps: {steps}, Exit Reason: {exit_reason}")
            writer.writerow({
                'Episode': e + 1,
                'Epsilon': agent.epsilon,
                'Total Reward': total_reward,
                'Loss': total_loss,
                'Steps': steps,
                'Collisions': collisions,
                'Exit Reason': exit_reason
            })

    print("Training data has been logged to 'training_log.csv'.")

if __name__ == "__main__":
    main()
