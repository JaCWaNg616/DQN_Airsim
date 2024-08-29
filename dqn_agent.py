import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim  # 添加优化器的导入
from config import Config

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=Config.MEMORY_SIZE)
        self.gamma = Config.GAMMA
        self.epsilon = Config.EPSILON
        self.epsilon_min = Config.EPSILON_MIN
        self.epsilon_decay = Config.EPSILON_DECAY
        self.learning_rate = Config.ALPHA

        # 创建主模型和目标模型
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        # 定义优化器和损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
    def _build_model(self):
        """ 构建神经网络模型 """
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model

    def update_target_model(self):
        """ 将主模型的权重复制到目标模型 """
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """ 存储回忆 """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """ 选择动作，使用 ε-greedy 策略 """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # 随机选择一个动作
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # 添加一个 batch 维度
        act_values = self.model(state_tensor)
        return torch.argmax(act_values[0]).item()  # 返回 Q 值最大的动作

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        total_loss = 0
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            self.model.eval()
            with torch.no_grad():
                q_values = self.model(state_tensor).squeeze(1)  # 去除多余的维度
                target_f = q_values.clone().cpu().numpy().squeeze()

            target = reward
            if not done:
                q_values_next = self.target_model(next_state_tensor).squeeze(1)  # 去除多余的维度
                target = reward + self.gamma * torch.max(q_values_next).item()

            if 0 <= action < target_f.shape[0]:
                target_f[action] = target
            else:
                print(f"Warning: Action {action} is out of bounds for target_f with shape {target_f.shape}")

            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(state_tensor).squeeze(1)  # 去除多余的维度
            target_f_tensor = torch.tensor(target_f, dtype=torch.float32).unsqueeze(0)  # 扩展维度
            loss = self.criterion(output, target_f_tensor)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / batch_size

    def load(self, name):
        """ 加载模型权重 """
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        """ 保存模型权重 """
        torch.save(self.model.state_dict(), name)
