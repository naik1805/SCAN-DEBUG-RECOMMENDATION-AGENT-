import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Tuple, List, Optional

# Require GPU — do not fall back to CPU
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required but not available. "
        "Install a CUDA-enabled PyTorch build (e.g. pip install torch --index-url https://download.pytorch.org/whl/cu130)."
    )
device = torch.device("cuda")

class QNetwork(nn.Module):
    """
    Multilayer Perceptron mapping state vectors to Q-values for each action.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ReplayBuffer:
    """
    Replay Buffer storing transition experience tuples for off-policy DQN training.
    """
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.int64),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.bool_)
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network Agent containing Q-networks, training step optimization,
    and action recommendation selection.
    """
    def __init__(
        self,
        state_dim: int = 10,
        action_dim: int = 5,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 200
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Exploration rate params
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        
        # Policy and Target Networks
        self.policy_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        self.memory = ReplayBuffer(buffer_capacity)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Selects an action using Epsilon-Greedy strategy.
        If evaluate=True, selects the action greedily (argmax Q) without exploration.
        """
        self.steps_done += 1
        
        # Epsilon decay
        if not evaluate:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
        # Select action
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                return int(q_values.argmax(dim=1).item())

    def get_action_confidences(self, state: np.ndarray) -> np.ndarray:
        """
        Computes Q-values for all actions and soft-normalizes them using Softmax
        to output confidence probabilities.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy().flatten()
            
        # Perform softmax over Q-values for readability as confidence metrics
        # Subtracting max for numerical stability
        exp_q = np.exp(q_values - np.max(q_values))
        confidences = exp_q / np.sum(exp_q)
        return confidences

    def optimize_model(self) -> Optional[float]:
        """
        Performs one gradient descent step over a sampled batch.
        """
        if len(self.memory) < self.batch_size:
            return None
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to torch tensors
        state_batch = torch.FloatTensor(states).to(device)
        action_batch = torch.LongTensor(actions).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_state_batch = torch.FloatTensor(next_states).to(device)
        done_batch = torch.FloatTensor(dones.astype(np.float32)).unsqueeze(1).to(device)
        
        # Get Q-values for current actions taken
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Get max Q-value predictions for next states from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(dim=1, keepdim=True)[0]
            # If terminal (done), the expected target is just the immediate reward
            target_q_values = reward_batch + (1.0 - done_batch) * self.gamma * next_q_values
            
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # Optimize weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Periodically update the target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return float(loss.item())

    def pretrain_supervised(self, states: List[np.ndarray], actions: List[int], epochs: int = 150):
        """
        Pre-trains the agent's policy network using supervised learning on a given dataset.
        Maps each state vector to its correct action with a high Q-value target.
        """
        states_t = torch.FloatTensor(np.array(states, dtype=np.float32)).to(device)
        actions_t = torch.LongTensor(actions).to(device)
        
        optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        
        self.policy_net.train()
        for epoch in range(epochs):
            q_values = self.policy_net(states_t)
            
            # Construct target Q-values: baseline of -10.0, target action set to +100.0
            targets = torch.full_like(q_values, -10.0)
            targets.scatter_(1, actions_t.unsqueeze(1), 100.0)
            
            loss = loss_fn(q_values, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Synchronize target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.eval()

    def save(self, filepath: str):
        """
        Saves the training weights to a file.
        """
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, filepath)

    def load(self, filepath: str):
        """
        Loads training weights from a file.
        """
        checkpoint = torch.load(filepath, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.steps_done = checkpoint.get('steps_done', self.steps_done)
