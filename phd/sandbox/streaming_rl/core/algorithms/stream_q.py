import torch
import numpy as np
import torch.nn as nn

from phd.streaming_rl.core.layers import LayerNormalization
from phd.streaming_rl.core.param_init import sparse_init
from phd.streaming_rl.core.obgd import ObGD
from phd.streaming_rl.core.processing import linear_schedule


def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)


class StreamQ(nn.Module):
    def __init__(self, n_actions=3, hidden_size=256, lr=1.0, epsilon_target=0.01, epsilon_start=1.0, exploration_fraction=0.1, total_steps=1_000_000, gamma=0.99, lamda=0.8, kappa_value=2.0):
        super(StreamQ, self).__init__()
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_target = epsilon_target
        self.epsilon = epsilon_start
        self.exploration_fraction = exploration_fraction
        self.total_steps = total_steps
        self.time_step = 0
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=5),
            LayerNormalization(),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 4, stride=3),
            LayerNormalization(),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=2),
            LayerNormalization(),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=0),
            nn.Linear(256, hidden_size),
            LayerNormalization(),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, n_actions)
        )
        self.apply(initialize_weights)
        self.optimizer = ObGD(list(self.parameters()), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value)

    def q(self, x):
        x = torch.tensor(np.array(x), dtype=torch.float)
        return self.network(x)

    def sample_action(self, s):
        self.time_step += 1
        self.epsilon = linear_schedule(self.epsilon_start, self.epsilon_target, self.exploration_fraction * self.total_steps, self.time_step)
        if isinstance(s, np.ndarray):
            s = torch.tensor(np.array(s), dtype=torch.float)
        if np.random.rand() < self.epsilon:
            q_values = self.q(s)
            greedy_action = torch.argmax(q_values, dim=-1).item()
            random_action = np.random.randint(0, self.n_actions)
            if greedy_action == random_action:
                return random_action, False
            else:
                return random_action, True
        else:
            q_values = self.q(s)
            return torch.argmax(q_values, dim=-1), False

    def update_params(self, s, a, r, s_prime, done, is_nongreedy, overshooting_info=False):
        done_mask = 0 if done else 1
        s, a, r, s_prime, done_mask = torch.tensor(np.array(s), dtype=torch.float), torch.tensor([a], dtype=torch.int).squeeze(0), \
                                         torch.tensor(np.array(r)), torch.tensor(np.array(s_prime), dtype=torch.float), \
                                         torch.tensor(np.array(done_mask), dtype=torch.float)

        q_sa = self.q(s)[a]
        max_q_s_prime_a_prime = torch.max(self.q(s_prime), dim=-1).values
        td_target = r + self.gamma * max_q_s_prime_a_prime * done_mask
        delta = td_target - q_sa

        q_output = -q_sa
        self.optimizer.zero_grad()
        q_output.backward()
        self.optimizer.step(delta.item(), reset=(done or is_nongreedy))