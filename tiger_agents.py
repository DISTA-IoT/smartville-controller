from smartController.neural_modules import DQN
import torch.optim as optim
from collections import deque
import torch
import random
import torch.nn as nn

class DDQNAgent:

    def __init__(self, state_size, action_size, kwargs):

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.replay_batch_size = kwargs['replay_batch_size'] 

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state):

        if torch.rand(1).item() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.model(state)
        return q_values.max(0)[1].item()


    def replay(self):
        if len(self.memory) < self.replay_batch_size:
            return
        
        minibatch = random.sample(self.memory, self.replay_batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            
            if not done:  
                target += self.gamma * torch.max(self.target_model(next_state)).item()
            
            target_f = self.model(state).detach()
            target_f[action] = target
            
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
