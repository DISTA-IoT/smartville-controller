from smartController.neural_modules import DQN, PolicyNet, EFENet
import torch.optim as optim
from collections import deque
import torch
import random
import torch.nn as nn
import torch.distributions as distributions
import torch.functional as F


class DAIAgent:
    def __init__(self, kwargs):
        
        self.efe_net = EFENet(kwargs['state_size'], kwargs['action_size'])
        self.target_efe_net = EFENet(kwargs['state_size'], kwargs['action_size'])
        self.update_target_efe_net()
        self.efe_net_optimizer = optim.Adam(self.efe_net.parameters(), lr=kwargs['lr'])
                
        self.policynet = PolicyNet(kwargs['state_size'], kwargs['action_size'])
        self.policynet_optimizer = optim.Adam(self.policynet.parameters(), lr=kwargs['lr'])

        self.memory = deque(maxlen=kwargs['memory_size'])
        self.replay_batch_size = kwargs['replay_batch_size']

        self.value_loss_fn = nn.MSELoss()


    def update_target_efe_net(self):
        self.target_efe_net.load_state_dict(self.efe_net.state_dict())


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state):

        action_probs = self.policynet(state)
        
        # sample from a categorical distribution 
        m = distributions.Categorical(action_probs)
        action = m.sample().item()

        return action
    

    def train_actor(self, states):
        """
        Trains the actor (policy network) by minimising the VFE
        WE do not have a POMDP ut only an MDP, so we do not minimise accuracy over observations.
        HOWEVER, we are missing the entropy term here... @TODO implement entropy regularization here.
        """
        policy_probabilities = self.policynet(states)

        estimated_efe_values = self.efe_net(states).detach()
        efe_actions = F.log_softmax(estimated_efe_values, dim=1)
        expected_logprobs = torch.sum(policy_probabilities * efe_actions, dim=1)
        critic_losses = -expected_logprobs.mean(dim=1)

        self.policynet_optimizer.zero_grad()
        critic_loss = critic_losses.mean()
        critic_loss.backward()
        self.policynet_optimizer.step()


    def replay(self):
        """
        Use temporal difference on expected free energy to update the critic (efe bootstrapped network)
        In this case, we do not have a transition model to update. we just update the efe network
        """
        if len(self.memory) < self.replay_batch_size:
            return
        
        minibatch = random.sample(self.memory, self.replay_batch_size)

        for state, action, reward, next_state, done in minibatch:

            target = reward
            
            if not done:
                
                # Select action using online network
                policy_probabilities = self.policynet(next_state)
                estimated_EFE_values = self.efe_net(next_state)

                # Compute efe value under the current policy
                expected_value = torch.sum(policy_probabilities * estimated_EFE_values, dim=1)

                # Update target
                target += 0.99 * expected_value.detach()

            target_f = self.efe_net(state).detach()
            target_f[action] = target

            self.efe_net_optimizer.zero_grad()
            predicted_values = self.efe_net(state)
            value_losses = self.value_loss_fn(predicted_values, target_f)
            value_losses.backward()
            self.efe_net_optimizer.step()
    

class ValueLearningAgent:

    def __init__(self, state_size, action_size, replay_batch_size, kwargs):

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = float((kwargs['greedy_decay'] if 'greedy_decay' in kwargs else 0.995))
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.replay_batch_size = replay_batch_size
        self.algorithm = (kwargs['agent'] if 'agent' in kwargs else 'DQN') 

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

                if self.algorithm == 'DQN':  
                    # take the max Q-value from target network
                    target += self.gamma * torch.max(self.target_model(next_state)).item()
                elif self.algorithm == 'DDQN':
                    # Select action using online network
                    next_action = self.model(next_state).max(0)[1].item()
                    # Evaluate using target network
                    target += self.gamma * self.target_model(next_state)[next_action].item()  
        
            
            target_f = self.model(state).detach()
            target_f[action] = target


            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
